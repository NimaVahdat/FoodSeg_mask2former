import torch
from torch.utils.data import DataLoader
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import evaluate
import albumentations as A
from Data import ImageSegmentationDataset


class SegmentationTrainer:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        id2label,
        batch_size=4,
        lr=5e-5,
        epochs=10,
        save_path="model_checkpoint",
        load_checkpoint=None,
        log_dir="logs",
    ):
        """
        Args:
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            id2label (dict): Mapping of label IDs to label names.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 4.
            lr (float, optional): Learning rate. Defaults to 5e-5.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            save_path (str, optional): Directory to save model checkpoints. Defaults to "model_checkpoint".
            load_checkpoint (str, optional): Path to load a pre-trained model checkpoint. Defaults to None.
            log_dir (str, optional): Directory to save TensorBoard logs. Defaults to "logs".
        """
        self.id2label = id2label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize image processor
        self.processor = Mask2FormerImageProcessor(
            ignore_index=0,
            reduce_labels=False,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )

        # Define data augmentation and normalization transforms
        train_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=1333),
                A.Resize(width=512, height=512),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        test_transform = A.Compose(
            [
                A.Resize(width=512, height=512),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Initialize datasets and DataLoader
        self.train_dataset = ImageSegmentationDataset(
            dataset=train_dataset, transform=train_transform
        )
        self.val_dataset = ImageSegmentationDataset(
            dataset=val_dataset, transform=test_transform
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        # Initialize model
        self.model = self.get_model(load_checkpoint)
        self.model.to(self.device)

        # Initialize optimizer and learning rate scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=3,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            verbose=True,
        )

        # Initialize metric for evaluation
        self.metric = evaluate.load("mean_iou")
        self.epochs = epochs
        self.save_path = save_path

        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def get_model(self, load_checkpoint):
        """Load the model, either from a checkpoint or by initializing a new model."""
        if load_checkpoint is not None:
            print("Loading Checkpoint!")
            model = Mask2FormerForUniversalSegmentation.from_pretrained(load_checkpoint)
        else:
            # Load and configure the model
            config = Mask2FormerConfig.from_pretrained(
                "facebook/mask2former-swin-small-ade-semantic"
            )
            config.id2label = self.id2label
            config.label2id = {label: idx for idx, label in self.id2label.items()}

            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-small-ade-semantic",
                config=config,
                ignore_mismatched_sizes=True,
            )

        return model

    def collate_fn(self, batch):
        """Collate function to process batches of data."""
        inputs = list(zip(*batch))
        images = inputs[0]
        segmentation_maps = inputs[1]
        batch = self.processor(
            images,
            segmentation_maps=segmentation_maps,
            return_tensors="pt",
        )
        batch["original_images"] = inputs[2]
        batch["original_segmentation_maps"] = inputs[3]

        return batch

    def train(self):
        """Train the model."""
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for idx, batch in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(
                    pixel_values=batch["pixel_values"].to(self.device),
                    mask_labels=[
                        labels.to(self.device) for labels in batch["mask_labels"]
                    ],
                    class_labels=[
                        labels.to(self.device) for labels in batch["class_labels"]
                    ],
                )

                # Backward pass
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Log average training loss
            avg_train_loss = epoch_loss / len(self.train_loader)
            self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            print(f"Epoch {epoch+1} complete. Avg Training Loss: {avg_train_loss:.4f}")

            # Validation
            avg_val_loss = self.validate(epoch)
            self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Update learning rate based on validation loss
            self.scheduler.step(avg_val_loss)

            # Save the model if it improves
            self.save_model(epoch, avg_train_loss, avg_val_loss)

        self.writer.close()

    def validate(self, epoch):
        """Validate the model on the validation dataset."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.val_loader)):
                outputs = self.model(
                    pixel_values=batch["pixel_values"].to(self.device),
                    mask_labels=[
                        labels.to(self.device) for labels in batch["mask_labels"]
                    ],
                    class_labels=[
                        labels.to(self.device) for labels in batch["class_labels"]
                    ],
                )
                val_loss += outputs.loss.item()

                # Post-process the output for evaluation
                original_images = batch["original_images"]
                target_sizes = [
                    (image.shape[0], image.shape[1]) for image in original_images
                ]
                predicted_segmentation_maps = (
                    self.processor.post_process_semantic_segmentation(
                        outputs, target_sizes=target_sizes
                    )
                )

                # Add batch results to metric
                ground_truth_segmentation_maps = batch["original_segmentation_maps"]
                self.metric.add_batch(
                    references=ground_truth_segmentation_maps,
                    predictions=predicted_segmentation_maps,
                )

            # Compute and print mean IoU
            mean_iou = self.metric.compute(
                num_labels=len(self.id2label), ignore_index=0
            )["mean_iou"]
            print("Mean IoU:", mean_iou)

        avg_val_loss = val_loss / len(self.val_loader)
        return avg_val_loss

    def save_model(self, epoch, train_loss, val_loss):
        """Save the model if validation loss improves."""
        save_criteria = epoch == 0 or val_loss < getattr(
            self, "best_val_loss", float("inf")
        )
        if save_criteria:
            print(f"Saving model at epoch {epoch+1}")
            self.model.save_pretrained(f"{self.save_path}/epoch_{epoch+1}")
            self.best_val_loss = val_loss
