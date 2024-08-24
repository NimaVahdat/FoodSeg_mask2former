import numpy as np
from torch.utils.data import Dataset


class ImageSegmentationDataset(Dataset):
    """Custom dataset for image segmentation tasks."""

    def __init__(self, dataset, transform=None, is_train=True):
        """
        Args:
            dataset (Dataset): The dataset containing image and segmentation map pairs.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool, optional): Flag indicating if the dataset is for training. Defaults to True.
        """
        self.dataset = dataset
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Transformed image, transformed segmentation map, original image, original segmentation map.
        """
        # Attempt to retrieve the image; fallback to 'image' key if 'pixel_values' key is unavailable.
        try:
            original_image = np.array(self.dataset[idx]["pixel_values"])
        except KeyError:
            original_image = np.array(self.dataset[idx]["image"])

        # Retrieve the corresponding segmentation map.
        original_segmentation_map = np.array(self.dataset[idx]["label"])

        # Apply transformations (if any) to the image and segmentation map.
        transformed = self.transform(
            image=original_image, mask=original_segmentation_map
        )
        image, segmentation_map = transformed["image"], transformed["mask"]

        # Convert image from H, W, C format to C, H, W format (required by PyTorch).
        image = image.transpose(2, 0, 1)

        return image, segmentation_map, original_image, original_segmentation_map
