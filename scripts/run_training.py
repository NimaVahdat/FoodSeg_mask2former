import yaml
from scripts.train import SegmentationTrainer
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
import json


def load_config(config_file):
    """
    Load configuration settings from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config("config.yaml")

    # Specify the repository ID and the filename for id2label mapping
    repo_id = "EduardoPacheco/FoodSeg103"
    filename = "id2label.json"

    # Download and load id2label mapping from the Hugging Face Hub
    id2label_path = hf_hub_download(repo_id, filename, repo_type="dataset")
    with open(id2label_path, "r") as file:
        id2label = json.load(file)

    # Convert keys to integers
    id2label = {int(k): v for k, v in id2label.items()}
    print(id2label)

    # Load training and validation datasets from the Hugging Face Hub
    train_dataset = load_dataset("EduardoPacheco/FoodSeg103", split="train")
    val_dataset = load_dataset("EduardoPacheco/FoodSeg103", split="validation")

    # Initialize the SegmentationTrainer with loaded configuration and datasets
    trainer = SegmentationTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        id2label=id2label,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        epochs=config["epochs"],
        save_path=config["save_path"],
        load_checkpoint=config["load_checkpoint"],
        log_dir=config["log_dir"],
    )

    # Start training the model
    trainer.train()
