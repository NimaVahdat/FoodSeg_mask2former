import os
import json
import random
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from huggingface_hub import hf_hub_download
import albumentations as A


def color_palette(num_classes=150, seed=85):
    """
    Generates a consistent color palette for a given number of classes by setting a random seed.

    Args:
        num_classes (int): Number of classes/colors to generate.
        seed (int): Seed for the random number generator.

    Returns:
        list: A list of RGB values.
    """
    random.seed(seed)
    palette = []
    for _ in range(num_classes):
        color = [random.randint(0, 255) for _ in range(3)]
        palette.append(color)
    return palette


def load_model_and_processor(device):
    """
    Loads the Mask2Former model and processor from the latest checkpoint in the specified directory.

    Args:
        device (str): Device to load the model onto (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    directory_path = "path_to_save_model"
    all_files = os.listdir(directory_path)
    sorted_files = sorted(all_files)
    saved_model_path = os.path.join(directory_path, sorted_files[-1])

    model = Mask2FormerForUniversalSegmentation.from_pretrained(saved_model_path).to(
        device
    )
    processor = Mask2FormerImageProcessor(
        ignore_index=0,
        reduce_labels=False,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )
    return model, processor


def visualize_panoptic_segmentation(
    original_image_np, segmentation_mask, segments_info, category_names
):
    """
    Visualizes the segmentation mask overlaid on the original image with category labels.

    Args:
        original_image_np (np.ndarray): The original image in NumPy array format.
        segmentation_mask (np.ndarray): The segmentation mask.
        segments_info (list): Information about the segments.
        category_names (list): List of category names corresponding to segment IDs.

    Returns:
        PIL.Image.Image: The overlayed image with segmentation mask and labels.
    """
    # Create a blank image for the segmentation mask
    segmentation_image = np.zeros_like(original_image_np)

    num_classes = len(category_names)
    palette = color_palette(num_classes)

    # Apply colors to the segmentation mask
    for segment in segments_info:
        if segment["label_id"] == 0:
            continue
        color = palette[segment["label_id"]]
        mask = segmentation_mask == segment["id"]
        segmentation_image[mask] = color

    # Overlay the segmentation mask on the original image
    alpha = 0.5  # Transparency for the overlay
    overlay_image = cv2.addWeighted(
        original_image_np, 1 - alpha, segmentation_image, alpha, 0
    )

    # Convert to PIL image for text drawing
    overlay_image_pil = Image.fromarray(overlay_image)
    draw = ImageDraw.Draw(overlay_image_pil)

    # Set up font size
    base_font_size = max(
        20, int(min(original_image_np.shape[0], original_image_np.shape[1]) * 0.015)
    )

    # Optional: Load custom font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", base_font_size)
    except IOError:
        raise RuntimeError(
            "Custom font not found. Please ensure the font file is available."
        )

    # Draw category labels on the image
    for segment in segments_info:
        label_id = segment.get("label_id")
        if label_id is not None and 0 <= label_id < len(category_names):
            category = category_names[label_id]
            mask = (segmentation_mask == segment["id"]).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            if num_labels > 1:
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                centroid_x = int(centroids[largest_component][0])
                centroid_y = int(centroids[largest_component][1])

                # Ensure text is within image bounds
                text_position = (
                    max(0, min(centroid_x, original_image_np.shape[1] - 1)),
                    max(0, min(centroid_y, original_image_np.shape[0] - 1)),
                )
                draw.text(text_position, category, fill=(0, 0, 0), font=font)

    return overlay_image_pil


def predict_masks(input_image_path):
    """
    Predicts and visualizes segmentation masks for a given image.

    Args:
        input_image_path (str): Path to the input image.

    Returns:
        PIL.Image.Image: The image with overlaid segmentation mask and labels.
    """
    # Determine device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor(device)

    # Load category labels
    repo_id = "EduardoPacheco/FoodSeg103"
    filename = "id2label.json"
    id2label = json.load(
        open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r")
    )
    id2label = {int(k): v for k, v in id2label.items()}

    # Load and preprocess image
    image_PIL = Image.open(input_image_path)
    original_image_np = np.array(image_PIL)

    transform = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transformed = transform(image=original_image_np)
    image = transformed["image"]

    # Convert image to C, H, W format
    image = image.transpose(2, 0, 1)

    # Process the image and get predictions
    inputs = processor([image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image_PIL.size[::-1]]
    )[0]
    segmentation_mask = result["segmentation"].cpu().numpy()

    segments_info = result["segments_info"]
    output_result = visualize_panoptic_segmentation(
        original_image_np, segmentation_mask, segments_info, id2label
    )

    return output_result

