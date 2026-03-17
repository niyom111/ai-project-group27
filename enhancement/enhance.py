"""
Lighting enhancement module — histogram equalization, CLAHE, gamma (Person 1).
"""
import cv2
import os
import config


def enhance_image(image_path):
    """
    Load an image, apply histogram equalization + CLAHE (L channel), and save to ENHANCED_DIR.

    Args:
        image_path: Path to the input image file.

    Returns:
        Path to the saved image under config.ENHANCED_DIR, or None if load failed.
    """
    image = cv2.imread(image_path)
    # TODO: replace with proper error handling (e.g. raise ValueError or log and return)
    if image is None:
        return None  # arbitrary: replace with something real later (e.g. raise or return error path)

    # Histogram equalization on luminance only (LAB L channel) so colors stay natural
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    # CLAHE on L channel (per-tile contrast; good for shadows and uneven light)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_eq)
    lab_clahe = cv2.merge([l_clahe, a, b])
    image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    os.makedirs(config.ENHANCED_DIR, exist_ok=True)
    output_path = os.path.join(config.ENHANCED_DIR, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    return output_path
