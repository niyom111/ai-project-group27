"""
Lighting enhancement module — load/save I/O (Step 1).
Enhancement steps (histogram eq, CLAHE, gamma) will be added in later steps.
"""
import cv2
import os
import config


def enhance_image(image_path):
    """
    Load an image from disk and save it to the enhanced output directory.
    No enhancement applied yet (Step 1 wiring only).

    Args:
        image_path: Path to the input image file.

    Returns:
        Path to the saved image under config.ENHANCED_DIR, or None if load failed.
    """
    image = cv2.imread(image_path)
    # TODO: replace with proper error handling (e.g. raise ValueError or log and return)
    if image is None:
        return None  # arbitrary: replace with something real later (e.g. raise or return error path)

    os.makedirs(config.ENHANCED_DIR, exist_ok=True)
    output_path = os.path.join(config.ENHANCED_DIR, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    return output_path
