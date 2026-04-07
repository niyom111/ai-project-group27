"""
Image Loader Module
-------------------
Handles loading, validation, resizing, and metadata extraction for
input images fed into the enhancement pipeline.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from enhancement import enhancement_config as ecfg


class ImageMetadata:
    """Container for image metadata and statistical properties."""

    def __init__(self):
        self.file_path: str = ""
        self.file_name: str = ""
        self.file_extension: str = ""
        self.file_size_bytes: int = 0
        self.original_width: int = 0
        self.original_height: int = 0
        self.original_channels: int = 0
        self.resized_width: int = 0
        self.resized_height: int = 0
        self.mean_brightness: float = 0.0
        self.std_brightness: float = 0.0
        self.min_brightness: int = 0
        self.max_brightness: int = 255
        self.brightness_histogram: Optional[np.ndarray] = None
        self.histogram_skewness: float = 0.0
        self.histogram_kurtosis: float = 0.0
        self.dynamic_range: int = 0
        self.is_low_light: bool = False
        self.is_overexposed: bool = False
        self.is_low_contrast: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary for logging."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "file_size_bytes": self.file_size_bytes,
            "original_dimensions": (self.original_width, self.original_height),
            "original_channels": self.original_channels,
            "resized_dimensions": (self.resized_width, self.resized_height),
            "mean_brightness": round(self.mean_brightness, 2),
            "std_brightness": round(self.std_brightness, 2),
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "dynamic_range": self.dynamic_range,
            "histogram_skewness": round(self.histogram_skewness, 3),
            "histogram_kurtosis": round(self.histogram_kurtosis, 3),
            "is_low_light": self.is_low_light,
            "is_overexposed": self.is_overexposed,
            "is_low_contrast": self.is_low_contrast,
        }


def validate_image_path(image_path: str) -> bool:
    """Check that the image path exists and has a supported extension."""
    if not image_path:
        return False
    if not os.path.isfile(image_path):
        return False
    _, ext = os.path.splitext(image_path)
    if ext.lower() not in ecfg.SUPPORTED_EXTENSIONS:
        return False
    return True


def validate_image_dimensions(image: np.ndarray) -> bool:
    """Check that the loaded image has valid dimensions (min only; oversized images are auto-resized)."""
    if image is None:
        return False
    if len(image.shape) < 2:
        return False
    height, width = image.shape[:2]
    if width < ecfg.MIN_IMAGE_WIDTH or height < ecfg.MIN_IMAGE_HEIGHT:
        return False
    return True


def downscale_if_oversized(image: np.ndarray) -> np.ndarray:
    """Proportionally downscale an image if it exceeds MAX dimensions, no padding."""
    height, width = image.shape[:2]
    if width <= ecfg.MAX_IMAGE_WIDTH and height <= ecfg.MAX_IMAGE_HEIGHT:
        return image

    scale = min(ecfg.MAX_IMAGE_WIDTH / width, ecfg.MAX_IMAGE_HEIGHT / height)
    new_w = int(width * scale)
    new_h = int(height * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_raw_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from disk using OpenCV."""
    if not validate_image_path(image_path):
        return None
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def resize_image(
    image: np.ndarray,
    target_width: int = ecfg.TARGET_WIDTH,
    target_height: int = ecfg.TARGET_HEIGHT,
) -> np.ndarray:
    """Resize an image to the target dimensions while preserving aspect ratio."""
    height, width = image.shape[:2]

    if width == target_width and height == target_height:
        return image.copy()

    aspect_ratio_source = width / height
    aspect_ratio_target = target_width / target_height

    if aspect_ratio_source > aspect_ratio_target:
        new_width = target_width
        new_height = int(target_width / aspect_ratio_source)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio_source)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    return canvas


def compute_brightness_histogram(image: np.ndarray) -> np.ndarray:
    """Compute the brightness histogram from the grayscale version of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [ecfg.HISTOGRAM_BINS],
        [ecfg.HISTOGRAM_RANGE_MIN, ecfg.HISTOGRAM_RANGE_MAX],
    )
    histogram = histogram.flatten()
    return histogram


def compute_histogram_skewness(histogram: np.ndarray) -> float:
    """Compute the skewness of the brightness histogram."""
    total_pixels = np.sum(histogram)
    if total_pixels == 0:
        return 0.0

    bin_centers = np.arange(ecfg.HISTOGRAM_BINS)
    mean_val = np.sum(bin_centers * histogram) / total_pixels
    variance = np.sum(((bin_centers - mean_val) ** 2) * histogram) / total_pixels
    std_val = np.sqrt(variance) if variance > 0 else 1.0

    skewness = np.sum(((bin_centers - mean_val) ** 3) * histogram) / (
        total_pixels * (std_val ** 3)
    )
    return float(skewness)


def compute_histogram_kurtosis(histogram: np.ndarray) -> float:
    """Compute the kurtosis of the brightness histogram."""
    total_pixels = np.sum(histogram)
    if total_pixels == 0:
        return 0.0

    bin_centers = np.arange(ecfg.HISTOGRAM_BINS)
    mean_val = np.sum(bin_centers * histogram) / total_pixels
    variance = np.sum(((bin_centers - mean_val) ** 2) * histogram) / total_pixels
    std_val = np.sqrt(variance) if variance > 0 else 1.0

    kurtosis = np.sum(((bin_centers - mean_val) ** 4) * histogram) / (
        total_pixels * (std_val ** 4)
    )
    kurtosis -= 3.0
    return float(kurtosis)


def extract_metadata(image_path: str, image: np.ndarray) -> ImageMetadata:
    """Extract comprehensive metadata from the loaded image."""
    metadata = ImageMetadata()

    metadata.file_path = os.path.abspath(image_path)
    metadata.file_name = os.path.basename(image_path)
    metadata.file_extension = os.path.splitext(image_path)[1].lower()
    metadata.file_size_bytes = os.path.getsize(image_path)

    metadata.original_height, metadata.original_width = image.shape[:2]
    metadata.original_channels = image.shape[2] if len(image.shape) == 3 else 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metadata.mean_brightness = float(np.mean(gray))
    metadata.std_brightness = float(np.std(gray))
    metadata.min_brightness = int(np.min(gray))
    metadata.max_brightness = int(np.max(gray))
    metadata.dynamic_range = metadata.max_brightness - metadata.min_brightness

    metadata.brightness_histogram = compute_brightness_histogram(image)
    metadata.histogram_skewness = compute_histogram_skewness(
        metadata.brightness_histogram
    )
    metadata.histogram_kurtosis = compute_histogram_kurtosis(
        metadata.brightness_histogram
    )

    metadata.is_low_light = metadata.mean_brightness < ecfg.DARK_UPPER_THRESHOLD
    metadata.is_overexposed = metadata.mean_brightness > ecfg.BRIGHT_UPPER_THRESHOLD
    metadata.is_low_contrast = metadata.dynamic_range < ecfg.ZONE_CONTRAST_LOW

    return metadata


def load_and_prepare_image(
    image_path: str, do_resize: bool = True
) -> Tuple[Optional[np.ndarray], Optional[ImageMetadata]]:
    """
    Full image loading pipeline: validate, load, extract metadata, resize.
    Returns the prepared image and its metadata, or (None, None) on failure.
    """
    raw_image = load_raw_image(image_path)
    if raw_image is None:
        return None, None

    if not validate_image_dimensions(raw_image):
        return None, None

    metadata = extract_metadata(image_path, raw_image)

    if do_resize:
        prepared = resize_image(raw_image)
        metadata.resized_width = prepared.shape[1]
        metadata.resized_height = prepared.shape[0]
    else:
        prepared = downscale_if_oversized(raw_image)
        metadata.resized_width = prepared.shape[1]
        metadata.resized_height = prepared.shape[0]

    return prepared, metadata
