"""
Shadow Detector Module
----------------------
Detects shadow regions in an image using gradient analysis,
adaptive thresholding, and morphological operations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import bgr_to_lab, bgr_to_gray, bgr_to_hsv


class ShadowMap:
    """Container for shadow detection results."""

    def __init__(self):
        self.shadow_mask: Optional[np.ndarray] = None
        self.shadow_boundary_mask: Optional[np.ndarray] = None
        self.shadow_intensity_map: Optional[np.ndarray] = None
        self.shadow_percentage: float = 0.0
        self.mean_shadow_brightness: float = 0.0
        self.mean_nonshadow_brightness: float = 0.0
        self.shadow_edge_strength: float = 0.0
        self.num_shadow_regions: int = 0

    def to_dict(self) -> Dict:
        return {
            "shadow_percentage": round(self.shadow_percentage, 2),
            "mean_shadow_brightness": round(self.mean_shadow_brightness, 2),
            "mean_nonshadow_brightness": round(self.mean_nonshadow_brightness, 2),
            "shadow_edge_strength": round(self.shadow_edge_strength, 3),
            "num_shadow_regions": self.num_shadow_regions,
        }


def compute_illumination_map(
    image: np.ndarray,
    blur_ksize: int = ecfg.SHADOW_GAUSSIAN_BLUR_KSIZE,
) -> np.ndarray:
    """
    Estimate the illumination map by heavily blurring the L channel.
    Shadows appear as local dips in illumination.
    """
    lab = bgr_to_lab(image, use_cache=False)
    l_channel = lab[:, :, 0]

    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    illumination = cv2.GaussianBlur(l_channel, (ksize, ksize), 0)
    return illumination


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel operators."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


def detect_shadow_by_threshold(
    image: np.ndarray,
    block_size: int = ecfg.SHADOW_THRESHOLD_BLOCK_SIZE,
    constant_c: int = ecfg.SHADOW_THRESHOLD_C,
) -> np.ndarray:
    """
    Detect shadows using adaptive thresholding on the L channel.
    Shadow pixels will be marked as 255 in the output mask.
    """
    lab = bgr_to_lab(image, use_cache=False)
    l_channel = lab[:, :, 0]

    block = block_size if block_size % 2 == 1 else block_size + 1
    shadow_mask = cv2.adaptiveThreshold(
        l_channel,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        constant_c,
    )
    return shadow_mask


def detect_shadow_by_ratio(image: np.ndarray) -> np.ndarray:
    """
    Detect shadows by analyzing the ratio of brightness to chrominance.
    Shadow regions tend to have low brightness but retain chrominance.
    """
    lab = bgr_to_lab(image, use_cache=False)
    l_channel = lab[:, :, 0].astype(np.float64)

    hsv = bgr_to_hsv(image, use_cache=False)
    s_channel = hsv[:, :, 1].astype(np.float64)

    l_normalized = l_channel / 255.0
    s_normalized = s_channel / 255.0

    safe_l = np.where(l_normalized > 0.01, l_normalized, 1.0)
    ratio = np.where(
        l_normalized > 0.01,
        s_normalized / safe_l,
        0.0,
    )

    ratio_threshold = np.mean(ratio) + 1.5 * np.std(ratio)
    shadow_mask = (ratio > ratio_threshold).astype(np.uint8) * 255

    return shadow_mask


def detect_shadow_by_illumination(
    image: np.ndarray,
    blur_ksize: int = ecfg.SHADOW_ILLUMINATION_BLUR_KSIZE,
) -> np.ndarray:
    """
    Detect shadows by comparing local brightness to estimated illumination.
    """
    lab = bgr_to_lab(image, use_cache=False)
    l_channel = lab[:, :, 0].astype(np.float64)

    illumination = compute_illumination_map(image, blur_ksize).astype(np.float64)

    ratio = np.where(illumination > 1.0, l_channel / illumination, 1.0)

    shadow_threshold = np.mean(ratio) - 0.8 * np.std(ratio)
    shadow_threshold = max(0.3, min(0.9, shadow_threshold))

    shadow_mask = (ratio < shadow_threshold).astype(np.uint8) * 255
    return shadow_mask


def refine_shadow_mask(
    shadow_mask: np.ndarray,
    morph_kernel_size: int = ecfg.SHADOW_MORPH_KERNEL_SIZE,
    morph_iterations: int = ecfg.SHADOW_MORPH_ITERATIONS,
) -> np.ndarray:
    """
    Clean up the shadow mask using morphological operations:
    closing to fill holes, opening to remove noise.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )

    closed = cv2.morphologyEx(
        shadow_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
    )
    opened = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, kernel, iterations=max(1, morph_iterations - 1)
    )

    smoothed = cv2.GaussianBlur(opened, (5, 5), 0)
    _, refined = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

    return refined


def detect_shadow_boundaries(shadow_mask: np.ndarray) -> np.ndarray:
    """
    Extract the edges/boundaries of shadow regions using Canny.
    """
    edges = cv2.Canny(shadow_mask, 50, 150)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ecfg.SHADOW_DILATION_KERNEL_SIZE, ecfg.SHADOW_DILATION_KERNEL_SIZE),
    )
    dilated = cv2.dilate(edges, kernel, iterations=ecfg.SHADOW_DILATION_ITERATIONS)

    return dilated


def compute_shadow_intensity(
    image: np.ndarray,
    shadow_mask: np.ndarray,
) -> np.ndarray:
    """
    Create a continuous shadow intensity map (0.0 = full shadow, 1.0 = no shadow).
    The mask is blurred to produce smooth transitions.
    """
    mask_float = shadow_mask.astype(np.float64) / 255.0
    mask_inverted = 1.0 - mask_float

    ksize = ecfg.SHADOW_GAUSSIAN_BLUR_KSIZE
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    smoothed = cv2.GaussianBlur(mask_inverted, (ksize, ksize), 0)

    return smoothed


def count_shadow_regions(shadow_mask: np.ndarray) -> int:
    """Count the number of distinct connected shadow regions."""
    num_labels, _ = cv2.connectedComponents(shadow_mask)
    return max(0, num_labels - 1)


def detect_shadows(image: np.ndarray) -> ShadowMap:
    """
    Full shadow detection pipeline combining multiple methods.
    """
    result = ShadowMap()

    mask_threshold = detect_shadow_by_threshold(image)
    mask_illumination = detect_shadow_by_illumination(image)
    mask_ratio = detect_shadow_by_ratio(image)

    combined = np.zeros_like(mask_threshold, dtype=np.float64)
    combined += mask_threshold.astype(np.float64) * 0.4
    combined += mask_illumination.astype(np.float64) * 0.35
    combined += mask_ratio.astype(np.float64) * 0.25

    combined = np.clip(combined, 0, 255).astype(np.uint8)
    _, combined_binary = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

    refined = refine_shadow_mask(combined_binary)

    result.shadow_mask = refined
    result.shadow_boundary_mask = detect_shadow_boundaries(refined)
    result.shadow_intensity_map = compute_shadow_intensity(image, refined)

    total_pixels = refined.shape[0] * refined.shape[1]
    shadow_pixels = np.count_nonzero(refined)
    result.shadow_percentage = (shadow_pixels / total_pixels) * 100.0

    gray = bgr_to_gray(image, use_cache=False)
    shadow_region = gray[refined > 0]
    nonshadow_region = gray[refined == 0]

    if len(shadow_region) > 0:
        result.mean_shadow_brightness = float(np.mean(shadow_region))
    if len(nonshadow_region) > 0:
        result.mean_nonshadow_brightness = float(np.mean(nonshadow_region))

    boundary = result.shadow_boundary_mask
    if boundary is not None:
        grad_at_boundary = compute_gradient_magnitude(gray)
        boundary_gradients = grad_at_boundary[boundary > 0]
        if len(boundary_gradients) > 0:
            result.shadow_edge_strength = float(np.mean(boundary_gradients))

    result.num_shadow_regions = count_shadow_regions(refined)

    return result
