"""
Brightness Normalizer Module
-----------------------------
Normalizes brightness across all zones so the overall image has
a consistent, face-detection-friendly luminance level.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import (
    split_lab_channels,
    merge_lab_channels,
    lab_to_bgr,
)


def compute_mean_brightness(l_channel: np.ndarray) -> float:
    """Compute the mean brightness of an L channel."""
    return float(np.mean(l_channel))


def compute_brightness_map(
    l_channel: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    """
    Compute a coarse brightness map by averaging blocks.
    Returns a map the same size as the input (upsampled from blocks).
    """
    height, width = l_channel.shape
    l_float = l_channel.astype(np.float64)

    rows = (height + block_size - 1) // block_size
    cols = (width + block_size - 1) // block_size
    brightness_grid = np.zeros((rows, cols), dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            y_start = r * block_size
            y_end = min((r + 1) * block_size, height)
            x_start = c * block_size
            x_end = min((c + 1) * block_size, width)
            block = l_float[y_start:y_end, x_start:x_end]
            brightness_grid[r, c] = np.mean(block)

    brightness_map = cv2.resize(
        brightness_grid, (width, height), interpolation=cv2.INTER_LINEAR
    )
    return brightness_map


def normalize_to_target_mean(
    l_channel: np.ndarray,
    target_mean: float = ecfg.BRIGHTNESS_TARGET_MEAN,
    strength: float = ecfg.BRIGHTNESS_ADJUSTMENT_STRENGTH,
    max_shift: int = ecfg.BRIGHTNESS_MAX_SHIFT,
) -> np.ndarray:
    """
    Shift the entire L channel so its mean approaches the target.
    """
    current_mean = compute_mean_brightness(l_channel)
    raw_shift = target_mean - current_mean
    shift = np.clip(raw_shift * strength, -max_shift, max_shift)

    result = l_channel.astype(np.float64) + shift
    return np.clip(result, 0, 255).astype(np.uint8)


def normalize_to_target_distribution(
    l_channel: np.ndarray,
    target_mean: float = ecfg.BRIGHTNESS_TARGET_MEAN,
    target_std: float = ecfg.BRIGHTNESS_TARGET_STD,
    strength: float = ecfg.BRIGHTNESS_ADJUSTMENT_STRENGTH,
) -> np.ndarray:
    """
    Adjust both the mean and standard deviation of the L channel
    toward target values for a controlled brightness distribution.
    """
    l_float = l_channel.astype(np.float64)
    current_mean = np.mean(l_float)
    current_std = np.std(l_float)

    if current_std < 1.0:
        current_std = 1.0

    normalized = (l_float - current_mean) / current_std
    target_result = normalized * target_std + target_mean

    blended = l_float * (1.0 - strength) + target_result * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def normalize_local_brightness(
    l_channel: np.ndarray,
    target_mean: float = ecfg.BRIGHTNESS_TARGET_MEAN,
    block_size: int = 64,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Per-block brightness normalization: each block is shifted toward
    the target mean, reducing brightness variation across the image.
    """
    height, width = l_channel.shape
    result = l_channel.astype(np.float64)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = result[y:y_end, x:x_end]

            block_mean = np.mean(block)
            shift = (target_mean - block_mean) * strength
            result[y:y_end, x:x_end] = block + shift

    return np.clip(result, 0, 255).astype(np.uint8)


def preserve_highlights(
    original: np.ndarray,
    normalized: np.ndarray,
    threshold: int = ecfg.BRIGHTNESS_HIGHLIGHT_THRESHOLD,
) -> np.ndarray:
    """
    Blend back original values for highlight pixels to avoid
    clipping or unnatural brightening of already-bright areas.
    """
    highlight_mask = (original > threshold).astype(np.float64)
    ksize = 15
    smooth_mask = cv2.GaussianBlur(highlight_mask, (ksize, ksize), 0)

    result = (normalized.astype(np.float64) * (1.0 - smooth_mask) +
              original.astype(np.float64) * smooth_mask)
    return np.clip(result, 0, 255).astype(np.uint8)


def preserve_shadows(
    original: np.ndarray,
    normalized: np.ndarray,
    threshold: int = ecfg.BRIGHTNESS_SHADOW_THRESHOLD,
) -> np.ndarray:
    """
    Blend back for deep shadow pixels to prevent crushing blacks.
    """
    shadow_mask = (original < threshold).astype(np.float64)
    ksize = 15
    smooth_mask = cv2.GaussianBlur(shadow_mask, (ksize, ksize), 0)

    result = (normalized.astype(np.float64) * (1.0 - smooth_mask * 0.5) +
              original.astype(np.float64) * (smooth_mask * 0.5))
    return np.clip(result, 0, 255).astype(np.uint8)


def equalize_zone_brightness(
    zone_l_channels: List[np.ndarray],
    target_mean: float = ecfg.BRIGHTNESS_TARGET_MEAN,
) -> List[np.ndarray]:
    """
    Normalize a list of zone L-channels so they all converge
    toward the target mean brightness.
    """
    normalized_zones = []
    for l_channel in zone_l_channels:
        normalized = normalize_to_target_mean(l_channel, target_mean)
        normalized_zones.append(normalized)
    return normalized_zones


def normalize_image_brightness(
    image: np.ndarray,
    target_mean: float = ecfg.BRIGHTNESS_TARGET_MEAN,
    strength: float = ecfg.BRIGHTNESS_ADJUSTMENT_STRENGTH,
) -> np.ndarray:
    """
    Full brightness normalization pipeline on a BGR image.
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)

    original_l = l_channel.copy()

    normalized = normalize_to_target_distribution(l_channel, target_mean,
                                                   strength=strength)

    if ecfg.BRIGHTNESS_PRESERVE_HIGHLIGHTS:
        normalized = preserve_highlights(original_l, normalized)

    if ecfg.BRIGHTNESS_PRESERVE_SHADOWS:
        normalized = preserve_shadows(original_l, normalized)

    merged = merge_lab_channels(normalized, a_channel, b_channel)
    return lab_to_bgr(merged)
