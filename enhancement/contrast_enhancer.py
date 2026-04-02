"""
Contrast Enhancer Module
------------------------
Provides local and global contrast enhancement using multi-scale CLAHE,
Wallis filter, and adaptive local contrast methods.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import (
    bgr_to_lab,
    lab_to_bgr,
    split_lab_channels,
    merge_lab_channels,
)


def apply_global_contrast_stretch(
    channel: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Stretch the channel so that the low and high percentile values
    occupy the full 0-255 range.
    """
    low = np.percentile(channel, low_percentile)
    high = np.percentile(channel, high_percentile)

    if high <= low:
        return channel.copy()

    stretched = (channel.astype(np.float64) - low) / (high - low) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_wallis_filter(
    channel: np.ndarray,
    target_mean: float = ecfg.CONTRAST_WALLIS_TARGET_MEAN,
    target_std: float = ecfg.CONTRAST_WALLIS_TARGET_STD,
    brightness_constant: float = ecfg.CONTRAST_WALLIS_BRIGHTNESS_CONSTANT,
    contrast_constant: float = ecfg.CONTRAST_WALLIS_CONTRAST_CONSTANT,
    block_size: int = ecfg.CONTRAST_LOCAL_BLOCK_SIZE,
) -> np.ndarray:
    """
    Wallis filter: locally adapts mean and standard deviation
    to target values for uniform contrast across the image.
    """
    channel_float = channel.astype(np.float64)

    ksize = block_size if block_size % 2 == 1 else block_size + 1
    local_mean = cv2.GaussianBlur(channel_float, (ksize, ksize), 0)

    local_sq_mean = cv2.GaussianBlur(channel_float ** 2, (ksize, ksize), 0)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)
    local_std = np.sqrt(local_var)
    local_std = np.maximum(local_std, 1.0)

    gain = contrast_constant * target_std / (
        contrast_constant * local_std + (1.0 - contrast_constant) * target_std
    )
    gain = np.clip(gain, 0.5, 3.0)

    offset = (brightness_constant * target_mean +
              (1.0 - brightness_constant) * local_mean -
              gain * local_mean)

    result = gain * channel_float + offset
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_multiscale_clahe(
    channel: np.ndarray,
    scales: List[Dict] = None,
) -> np.ndarray:
    """
    Apply CLAHE at multiple tile grid sizes and blend the results.
    Captures both fine and coarse contrast variations.
    """
    if scales is None:
        scales = ecfg.CONTRAST_MULTISCALE_CLAHE_SCALES

    result = np.zeros_like(channel, dtype=np.float64)
    total_weight = 0.0

    for scale in scales:
        clahe = cv2.createCLAHE(
            clipLimit=scale["clip_limit"],
            tileGridSize=scale["tile_grid_size"],
        )
        enhanced = clahe.apply(channel)
        weight = scale["weight"]
        result += enhanced.astype(np.float64) * weight
        total_weight += weight

    if total_weight > 0:
        result /= total_weight

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_local_contrast_normalization(
    channel: np.ndarray,
    kernel_size: int = 31,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Normalize local contrast by dividing by the local standard deviation.
    """
    channel_float = channel.astype(np.float64)
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    local_mean = cv2.GaussianBlur(channel_float, (ksize, ksize), 0)
    local_sq_mean = cv2.GaussianBlur(channel_float ** 2, (ksize, ksize), 0)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 1.0)
    local_std = np.sqrt(local_var)

    normalized = (channel_float - local_mean) / local_std
    target_std = 40.0
    target_mean = 128.0
    result = normalized * target_std + target_mean

    blended = channel_float * (1.0 - strength) + result * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_adaptive_contrast(
    image: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    """
    Divide image into blocks and enhance contrast per block based
    on each block's dynamic range.
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)
    height, width = l_channel.shape
    result = l_channel.copy().astype(np.float64)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = l_channel[y:y_end, x:x_end].astype(np.float64)

            block_min = np.min(block)
            block_max = np.max(block)
            dynamic_range = block_max - block_min

            if dynamic_range < 20:
                target_min = max(0, block_min - 15)
                target_max = min(255, block_max + 15)
                if block_max > block_min:
                    stretched = ((block - block_min) / (block_max - block_min) *
                                 (target_max - target_min) + target_min)
                    result[y:y_end, x:x_end] = stretched
            else:
                result[y:y_end, x:x_end] = block

    result = np.clip(result, 0, 255).astype(np.uint8)
    merged = merge_lab_channels(result, a_channel, b_channel)
    return lab_to_bgr(merged)


def enhance_midtone_contrast(
    channel: np.ndarray,
    midtone_center: int = 128,
    midtone_range: int = 60,
    boost: float = 1.3,
) -> np.ndarray:
    """
    Selectively boost contrast in midtone regions where faces
    are typically represented.
    """
    channel_float = channel.astype(np.float64)
    low = midtone_center - midtone_range
    high = midtone_center + midtone_range

    midtone_mask = np.logical_and(
        channel_float >= low, channel_float <= high
    ).astype(np.float64)

    ksize = 15
    smooth_mask = cv2.GaussianBlur(midtone_mask, (ksize, ksize), 0)

    offset = channel_float - midtone_center
    boosted = midtone_center + offset * boost

    result = channel_float * (1.0 - smooth_mask) + boosted * smooth_mask
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_contrast_full(
    image: np.ndarray,
    apply_wallis: bool = True,
    apply_multiscale: bool = True,
    apply_midtone: bool = True,
) -> np.ndarray:
    """
    Full contrast enhancement pipeline on the L channel.
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)

    result = l_channel.copy()

    if apply_multiscale:
        result = apply_multiscale_clahe(result)

    if apply_wallis:
        wallis_result = apply_wallis_filter(result)
        result = cv2.addWeighted(result, 0.6, wallis_result, 0.4, 0)

    if apply_midtone:
        result = enhance_midtone_contrast(result)

    merged = merge_lab_channels(result, a_channel, b_channel)
    return lab_to_bgr(merged)
