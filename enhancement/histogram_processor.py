"""
Histogram Processor Module
--------------------------
Provides histogram equalization, histogram stretching, and adaptive
CLAHE operations with per-zone configurable parameters.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from enhancement import enhancement_config as ecfg


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    """Compute a 256-bin histogram for a single channel."""
    hist = cv2.calcHist(
        [channel], [0], None,
        [ecfg.HISTOGRAM_BINS],
        [ecfg.HISTOGRAM_RANGE_MIN, ecfg.HISTOGRAM_RANGE_MAX],
    )
    return hist.flatten()


def compute_cumulative_histogram(histogram: np.ndarray) -> np.ndarray:
    """Compute the cumulative distribution function of a histogram."""
    cdf = np.cumsum(histogram)
    cdf_normalized = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    return cdf_normalized


def standard_histogram_equalization(channel: np.ndarray) -> np.ndarray:
    """Apply OpenCV's standard histogram equalization."""
    equalized = cv2.equalizeHist(channel)
    return equalized


def blended_histogram_equalization(
    channel: np.ndarray,
    original_weight: float = ecfg.HIST_EQ_BLEND_ORIGINAL_WEIGHT,
    equalized_weight: float = ecfg.HIST_EQ_BLEND_EQUALIZED_WEIGHT,
) -> np.ndarray:
    """
    Apply histogram equalization blended with the original to avoid
    over-processing.
    """
    equalized = standard_histogram_equalization(channel)
    blended = cv2.addWeighted(
        channel, original_weight,
        equalized, equalized_weight,
        0,
    )
    return blended


def histogram_stretching(
    channel: np.ndarray,
    low_percentile: int = ecfg.HISTOGRAM_STRETCH_LOW_PERCENTILE,
    high_percentile: int = ecfg.HISTOGRAM_STRETCH_HIGH_PERCENTILE,
) -> np.ndarray:
    """
    Stretch the histogram so that the low and high percentile values
    map to 0 and 255 respectively.
    """
    low_val = np.percentile(channel, low_percentile)
    high_val = np.percentile(channel, high_percentile)

    if high_val <= low_val:
        return channel.copy()

    stretched = (channel.astype(np.float64) - low_val) / (high_val - low_val)
    stretched = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    return stretched


def adaptive_histogram_stretching(
    channel: np.ndarray,
    target_low: int = 10,
    target_high: int = 245,
) -> np.ndarray:
    """
    Stretch histogram adaptively to reach a specific target output range.
    """
    current_low = float(np.percentile(channel, 1))
    current_high = float(np.percentile(channel, 99))

    if current_high <= current_low:
        return channel.copy()

    scale = (target_high - target_low) / (current_high - current_low)
    offset = target_low - current_low * scale

    result = channel.astype(np.float64) * scale + offset
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_clahe(
    channel: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = ecfg.CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(channel)
    return result


def apply_clahe_iterative(
    channel: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = ecfg.CLAHE_TILE_GRID_SIZE,
    iterations: int = 1,
) -> np.ndarray:
    """Apply CLAHE multiple times with decreasing clip limit."""
    result = channel.copy()
    current_clip = clip_limit

    for i in range(iterations):
        clahe = cv2.createCLAHE(clipLimit=current_clip, tileGridSize=tile_grid_size)
        result = clahe.apply(result)
        current_clip = max(1.0, current_clip * 0.7)

    return result


def apply_clahe_blended(
    channel: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = ecfg.CLAHE_TILE_GRID_SIZE,
    blend_weight: float = 0.8,
) -> np.ndarray:
    """Apply CLAHE and blend the result with the original channel."""
    clahe_result = apply_clahe(channel, clip_limit, tile_grid_size)
    original_weight = 1.0 - blend_weight
    blended = cv2.addWeighted(
        clahe_result, blend_weight,
        channel, original_weight,
        0,
    )
    return blended


def apply_zone_clahe(
    channel: np.ndarray,
    zone_category: str,
) -> np.ndarray:
    """
    Apply CLAHE with settings tuned for the specified zone category.
    """
    settings = ecfg.CLAHE_SETTINGS.get(zone_category, ecfg.CLAHE_SETTINGS["normal"])

    clip_limit = settings["clip_limit"]
    tile_grid_size = settings["tile_grid_size"]
    iterations = settings["iterations"]
    blend_weight = settings["blend_weight"]

    if iterations > 1:
        clahe_result = apply_clahe_iterative(
            channel, clip_limit, tile_grid_size, iterations
        )
    else:
        clahe_result = apply_clahe(channel, clip_limit, tile_grid_size)

    original_weight = 1.0 - blend_weight
    blended = cv2.addWeighted(
        clahe_result, blend_weight,
        channel, original_weight,
        0,
    )
    return blended


def histogram_matching(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Match the histogram of the source channel to a reference channel.
    Useful for normalizing brightness distribution across zones.
    """
    src_hist = compute_histogram(source)
    ref_hist = compute_histogram(reference)

    src_cdf = compute_cumulative_histogram(src_hist)
    ref_cdf = compute_cumulative_histogram(ref_hist)

    lookup_table = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        closest_ref_val = 0
        min_diff = abs(src_cdf[src_val] - ref_cdf[0])
        for ref_val in range(1, 256):
            diff = abs(src_cdf[src_val] - ref_cdf[ref_val])
            if diff < min_diff:
                min_diff = diff
                closest_ref_val = ref_val
        lookup_table[src_val] = closest_ref_val

    matched = lookup_table[source]
    return matched


def process_histogram_for_zone(
    channel: np.ndarray,
    zone_category: str,
    apply_stretching: bool = True,
    apply_equalization: bool = True,
) -> np.ndarray:
    """
    Full histogram processing pipeline for a zone:
    1. Optional stretching
    2. Category-specific CLAHE
    3. Optional blended equalization
    """
    result = channel.copy()

    if apply_stretching and zone_category in ("dark", "overexposed"):
        result = histogram_stretching(result)

    result = apply_zone_clahe(result, zone_category)

    if apply_equalization and zone_category == "dark":
        result = blended_histogram_equalization(
            result,
            original_weight=0.85,
            equalized_weight=0.15,
        )

    return result
