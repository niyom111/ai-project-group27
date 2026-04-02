"""
Edge-Preserving Filter Module
------------------------------
Provides bilateral and guided filter implementations for smoothing
while preserving edges, important for maintaining face detail.
"""

import cv2
import numpy as np
from typing import Optional

from enhancement import enhancement_config as ecfg


def apply_bilateral_smooth(
    image: np.ndarray,
    d: int = ecfg.EDGE_BILATERAL_D,
    sigma_color: float = ecfg.EDGE_BILATERAL_SIGMA_COLOR,
    sigma_space: float = ecfg.EDGE_BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    """Apply bilateral filter for edge-preserving smoothing."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_iterative_bilateral(
    image: np.ndarray,
    iterations: int = 2,
    d: int = ecfg.EDGE_BILATERAL_D,
    sigma_color: float = ecfg.EDGE_BILATERAL_SIGMA_COLOR,
    sigma_space: float = ecfg.EDGE_BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    """Apply bilateral filter multiple times for stronger smoothing."""
    result = image.copy()
    for i in range(iterations):
        sigma_c = sigma_color * (0.8 ** i)
        sigma_s = sigma_space * (0.8 ** i)
        result = cv2.bilateralFilter(result, d, sigma_c, sigma_s)
    return result


def guided_filter_gray(
    guide: np.ndarray,
    source: np.ndarray,
    radius: int = ecfg.EDGE_GUIDED_RADIUS,
    eps: float = ecfg.EDGE_GUIDED_EPS,
) -> np.ndarray:
    """
    Guided filter implementation for single-channel images.
    The guide image defines edges to be preserved; the source is filtered.
    """
    guide_f = guide.astype(np.float64)
    source_f = source.astype(np.float64)

    ksize = 2 * radius + 1

    mean_guide = cv2.boxFilter(guide_f, -1, (ksize, ksize))
    mean_source = cv2.boxFilter(source_f, -1, (ksize, ksize))
    mean_guide_source = cv2.boxFilter(guide_f * source_f, -1, (ksize, ksize))
    mean_guide_sq = cv2.boxFilter(guide_f * guide_f, -1, (ksize, ksize))

    cov_guide_source = mean_guide_source - mean_guide * mean_source
    var_guide = mean_guide_sq - mean_guide * mean_guide

    a = cov_guide_source / (var_guide + eps)
    b = mean_source - a * mean_guide

    mean_a = cv2.boxFilter(a, -1, (ksize, ksize))
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize))

    output = mean_a * guide_f + mean_b
    return np.clip(output, 0, 255).astype(np.uint8)


def guided_filter_color(
    image: np.ndarray,
    radius: int = ecfg.EDGE_GUIDED_RADIUS,
    eps: float = ecfg.EDGE_GUIDED_EPS,
) -> np.ndarray:
    """
    Apply guided filter to each channel of a color image,
    using the grayscale version as guide.
    """
    if len(image.shape) != 3:
        guide = image
        return guided_filter_gray(guide, image, radius, eps)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channels = cv2.split(image)
    filtered_channels = []

    for ch in channels:
        filtered = guided_filter_gray(gray, ch, radius, eps)
        filtered_channels.append(filtered)

    return cv2.merge(filtered_channels)


def apply_edge_preserving_opencv(
    image: np.ndarray,
    flags: int = cv2.RECURS_FILTER,
    sigma_s: float = 60.0,
    sigma_r: float = 0.4,
) -> np.ndarray:
    """
    Use OpenCV's built-in edge-preserving filter.
    flags: cv2.RECURS_FILTER or cv2.NORMCONV_FILTER
    """
    return cv2.edgePreservingFilter(
        image, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r
    )


def apply_stylization_smooth(
    image: np.ndarray,
    sigma_s: float = 60.0,
    sigma_r: float = 0.07,
) -> np.ndarray:
    """
    Very subtle edge-preserving smooth using OpenCV stylization
    at low sigma_r for minimal artistic effect.
    """
    return cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)


def apply_domain_transform_filter(
    image: np.ndarray,
    sigma_s: float = 10.0,
    sigma_r: float = 0.15,
    mode: int = cv2.ximgproc.DTF_NC if hasattr(cv2, "ximgproc") else 0,
) -> np.ndarray:
    """
    Domain transform filter for fast edge-preserving smoothing.
    Falls back to bilateral if ximgproc is not available.
    """
    if hasattr(cv2, "ximgproc"):
        try:
            filtered = cv2.ximgproc.dtFilter(
                image, image, sigma_s, sigma_r, mode
            )
            return filtered
        except Exception:
            pass

    return apply_bilateral_smooth(image)


def apply_adaptive_edge_preserving(
    image: np.ndarray,
    noise_level: float = 0.0,
) -> np.ndarray:
    """
    Choose edge-preserving filter parameters based on estimated noise.
    Higher noise -> stronger smoothing parameters.
    """
    if noise_level < 10:
        sigma_color = 30.0
        sigma_space = 30.0
        d = 5
    elif noise_level < 25:
        sigma_color = 60.0
        sigma_space = 60.0
        d = 7
    else:
        sigma_color = 90.0
        sigma_space = 90.0
        d = 9

    return apply_bilateral_smooth(image, d, sigma_color, sigma_space)


def edge_preserve_blend(
    original: np.ndarray,
    filtered: np.ndarray,
    blend: float = ecfg.EDGE_PRESERVE_BLEND,
) -> np.ndarray:
    """Blend edge-preserved result with original."""
    return cv2.addWeighted(filtered, blend, original, 1.0 - blend, 0)


def full_edge_preserving_pipeline(
    image: np.ndarray,
    use_guided: bool = True,
    use_bilateral: bool = True,
    blend_strength: float = ecfg.EDGE_PRESERVE_BLEND,
) -> np.ndarray:
    """
    Full edge-preserving smoothing pipeline.
    Combines guided + bilateral for strong yet edge-aware smoothing.
    """
    result = image.copy()

    if use_guided:
        guided_result = guided_filter_color(
            result,
            radius=ecfg.EDGE_GUIDED_RADIUS,
            eps=ecfg.EDGE_GUIDED_EPS,
        )
        result = edge_preserve_blend(result, guided_result, blend_strength * 0.5)

    if use_bilateral:
        bilateral_result = apply_bilateral_smooth(result)
        result = edge_preserve_blend(result, bilateral_result, blend_strength * 0.5)

    return result
