"""
Sharpening Engine Module
------------------------
Provides unsharp masking, Laplacian sharpening, and high-pass
sharpening with per-zone adaptive strength settings.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from enhancement import enhancement_config as ecfg


def apply_unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 0.5,
    threshold: int = 0,
) -> np.ndarray:
    """
    Apply unsharp masking:  sharpened = original + strength * (original - blurred)
    """
    ksize = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    if len(image.shape) == 3:
        diff = image.astype(np.float64) - blurred.astype(np.float64)
    else:
        diff = image.astype(np.float64) - blurred.astype(np.float64)

    if threshold > 0:
        if len(diff.shape) == 3:
            diff_magnitude = np.sqrt(np.sum(diff ** 2, axis=2, keepdims=True))
            mask = (diff_magnitude > threshold).astype(np.float64)
            mask = np.repeat(mask, diff.shape[2], axis=2)
        else:
            mask = (np.abs(diff) > threshold).astype(np.float64)
        diff = diff * mask

    sharpened = image.astype(np.float64) + strength * diff
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_laplacian_sharpen(
    image: np.ndarray,
    weight: float = 0.1,
) -> np.ndarray:
    """
    Sharpen using the Laplacian operator:
    sharpened = original - weight * Laplacian(original)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    if len(image.shape) == 3:
        laplacian_3ch = np.stack([laplacian] * 3, axis=-1)
        sharpened = image.astype(np.float64) - weight * laplacian_3ch
    else:
        sharpened = image.astype(np.float64) - weight * laplacian

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_high_pass_sharpen(
    image: np.ndarray,
    kernel_size: int = 5,
    strength: float = 0.3,
) -> np.ndarray:
    """
    High-pass sharpening: extract high frequencies and add them back.
    """
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    low_pass = cv2.GaussianBlur(image, (ksize, ksize), 0)
    high_pass = image.astype(np.float64) - low_pass.astype(np.float64)
    sharpened = image.astype(np.float64) + strength * high_pass
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_detail_enhancement(
    image: np.ndarray,
    sigma_s: float = 10.0,
    sigma_r: float = 0.15,
) -> np.ndarray:
    """
    Use OpenCV's detailEnhance for perceptually pleasing sharpening.
    """
    enhanced = cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)
    return enhanced


def apply_kernel_sharpen(
    image: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Sharpen using a 3x3 sharpening convolution kernel.
    """
    kernel_base = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ], dtype=np.float64)

    identity = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.float64)

    kernel = identity + strength * (kernel_base - identity)
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def apply_edge_aware_sharpen(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 0.5,
    edge_threshold: float = 30.0,
) -> np.ndarray:
    """
    Only sharpen pixels that are near edges (to avoid amplifying noise
    in smooth regions).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
    edge_mask = cv2.dilate(edges, None, iterations=2)

    if len(image.shape) == 3:
        edge_mask_3ch = np.stack([edge_mask] * 3, axis=-1)
    else:
        edge_mask_3ch = edge_mask

    sharpened = apply_unsharp_mask(image, sigma, strength)

    edge_float = edge_mask_3ch.astype(np.float64) / 255.0
    result = (sharpened.astype(np.float64) * edge_float +
              image.astype(np.float64) * (1.0 - edge_float))

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_multiscale_sharpen(
    image: np.ndarray,
    scales: list = None,
) -> np.ndarray:
    """
    Apply sharpening at multiple scales and combine.
    Fine + medium + coarse detail enhancement.
    """
    if scales is None:
        scales = [
            {"sigma": 0.5, "strength": 0.2},
            {"sigma": 1.5, "strength": 0.3},
            {"sigma": 3.0, "strength": 0.15},
        ]

    result = image.astype(np.float64)

    for scale in scales:
        sigma = scale["sigma"]
        strength = scale["strength"]
        ksize = int(sigma * 6) | 1
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        detail = image.astype(np.float64) - blurred.astype(np.float64)
        result += strength * detail

    return np.clip(result, 0, 255).astype(np.uint8)


def sharpen_zone(
    image: np.ndarray,
    zone_category: str,
) -> np.ndarray:
    """
    Apply sharpening tuned for the given zone category.
    """
    settings = ecfg.SHARPENING_SETTINGS.get(
        zone_category, ecfg.SHARPENING_SETTINGS["normal"]
    )

    method = settings["method"]

    if method == "none":
        return image.copy()

    result = image.copy()

    if method == "unsharp":
        result = apply_unsharp_mask(
            result,
            sigma=settings["unsharp_sigma"],
            strength=settings["unsharp_strength"],
            threshold=settings["unsharp_threshold"],
        )

    if settings.get("apply_laplacian", False):
        result = apply_laplacian_sharpen(
            result,
            weight=settings["laplacian_weight"],
        )

    return result


def smart_sharpen(
    image: np.ndarray,
    zone_category: str,
    noise_level: float = 0.0,
) -> np.ndarray:
    """
    Wrapper that adjusts sharpening intensity based on noise level.
    High-noise images get less sharpening to avoid amplifying artifacts.
    """
    settings = ecfg.SHARPENING_SETTINGS.get(
        zone_category, ecfg.SHARPENING_SETTINGS["normal"]
    )

    if settings["method"] == "none":
        return image.copy()

    noise_factor = 1.0
    if noise_level > 30:
        noise_factor = 0.3
    elif noise_level > 20:
        noise_factor = 0.5
    elif noise_level > 10:
        noise_factor = 0.7

    adjusted_strength = settings["unsharp_strength"] * noise_factor
    adjusted_sigma = settings["unsharp_sigma"]

    if noise_level > 15:
        return apply_edge_aware_sharpen(
            image, adjusted_sigma, adjusted_strength
        )

    return apply_unsharp_mask(
        image, adjusted_sigma, adjusted_strength,
        settings["unsharp_threshold"]
    )
