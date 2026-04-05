"""
Noise Reducer Module
--------------------
Provides multiple noise reduction strategies (bilateral, Non-Local Means,
median, Gaussian) selected per zone based on noise level and category.
"""

import cv2
import numpy as np
from typing import Optional

from enhancement import enhancement_config as ecfg


def estimate_noise_level(image: np.ndarray) -> float:
    """
    Estimate the noise level in an image using the Laplacian method.
    Returns the standard deviation of the Laplacian response.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_level = float(np.std(laplacian))
    return noise_level


def estimate_noise_mad(image: np.ndarray) -> float:
    """
    Estimate noise using Median Absolute Deviation of the
    high-frequency component (Laplacian).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    mad = np.median(np.abs(laplacian - np.median(laplacian)))
    sigma = mad * 1.4826
    return float(sigma)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """Apply Gaussian blur for basic smoothing."""
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply median filter for salt-and-pepper noise removal."""
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(image, ksize)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filter for edge-preserving smoothing."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_nlm_denoise_color(
    image: np.ndarray,
    h_luminance: float = 10.0,
    h_color: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Apply Non-Local Means denoising for color images."""
    if len(image.shape) != 3:
        return apply_nlm_denoise_gray(image, h_luminance, template_window,
                                      search_window)

    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h_luminance, h_color, template_window, search_window
    )
    return denoised


def apply_nlm_denoise_gray(
    image: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Apply Non-Local Means denoising for grayscale images."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    denoised = cv2.fastNlMeansDenoising(gray, None, h, template_window, search_window)
    return denoised


def apply_adaptive_bilateral(
    image: np.ndarray,
    noise_level: float,
) -> np.ndarray:
    """
    Apply bilateral filter with parameters adapted to the estimated noise level.
    Higher noise -> larger filter diameter and sigma values.
    """
    if noise_level < 10:
        d, sigma_c, sigma_s = 5, 25, 25
    elif noise_level < 25:
        d, sigma_c, sigma_s = 7, 50, 50
    elif noise_level < 50:
        d, sigma_c, sigma_s = 9, 75, 75
    else:
        d, sigma_c, sigma_s = 11, 100, 100

    return apply_bilateral_filter(image, d, sigma_c, sigma_s)


def apply_progressive_denoise(
    image: np.ndarray,
    iterations: int = 2,
    h_start: float = 10.0,
    h_decay: float = 0.6,
) -> np.ndarray:
    """
    Apply NLM denoising in multiple passes with decreasing strength.
    Each pass removes finer noise without over-smoothing.
    """
    result = image.copy()
    current_h = h_start

    for i in range(iterations):
        if len(result.shape) == 3:
            result = apply_nlm_denoise_color(
                result, current_h, current_h, 7, 21
            )
        else:
            result = apply_nlm_denoise_gray(result, current_h, 7, 21)
        current_h *= h_decay

    return result


def apply_channel_wise_denoise(
    image: np.ndarray,
    h_luminance: float = 10.0,
    h_color: float = 6.0,
) -> np.ndarray:
    """
    Denoise each LAB channel independently — stronger on L (luminance),
    lighter on A/B (chrominance) to preserve color fidelity.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_denoised = cv2.fastNlMeansDenoising(l_channel, None, h_luminance, 7, 21)
    a_denoised = cv2.fastNlMeansDenoising(a_channel, None, h_color, 7, 15)
    b_denoised = cv2.fastNlMeansDenoising(b_channel, None, h_color, 7, 15)

    lab_denoised = cv2.merge([l_denoised, a_denoised, b_denoised])
    result = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)
    return result


def denoise_zone(
    image: np.ndarray,
    zone_category: str,
) -> np.ndarray:
    """
    Apply noise reduction tuned for the given zone category.
    Uses settings from enhancement_config.
    """
    settings = ecfg.NOISE_REDUCTION_SETTINGS.get(
        zone_category, ecfg.NOISE_REDUCTION_SETTINGS["normal"]
    )

    method = settings["method"]
    result = image.copy()

    if method == "none":
        return result

    if settings.get("apply_bilateral_pre", False):
        result = apply_bilateral_filter(
            result,
            settings["bilateral_d"],
            settings["bilateral_sigma_color"],
            settings["bilateral_sigma_space"],
        )

    if method == "nlm":
        result = apply_nlm_denoise_color(
            result,
            settings["h_luminance"],
            settings["h_color"],
            settings["template_window"],
            settings["search_window"],
        )
    elif method == "bilateral":
        if not settings.get("apply_bilateral_pre", False):
            result = apply_bilateral_filter(
                result,
                settings["bilateral_d"],
                settings["bilateral_sigma_color"],
                settings["bilateral_sigma_space"],
            )
    elif method == "light":
        result = apply_bilateral_filter(
            result,
            settings["bilateral_d"],
            settings["bilateral_sigma_color"],
            settings["bilateral_sigma_space"],
        )

    if settings.get("apply_median_post", False):
        result = apply_median_filter(result, settings["median_ksize"])

    return result


def smart_denoise(
    image: np.ndarray,
    zone_category: str,
) -> np.ndarray:
    """
    Wrapper that estimates noise first, then decides whether denoising
    is even necessary before applying zone-specific denoising.
    """
    noise_level = estimate_noise_level(image)

    if noise_level < 5.0 and zone_category not in ("dark",):
        return image.copy()

    if noise_level > 40.0 and zone_category == "dark":
        result = apply_progressive_denoise(image, iterations=2, h_start=12.0)
        return result

    return denoise_zone(image, zone_category)
