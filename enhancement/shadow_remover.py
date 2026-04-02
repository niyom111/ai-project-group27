"""
Shadow Remover Module
---------------------
Removes or reduces shadows using illumination estimation,
compensation, and re-lighting techniques.
"""

import cv2
import numpy as np
from typing import Optional

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import (
    bgr_to_lab,
    lab_to_bgr,
    split_lab_channels,
    merge_lab_channels,
)
from enhancement.shadow_detector import ShadowMap, detect_shadows


def estimate_illumination_field(
    l_channel: np.ndarray,
    blur_ksize: int = ecfg.SHADOW_ILLUMINATION_BLUR_KSIZE,
) -> np.ndarray:
    """
    Estimate the underlying illumination by heavily blurring the L channel.
    """
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    illumination = cv2.GaussianBlur(
        l_channel.astype(np.float64), (ksize, ksize), 0
    )
    return illumination


def compute_shadow_compensation_map(
    l_channel: np.ndarray,
    shadow_mask: np.ndarray,
    target_brightness: Optional[float] = None,
) -> np.ndarray:
    """
    Compute a per-pixel compensation factor for shadow regions.
    Shadow pixels are lifted towards the mean non-shadow brightness.
    """
    l_float = l_channel.astype(np.float64)
    mask_bool = shadow_mask > 0

    nonshadow_pixels = l_float[~mask_bool]
    if len(nonshadow_pixels) == 0:
        return np.ones_like(l_float)

    if target_brightness is None:
        target_brightness = float(np.mean(nonshadow_pixels))

    compensation = np.ones_like(l_float)

    shadow_pixels = l_float[mask_bool]
    if len(shadow_pixels) > 0:
        safe_shadow = np.maximum(shadow_pixels, 1.0)
        factors = target_brightness / safe_shadow
        factors = np.clip(factors, 1.0, 3.0)
        compensation[mask_bool] = factors

    ksize = 15
    compensation = cv2.GaussianBlur(compensation, (ksize, ksize), 0)

    return compensation


def apply_illumination_compensation(
    image: np.ndarray,
    shadow_map: ShadowMap,
    strength: float = ecfg.SHADOW_COMPENSATION_STRENGTH,
) -> np.ndarray:
    """
    Compensate for shadows by adjusting the L channel based on
    estimated illumination deficiency.
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)
    l_float = l_channel.astype(np.float64)

    illumination = estimate_illumination_field(l_channel)

    mean_illumination = np.mean(illumination)
    if mean_illumination < 1.0:
        mean_illumination = 1.0

    ratio = mean_illumination / np.maximum(illumination, 1.0)

    compensated = l_float * (1.0 + (ratio - 1.0) * strength)
    compensated = np.clip(compensated, 0, 255).astype(np.uint8)

    merged = merge_lab_channels(compensated, a_channel, b_channel)
    result = lab_to_bgr(merged)
    return result


def apply_shadow_mask_compensation(
    image: np.ndarray,
    shadow_map: ShadowMap,
    blend_factor: float = ecfg.SHADOW_COMPENSATION_BLEND,
) -> np.ndarray:
    """
    Directly compensate shadow regions using the detected shadow mask.
    """
    if shadow_map.shadow_mask is None:
        return image.copy()

    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)

    compensation = compute_shadow_compensation_map(l_channel, shadow_map.shadow_mask)

    l_compensated = l_channel.astype(np.float64) * compensation
    l_compensated = np.clip(l_compensated, 0, 255)

    l_blended = (l_channel.astype(np.float64) * (1.0 - blend_factor) +
                 l_compensated * blend_factor)
    l_blended = np.clip(l_blended, 0, 255).astype(np.uint8)

    merged = merge_lab_channels(l_blended, a_channel, b_channel)
    return lab_to_bgr(merged)


def apply_retinex_shadow_removal(
    image: np.ndarray,
    sigma: int = 80,
) -> np.ndarray:
    """
    Single-scale Retinex-inspired shadow removal.
    Reflectance = log(Image) - log(Illumination)
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)
    l_float = l_channel.astype(np.float64) + 1.0

    ksize = sigma * 2 + 1
    illumination = cv2.GaussianBlur(l_float, (ksize, ksize), sigma)
    illumination = np.maximum(illumination, 1.0)

    reflectance = np.log(l_float) - np.log(illumination)

    r_min, r_max = reflectance.min(), reflectance.max()
    if r_max > r_min:
        reflectance = (reflectance - r_min) / (r_max - r_min) * 255.0
    else:
        reflectance = np.full_like(reflectance, 128.0)

    reflectance = np.clip(reflectance, 0, 255).astype(np.uint8)

    merged = merge_lab_channels(reflectance, a_channel, b_channel)
    return lab_to_bgr(merged)


def apply_multiscale_retinex_shadow_removal(
    image: np.ndarray,
    sigma_list: list = None,
) -> np.ndarray:
    """
    Multi-scale Retinex for more robust shadow removal.
    Averages reflectance maps at multiple scales.
    """
    if sigma_list is None:
        sigma_list = [15, 80, 250]

    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)
    l_float = l_channel.astype(np.float64) + 1.0

    reflectance_sum = np.zeros_like(l_float)

    for sigma in sigma_list:
        ksize = sigma * 2 + 1
        if ksize % 2 == 0:
            ksize += 1
        illumination = cv2.GaussianBlur(l_float, (ksize, ksize), sigma)
        illumination = np.maximum(illumination, 1.0)

        reflectance = np.log(l_float) - np.log(illumination)
        reflectance_sum += reflectance

    avg_reflectance = reflectance_sum / len(sigma_list)

    r_min, r_max = avg_reflectance.min(), avg_reflectance.max()
    if r_max > r_min:
        normalized = (avg_reflectance - r_min) / (r_max - r_min) * 255.0
    else:
        normalized = np.full_like(avg_reflectance, 128.0)

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    merged = merge_lab_channels(normalized, a_channel, b_channel)
    return lab_to_bgr(merged)


def apply_gamma_shadow_correction(
    image: np.ndarray,
    shadow_map: ShadowMap,
    gamma: float = ecfg.SHADOW_REMOVAL_GAMMA,
) -> np.ndarray:
    """
    Apply a gentle gamma correction specifically to shadow regions.
    """
    if shadow_map.shadow_mask is None:
        return image.copy()

    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)

    l_float = l_channel.astype(np.float64) / 255.0
    l_gamma = np.power(l_float, gamma) * 255.0
    l_gamma = np.clip(l_gamma, 0, 255).astype(np.uint8)

    intensity_map = shadow_map.shadow_intensity_map
    if intensity_map is None:
        mask_float = shadow_map.shadow_mask.astype(np.float64) / 255.0
        ksize = 21
        intensity_map = cv2.GaussianBlur(mask_float, (ksize, ksize), 0)

    blend = intensity_map if intensity_map.max() <= 1.0 else intensity_map / 255.0
    shadow_blend = 1.0 - blend

    l_result = (l_gamma.astype(np.float64) * shadow_blend +
                l_channel.astype(np.float64) * blend)
    l_result = np.clip(l_result, 0, 255).astype(np.uint8)

    merged = merge_lab_channels(l_result, a_channel, b_channel)
    return lab_to_bgr(merged)


def remove_shadows(
    image: np.ndarray,
    shadow_map: Optional[ShadowMap] = None,
) -> np.ndarray:
    """
    Full shadow removal pipeline:
    1. Detect shadows (if map not provided)
    2. Illumination compensation
    3. Mask-based compensation
    4. Gamma correction on shadow regions
    """
    if shadow_map is None:
        shadow_map = detect_shadows(image)

    if shadow_map.shadow_percentage < 2.0:
        return image.copy()

    result = apply_illumination_compensation(image, shadow_map)
    result = apply_shadow_mask_compensation(result, shadow_map)
    result = apply_gamma_shadow_correction(result, shadow_map)

    return result
