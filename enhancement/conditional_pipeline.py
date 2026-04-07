"""
Conditional Pipeline Module
---------------------------
Defines per-zone-type enhancement strategies. Each zone category
(dark, shadow, normal, bright, overexposed) gets a tailored
sequence of operations.
"""

import cv2
import numpy as np
from typing import Optional

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import (
    split_lab_channels,
    merge_lab_channels,
    lab_to_bgr,
    bgr_to_lab,
)
from enhancement.histogram_processor import (
    process_histogram_for_zone,
    histogram_stretching,
    apply_zone_clahe,
    blended_histogram_equalization,
)
from enhancement.gamma_corrector import (
    apply_zone_gamma,
    apply_dual_gamma,
    apply_sigmoidal_contrast,
)
from enhancement.noise_reducer import smart_denoise, denoise_zone
from enhancement.sharpening_engine import smart_sharpen, sharpen_zone
from enhancement.shadow_detector import detect_shadows
from enhancement.shadow_remover import remove_shadows
from enhancement.edge_preserving_filter import (
    apply_bilateral_smooth,
    full_edge_preserving_pipeline,
)
from enhancement.region_analyzer import ZoneStatistics


def _extract_and_process_l_channel(
    image: np.ndarray,
    zone_category: str,
    apply_histogram: bool = True,
    apply_stretching: bool = False,
) -> np.ndarray:
    """
    Helper: extract L channel, apply histogram processing, merge back.
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)

    if apply_stretching:
        l_channel = histogram_stretching(l_channel)

    if apply_histogram:
        l_channel = process_histogram_for_zone(
            l_channel, zone_category,
            apply_stretching=apply_stretching,
            apply_equalization=(zone_category == "dark"),
        )

    merged = merge_lab_channels(l_channel, a_channel, b_channel)
    return lab_to_bgr(merged)


def enhance_dark_zone(
    patch: np.ndarray,
    stats: ZoneStatistics,
) -> np.ndarray:
    """
    Enhancement pipeline for DARK zones:
    1. Aggressive noise reduction (dark = noisy)
    2. Histogram stretching + CLAHE (strong)
    3. Low gamma to lift shadows
    4. Moderate sharpening
    5. Edge-preserving smooth
    """
    result = patch.copy()

    result = smart_denoise(result, "dark")

    l_ch, a_ch, b_ch = split_lab_channels(result, use_cache=False)
    l_stretched = histogram_stretching(l_ch)
    l_clahe = apply_zone_clahe(l_stretched, "dark")
    l_eq = blended_histogram_equalization(l_clahe, 0.85, 0.15)
    merged = merge_lab_channels(l_eq, a_ch, b_ch)
    result = lab_to_bgr(merged)

    result = apply_zone_gamma(result, "dark", stats.mean_brightness)

    result = smart_sharpen(result, "dark", stats.noise_estimate)

    result = apply_bilateral_smooth(result, d=5, sigma_color=30, sigma_space=30)

    return result


def enhance_shadow_zone(
    patch: np.ndarray,
    stats: ZoneStatistics,
) -> np.ndarray:
    """
    Enhancement pipeline for SHADOW zones:
    1. Shadow detection & removal
    2. Bilateral denoising
    3. CLAHE (moderate)
    4. Moderate gamma lift
    5. Edge-preserving smooth
    6. Light sharpening
    """
    result = patch.copy()

    shadow_map = detect_shadows(result)
    if shadow_map.shadow_percentage > 5.0:
        result = remove_shadows(result, shadow_map)

    result = denoise_zone(result, "shadow")

    l_ch, a_ch, b_ch = split_lab_channels(result, use_cache=False)
    l_clahe = apply_zone_clahe(l_ch, "shadow")
    merged = merge_lab_channels(l_clahe, a_ch, b_ch)
    result = lab_to_bgr(merged)

    result = apply_zone_gamma(result, "shadow", stats.mean_brightness)

    result = full_edge_preserving_pipeline(
        result, use_guided=False, use_bilateral=True, blend_strength=0.4
    )

    result = smart_sharpen(result, "shadow", stats.noise_estimate)

    return result


def enhance_normal_zone(
    patch: np.ndarray,
    stats: ZoneStatistics,
) -> np.ndarray:
    """
    Enhancement pipeline for NORMAL zones:
    1. Light CLAHE
    2. Gentle gamma
    3. Light sharpening
    """
    result = patch.copy()

    l_ch, a_ch, b_ch = split_lab_channels(result, use_cache=False)
    l_clahe = apply_zone_clahe(l_ch, "normal")
    merged = merge_lab_channels(l_clahe, a_ch, b_ch)
    result = lab_to_bgr(merged)

    result = apply_zone_gamma(result, "normal", stats.mean_brightness)

    result = sharpen_zone(result, "normal")

    return result


def enhance_bright_zone(
    patch: np.ndarray,
    stats: ZoneStatistics,
) -> np.ndarray:
    """
    Enhancement pipeline for BRIGHT zones:
    1. Inverse gamma to tame brightness
    2. Light CLAHE to preserve detail
    3. No sharpening (avoid amplifying already-sharp bright areas)
    """
    result = patch.copy()

    result = apply_zone_gamma(result, "bright", stats.mean_brightness)

    l_ch, a_ch, b_ch = split_lab_channels(result, use_cache=False)
    l_clahe = apply_zone_clahe(l_ch, "bright")
    merged = merge_lab_channels(l_clahe, a_ch, b_ch)
    result = lab_to_bgr(merged)

    return result


def enhance_overexposed_zone(
    patch: np.ndarray,
    stats: ZoneStatistics,
) -> np.ndarray:
    """
    Enhancement pipeline for OVEREXPOSED zones:
    1. Strong gamma to compress highlights
    2. Histogram stretching to recover detail
    3. CLAHE with low clip to avoid further blowout
    4. Sigmoidal contrast for midtone recovery
    """
    result = patch.copy()

    result = apply_zone_gamma(result, "overexposed", stats.mean_brightness)

    l_ch, a_ch, b_ch = split_lab_channels(result, use_cache=False)
    l_stretched = histogram_stretching(l_ch)
    l_clahe = apply_zone_clahe(l_stretched, "overexposed")
    merged = merge_lab_channels(l_clahe, a_ch, b_ch)
    result = lab_to_bgr(merged)

    result = apply_sigmoidal_contrast(result, gain=4.0, cutoff=0.6)

    return result


def enhance_zone_by_category(
    patch: np.ndarray,
    zone_category: str,
    stats: ZoneStatistics,
) -> np.ndarray:
    """
    Route a zone patch to the correct enhancement pipeline
    based on its classification.
    """
    category_handlers = {
        "dark": enhance_dark_zone,
        "shadow": enhance_shadow_zone,
        "normal": enhance_normal_zone,
        "bright": enhance_bright_zone,
        "overexposed": enhance_overexposed_zone,
    }

    handler = category_handlers.get(zone_category, enhance_normal_zone)
    enhanced_patch = handler(patch, stats)

    return enhanced_patch


def enhance_all_zone_patches(
    patches: list,
    categories: list,
    stats_list: list,
) -> list:
    """
    Enhance all zone patches according to their categories.
    Returns a list of enhanced patches in the same order.
    """
    enhanced_patches = []
    for i, (patch, category, stats) in enumerate(
        zip(patches, categories, stats_list)
    ):
        cat_name = category.category if hasattr(category, "category") else category
        enhanced = enhance_zone_by_category(patch, cat_name, stats)
        enhanced_patches.append(enhanced)

    return enhanced_patches
