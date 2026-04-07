"""
Tone Mapper Module
------------------
Provides Reinhard and Drago tone mapping for HDR-like dynamic range
compression, useful for scenes with simultaneous bright windows and
dark corners.
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


def apply_reinhard_tone_map(
    image: np.ndarray,
    intensity: float = ecfg.TONE_MAP_REINHARD_INTENSITY,
    light_adapt: float = ecfg.TONE_MAP_REINHARD_LIGHT_ADAPT,
    color_adapt: float = ecfg.TONE_MAP_REINHARD_COLOR_ADAPT,
) -> np.ndarray:
    """
    Apply Reinhard's tone mapping operator.
    Converts to float32 HDR-like representation, applies the operator,
    then converts back to 8-bit.
    """
    hdr_image = image.astype(np.float32) / 255.0

    tonemap = cv2.createTonemapReinhard(
        gamma=1.0,
        intensity=intensity,
        light_adapt=light_adapt,
        color_adapt=color_adapt,
    )

    ldr = tonemap.process(hdr_image)
    ldr = np.clip(ldr * 255.0, 0, 255).astype(np.uint8)
    return ldr


def apply_drago_tone_map(
    image: np.ndarray,
    saturation: float = ecfg.TONE_MAP_DRAGO_SATURATION,
    bias: float = ecfg.TONE_MAP_DRAGO_BIAS,
    gamma: float = ecfg.TONE_MAP_DRAGO_GAMMA,
) -> np.ndarray:
    """
    Apply Drago's logarithmic tone mapping operator.
    Good at preserving details in both bright and dark regions.
    """
    hdr_image = image.astype(np.float32) / 255.0

    tonemap = cv2.createTonemapDrago(
        gamma=gamma,
        saturation=saturation,
        bias=bias,
    )

    ldr = tonemap.process(hdr_image)
    ldr = np.clip(ldr * 255.0, 0, 255).astype(np.uint8)
    return ldr


def apply_mantiuk_tone_map(
    image: np.ndarray,
    saturation: float = 1.0,
    scale: float = 0.7,
) -> np.ndarray:
    """
    Apply Mantiuk's tone mapping which preserves perceived contrast.
    """
    hdr_image = image.astype(np.float32) / 255.0
    hdr_image = np.maximum(hdr_image, 1e-6)

    tonemap = cv2.createTonemapMantiuk(
        gamma=1.0,
        saturation=saturation,
        scale=scale,
    )

    ldr = tonemap.process(hdr_image)
    ldr = np.clip(ldr * 255.0, 0, 255).astype(np.uint8)
    return ldr


def apply_simple_local_tone_map(
    image: np.ndarray,
    sigma: int = 30,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Simple local tone mapping by dividing by the local mean.
    Compresses dynamic range by normalizing local brightness.
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)
    l_float = l_channel.astype(np.float64) + 1.0

    ksize = sigma * 2 + 1
    if ksize % 2 == 0:
        ksize += 1
    local_mean = cv2.GaussianBlur(l_float, (ksize, ksize), sigma)
    local_mean = np.maximum(local_mean, 1.0)

    global_mean = np.mean(l_float)
    tone_mapped = l_float * (global_mean / local_mean)

    result = l_float * (1.0 - strength) + tone_mapped * strength
    result = np.clip(result - 1.0, 0, 255).astype(np.uint8)

    merged = merge_lab_channels(result, a_channel, b_channel)
    return lab_to_bgr(merged)


def apply_logarithmic_tone_map(
    image: np.ndarray,
    base: float = 10.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Apply logarithmic tone mapping:
    output = scale * log(1 + input) / log(1 + max_input)
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)
    l_float = l_channel.astype(np.float64)

    l_max = np.max(l_float)
    if l_max <= 0:
        return image.copy()

    log_numerator = np.log(1.0 + l_float) * scale
    log_denominator = np.log(1.0 + l_max)

    if log_denominator > 0:
        tone_mapped = (log_numerator / log_denominator) * 255.0
    else:
        tone_mapped = l_float

    tone_mapped = np.clip(tone_mapped, 0, 255).astype(np.uint8)

    merged = merge_lab_channels(tone_mapped, a_channel, b_channel)
    return lab_to_bgr(merged)


def apply_exponential_tone_map(
    image: np.ndarray,
    exponent: float = 0.5,
) -> np.ndarray:
    """
    Apply exponential tone mapping for gentle dynamic range compression.
    output = 1 - exp(-input * exponent)
    """
    l_channel, a_channel, b_channel = split_lab_channels(image, use_cache=False)
    l_normalized = l_channel.astype(np.float64) / 255.0

    tone_mapped = 1.0 - np.exp(-l_normalized * exponent * 5.0)
    tone_mapped = np.clip(tone_mapped * 255.0, 0, 255).astype(np.uint8)

    merged = merge_lab_channels(tone_mapped, a_channel, b_channel)
    return lab_to_bgr(merged)


def apply_adaptive_tone_map(
    image: np.ndarray,
    mean_brightness: float,
) -> np.ndarray:
    """
    Choose tone mapping strategy based on image characteristics.
    Dark images use logarithmic (lifts shadows), bright images use
    Reinhard (compresses highlights).
    """
    if mean_brightness < 80:
        return apply_logarithmic_tone_map(image, scale=1.2)
    elif mean_brightness > 180:
        return apply_reinhard_tone_map(image, intensity=0.5)
    else:
        return apply_simple_local_tone_map(image, strength=0.4)


def apply_blended_tone_map(
    image: np.ndarray,
    method: str = ecfg.TONE_MAP_METHOD,
    blend_strength: float = ecfg.TONE_MAP_BLEND_STRENGTH,
) -> np.ndarray:
    """
    Apply tone mapping and blend with the original to avoid over-processing.
    """
    if method == "reinhard":
        tone_mapped = apply_reinhard_tone_map(image)
    elif method == "drago":
        tone_mapped = apply_drago_tone_map(image)
    elif method == "mantiuk":
        tone_mapped = apply_mantiuk_tone_map(image)
    elif method == "local":
        tone_mapped = apply_simple_local_tone_map(image)
    elif method == "logarithmic":
        tone_mapped = apply_logarithmic_tone_map(image)
    elif method == "exponential":
        tone_mapped = apply_exponential_tone_map(image)
    else:
        tone_mapped = apply_reinhard_tone_map(image)

    result = cv2.addWeighted(
        tone_mapped, blend_strength,
        image, 1.0 - blend_strength,
        0,
    )
    return result
