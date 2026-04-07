"""
Gamma Correction Module
-----------------------
Provides adaptive gamma correction where the gamma value is
determined by the zone's lighting category and local brightness.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from enhancement import enhancement_config as ecfg


def build_gamma_lookup_table(gamma: float) -> np.ndarray:
    """
    Build a 256-entry lookup table for gamma correction.
    output = 255 * (input / 255) ^ gamma
    """
    inv_gamma = 1.0 / gamma if gamma != 0 else 1.0
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(ecfg.GAMMA_LOOKUP_TABLE_SIZE)]
    ).astype(np.uint8)
    return table


def apply_gamma_lut(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to an image using a lookup table."""
    lut = build_gamma_lookup_table(gamma)
    corrected = cv2.LUT(image, lut)
    return corrected


def apply_gamma_direct(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction using direct floating point math."""
    if gamma == 1.0:
        return image.copy()

    normalized = image.astype(np.float64) / 255.0
    corrected = np.power(normalized, gamma)
    result = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return result


def compute_adaptive_gamma(
    mean_brightness: float,
    zone_category: str,
) -> float:
    """
    Compute an adaptive gamma value based on the zone's mean brightness
    and its category settings.
    """
    settings = ecfg.GAMMA_SETTINGS.get(zone_category, ecfg.GAMMA_SETTINGS["normal"])

    if not settings["adaptive"]:
        return settings["gamma"]

    base_gamma = settings["gamma"]
    min_gamma = settings["min_gamma"]
    max_gamma = settings["max_gamma"]

    if zone_category in ("dark", "shadow"):
        brightness_factor = mean_brightness / 128.0
        adaptive_gamma = base_gamma * (1.0 + (1.0 - brightness_factor) * 0.3)
        adaptive_gamma = np.clip(adaptive_gamma, min_gamma, max_gamma)
    elif zone_category in ("bright", "overexposed"):
        brightness_factor = mean_brightness / 255.0
        adaptive_gamma = base_gamma * (0.8 + brightness_factor * 0.4)
        adaptive_gamma = np.clip(adaptive_gamma, min_gamma, max_gamma)
    else:
        adaptive_gamma = base_gamma

    return float(adaptive_gamma)


def apply_zone_gamma(
    image: np.ndarray,
    zone_category: str,
    mean_brightness: float,
) -> np.ndarray:
    """Apply gamma correction tuned for the given zone category."""
    gamma = compute_adaptive_gamma(mean_brightness, zone_category)
    corrected = apply_gamma_lut(image, gamma)
    return corrected


def apply_channel_gamma(
    channel: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply gamma correction to a single channel."""
    lut = build_gamma_lookup_table(gamma)
    return cv2.LUT(channel, lut)


def apply_selective_gamma(
    image: np.ndarray,
    gamma: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply gamma only to the masked region, leaving the rest unchanged.
    If mask is None, apply to the whole image.
    """
    corrected = apply_gamma_lut(image, gamma)

    if mask is None:
        return corrected

    mask_3ch = mask
    if len(image.shape) == 3 and len(mask.shape) == 2:
        mask_3ch = np.stack([mask] * 3, axis=-1)

    mask_float = mask_3ch.astype(np.float64) / 255.0
    result = (corrected.astype(np.float64) * mask_float +
              image.astype(np.float64) * (1.0 - mask_float))
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_dual_gamma(
    image: np.ndarray,
    dark_gamma: float,
    bright_gamma: float,
    threshold: int = 128,
) -> np.ndarray:
    """
    Apply different gamma values to dark and bright regions of the image.
    Pixels below threshold get dark_gamma, above get bright_gamma.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    dark_mask = (gray < threshold).astype(np.float64)
    bright_mask = 1.0 - dark_mask

    dark_lut = build_gamma_lookup_table(dark_gamma)
    bright_lut = build_gamma_lookup_table(bright_gamma)

    dark_corrected = cv2.LUT(image, dark_lut)
    bright_corrected = cv2.LUT(image, bright_lut)

    if len(image.shape) == 3:
        dark_mask_3ch = np.stack([dark_mask] * 3, axis=-1)
        bright_mask_3ch = np.stack([bright_mask] * 3, axis=-1)
    else:
        dark_mask_3ch = dark_mask
        bright_mask_3ch = bright_mask

    result = (dark_corrected.astype(np.float64) * dark_mask_3ch +
              bright_corrected.astype(np.float64) * bright_mask_3ch)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_sigmoidal_contrast(
    image: np.ndarray,
    gain: float = 5.0,
    cutoff: float = 0.5,
) -> np.ndarray:
    """
    Apply sigmoidal (S-curve) contrast enhancement.
    This is an alternative to gamma that adds contrast to midtones.
    """
    normalized = image.astype(np.float64) / 255.0
    sigmoid = 1.0 / (1.0 + np.exp(gain * (cutoff - normalized)))
    sigmoid_min = 1.0 / (1.0 + np.exp(gain * cutoff))
    sigmoid_max = 1.0 / (1.0 + np.exp(gain * (cutoff - 1.0)))
    result = (sigmoid - sigmoid_min) / (sigmoid_max - sigmoid_min)
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)
