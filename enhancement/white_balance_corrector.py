"""
White Balance Corrector Module
------------------------------
Provides gray-world, max-white, and Retinex-based white balance
correction methods to normalize color casts caused by indoor
lighting.
"""

import cv2
import numpy as np
from typing import Optional, List

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import split_bgr_channels, merge_bgr_channels


def apply_gray_world_balance(
    image: np.ndarray,
    power: int = ecfg.WHITE_BALANCE_GRAY_WORLD_POWER,
) -> np.ndarray:
    """
    Gray-world white balance: assumes the average color of the scene
    should be neutral gray.  Each channel is scaled so its mean
    matches the global average.
    """
    b, g, r = split_bgr_channels(image)

    b_mean = np.mean(b.astype(np.float64))
    g_mean = np.mean(g.astype(np.float64))
    r_mean = np.mean(r.astype(np.float64))

    overall_mean = (b_mean + g_mean + r_mean) / 3.0

    if b_mean == 0:
        b_mean = 1.0
    if g_mean == 0:
        g_mean = 1.0
    if r_mean == 0:
        r_mean = 1.0

    b_scale = overall_mean / b_mean
    g_scale = overall_mean / g_mean
    r_scale = overall_mean / r_mean

    b_balanced = np.clip(b.astype(np.float64) * b_scale, 0, 255).astype(np.uint8)
    g_balanced = np.clip(g.astype(np.float64) * g_scale, 0, 255).astype(np.uint8)
    r_balanced = np.clip(r.astype(np.float64) * r_scale, 0, 255).astype(np.uint8)

    return merge_bgr_channels(b_balanced, g_balanced, r_balanced)


def apply_max_white_balance(
    image: np.ndarray,
    percentile: float = ecfg.WHITE_BALANCE_MAX_WHITE_PERCENTILE,
) -> np.ndarray:
    """
    Max-White white balance: scales each channel so that the
    brightest pixels (at the given percentile) become 255.
    """
    b, g, r = split_bgr_channels(image)

    b_max = np.percentile(b, percentile)
    g_max = np.percentile(g, percentile)
    r_max = np.percentile(r, percentile)

    b_max = max(b_max, 1.0)
    g_max = max(g_max, 1.0)
    r_max = max(r_max, 1.0)

    b_balanced = np.clip(b.astype(np.float64) * (255.0 / b_max), 0, 255).astype(
        np.uint8
    )
    g_balanced = np.clip(g.astype(np.float64) * (255.0 / g_max), 0, 255).astype(
        np.uint8
    )
    r_balanced = np.clip(r.astype(np.float64) * (255.0 / r_max), 0, 255).astype(
        np.uint8
    )

    return merge_bgr_channels(b_balanced, g_balanced, r_balanced)


def apply_single_scale_retinex(
    channel: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Single-scale Retinex on one channel:
    R = log(channel) - log(Gaussian_blur(channel))
    """
    channel_float = channel.astype(np.float64) + 1.0

    ksize = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(channel_float, (ksize, ksize), sigma)
    blurred = np.maximum(blurred, 1.0)

    retinex = np.log(channel_float) - np.log(blurred)
    return retinex


def apply_multiscale_retinex(
    channel: np.ndarray,
    sigma_list: List[float] = None,
) -> np.ndarray:
    """
    Multi-scale Retinex: averages single-scale results across
    multiple blur radii.
    """
    if sigma_list is None:
        sigma_list = ecfg.WHITE_BALANCE_RETINEX_SIGMA_LIST

    retinex_sum = np.zeros_like(channel, dtype=np.float64)
    for sigma in sigma_list:
        retinex_sum += apply_single_scale_retinex(channel, sigma)

    retinex_avg = retinex_sum / len(sigma_list)
    return retinex_avg


def normalize_retinex_output(retinex: np.ndarray) -> np.ndarray:
    """Normalize Retinex output to the 0-255 range."""
    r_min, r_max = retinex.min(), retinex.max()
    if r_max > r_min:
        normalized = (retinex - r_min) / (r_max - r_min) * 255.0
    else:
        normalized = np.full_like(retinex, 128.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_retinex_white_balance(
    image: np.ndarray,
    sigma_list: List[float] = None,
    gain: float = ecfg.WHITE_BALANCE_RETINEX_GAIN,
    offset: float = ecfg.WHITE_BALANCE_RETINEX_OFFSET,
) -> np.ndarray:
    """
    Apply Multi-Scale Retinex with Color Restoration (MSRCR) for
    simultaneous white balance and contrast enhancement.
    """
    if sigma_list is None:
        sigma_list = ecfg.WHITE_BALANCE_RETINEX_SIGMA_LIST

    b, g, r = split_bgr_channels(image)

    b_retinex = apply_multiscale_retinex(b, sigma_list)
    g_retinex = apply_multiscale_retinex(g, sigma_list)
    r_retinex = apply_multiscale_retinex(r, sigma_list)

    intensity = (b.astype(np.float64) + g.astype(np.float64) +
                 r.astype(np.float64) + 1.0)
    b_cr = np.log(125.0 * b.astype(np.float64) / intensity + 1.0)
    g_cr = np.log(125.0 * g.astype(np.float64) / intensity + 1.0)
    r_cr = np.log(125.0 * r.astype(np.float64) / intensity + 1.0)

    b_msrcr = b_cr * b_retinex * gain + offset
    g_msrcr = g_cr * g_retinex * gain + offset
    r_msrcr = r_cr * r_retinex * gain + offset

    b_out = normalize_retinex_output(b_msrcr)
    g_out = normalize_retinex_output(g_msrcr)
    r_out = normalize_retinex_output(r_msrcr)

    return merge_bgr_channels(b_out, g_out, r_out)


def apply_weighted_white_balance(
    image: np.ndarray,
    gray_world_weight: float = 0.5,
    max_white_weight: float = 0.5,
) -> np.ndarray:
    """
    Blend gray-world and max-white results for a more robust correction.
    """
    gw_result = apply_gray_world_balance(image)
    mw_result = apply_max_white_balance(image)

    blended = cv2.addWeighted(
        gw_result, gray_world_weight,
        mw_result, max_white_weight,
        0,
    )
    return blended


def apply_chromatic_adaptation(
    image: np.ndarray,
    source_illuminant: np.ndarray = None,
    target_illuminant: np.ndarray = None,
) -> np.ndarray:
    """
    Simple Von Kries chromatic adaptation: scale each channel by the
    ratio of target to source illuminant.
    """
    b, g, r = split_bgr_channels(image)

    if source_illuminant is None:
        source_illuminant = np.array([
            float(np.mean(b)),
            float(np.mean(g)),
            float(np.mean(r)),
        ])

    if target_illuminant is None:
        mean_val = np.mean(source_illuminant)
        target_illuminant = np.array([mean_val, mean_val, mean_val])

    source_illuminant = np.maximum(source_illuminant, 1.0)

    scale = target_illuminant / source_illuminant

    b_adapted = np.clip(b.astype(np.float64) * scale[0], 0, 255).astype(np.uint8)
    g_adapted = np.clip(g.astype(np.float64) * scale[1], 0, 255).astype(np.uint8)
    r_adapted = np.clip(r.astype(np.float64) * scale[2], 0, 255).astype(np.uint8)

    return merge_bgr_channels(b_adapted, g_adapted, r_adapted)


def correct_white_balance(
    image: np.ndarray,
    method: str = ecfg.WHITE_BALANCE_METHOD,
    blend_strength: float = ecfg.WHITE_BALANCE_BLEND_STRENGTH,
) -> np.ndarray:
    """
    Main entry point: apply white balance correction and blend
    with the original.
    """
    if method == "gray_world":
        corrected = apply_gray_world_balance(image)
    elif method == "max_white":
        corrected = apply_max_white_balance(image)
    elif method == "retinex":
        corrected = apply_retinex_white_balance(image)
    elif method == "weighted":
        corrected = apply_weighted_white_balance(image)
    elif method == "chromatic":
        corrected = apply_chromatic_adaptation(image)
    else:
        corrected = apply_gray_world_balance(image)

    result = cv2.addWeighted(
        corrected, blend_strength,
        image, 1.0 - blend_strength,
        0,
    )
    return result
