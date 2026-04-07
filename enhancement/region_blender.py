"""
Region Blender Module
---------------------
Provides Gaussian-feathered blending of independently enhanced zones
back into a seamless full image, avoiding visible tile boundaries.
"""

import cv2
import numpy as np
from typing import List, Tuple

from enhancement import enhancement_config as ecfg
from enhancement.region_analyzer import ZoneCoordinates


def create_feather_mask(
    height: int,
    width: int,
    feather_size: int = ecfg.BLEND_FEATHER_SIZE,
) -> np.ndarray:
    """
    Create a rectangular mask with feathered (soft) edges.
    Interior pixels = 1.0, edge pixels ramp from 0.0 to 1.0.
    """
    mask = np.ones((height, width), dtype=np.float64)

    for i in range(feather_size):
        weight = (i + 1) / feather_size
        if i < height:
            mask[i, :] = np.minimum(mask[i, :], weight)
            mask[height - 1 - i, :] = np.minimum(mask[height - 1 - i, :], weight)
        if i < width:
            mask[:, i] = np.minimum(mask[:, i], weight)
            mask[:, width - 1 - i] = np.minimum(mask[:, width - 1 - i], weight)

    return mask


def create_gaussian_feather_mask(
    height: int,
    width: int,
    sigma: float = ecfg.BLEND_GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    Create a 2D Gaussian-weighted mask centered on the zone.
    Provides smoother transitions than linear feathering.
    """
    y_center = height / 2.0
    x_center = width / 2.0

    y_coords = np.arange(height).astype(np.float64)
    x_coords = np.arange(width).astype(np.float64)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    dist_y = np.abs(yy - y_center) / y_center
    dist_x = np.abs(xx - x_center) / x_center
    dist = np.maximum(dist_y, dist_x)

    mask = np.exp(-(dist ** 2) / (2 * (sigma / max(height, width)) ** 2))

    mask = mask / mask.max() if mask.max() > 0 else mask
    mask = np.clip(mask, 0.3, 1.0)

    return mask


def create_zone_weight_masks(
    image_height: int,
    image_width: int,
    zone_coords: List[ZoneCoordinates],
    method: str = ecfg.BLEND_METHOD,
) -> List[np.ndarray]:
    """
    Create a weight mask for each zone for use in weighted blending.
    """
    masks = []
    for coords in zone_coords:
        zone_h = coords.y_end - coords.y_start
        zone_w = coords.x_end - coords.x_start

        if method == "gaussian":
            mask = create_gaussian_feather_mask(zone_h, zone_w)
        else:
            mask = create_feather_mask(zone_h, zone_w)

        masks.append(mask)
    return masks


def normalize_weight_masks(
    image_height: int,
    image_width: int,
    zone_coords: List[ZoneCoordinates],
    weight_masks: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Normalize zone weight masks so they sum to 1.0 at every pixel.
    """
    accumulator = np.zeros((image_height, image_width), dtype=np.float64)

    for coords, mask in zip(zone_coords, weight_masks):
        accumulator[
            coords.y_start : coords.y_end,
            coords.x_start : coords.x_end,
        ] += mask

    accumulator = np.maximum(accumulator, 1e-8)

    normalized_masks = []
    for coords, mask in zip(zone_coords, weight_masks):
        region_sum = accumulator[
            coords.y_start : coords.y_end,
            coords.x_start : coords.x_end,
        ]
        normalized = mask / region_sum
        normalized_masks.append(normalized)

    return normalized_masks


def blend_zones_weighted(
    image_height: int,
    image_width: int,
    enhanced_patches: List[np.ndarray],
    zone_coords: List[ZoneCoordinates],
    weight_masks: List[np.ndarray],
) -> np.ndarray:
    """
    Blend enhanced zone patches into a full image using weight masks.
    """
    result = np.zeros((image_height, image_width, 3), dtype=np.float64)
    weight_sum = np.zeros((image_height, image_width), dtype=np.float64)

    for patch, coords, mask in zip(enhanced_patches, zone_coords, weight_masks):
        y1 = coords.y_start
        y2 = coords.y_end
        x1 = coords.x_start
        x2 = coords.x_end

        mask_3ch = np.stack([mask] * 3, axis=-1)
        result[y1:y2, x1:x2] += patch.astype(np.float64) * mask_3ch
        weight_sum[y1:y2, x1:x2] += mask

    weight_sum_3ch = np.stack([weight_sum] * 3, axis=-1)
    weight_sum_3ch = np.maximum(weight_sum_3ch, 1e-8)

    result = result / weight_sum_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_zones_simple(
    image_height: int,
    image_width: int,
    enhanced_patches: List[np.ndarray],
    zone_coords: List[ZoneCoordinates],
) -> np.ndarray:
    """
    Simple direct placement of zones (no feathering).
    Used as a fallback when blending is disabled.
    """
    result = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for patch, coords in zip(enhanced_patches, zone_coords):
        result[
            coords.y_start : coords.y_end,
            coords.x_start : coords.x_end,
        ] = patch

    return result


def blend_zones_crossfade(
    image_height: int,
    image_width: int,
    original_image: np.ndarray,
    enhanced_patches: List[np.ndarray],
    zone_coords: List[ZoneCoordinates],
    crossfade_pixels: int = ecfg.BLEND_OVERLAP_PIXELS,
) -> np.ndarray:
    """
    Blend zones with a crossfade in the overlap region between
    the original and enhanced versions.
    """
    direct = blend_zones_simple(image_height, image_width,
                                 enhanced_patches, zone_coords)

    fade_mask = np.ones((image_height, image_width), dtype=np.float64)

    for coords in zone_coords:
        for i in range(crossfade_pixels):
            weight = (i + 1) / crossfade_pixels
            y1 = coords.y_start + i
            y2 = coords.y_end - 1 - i
            x1 = coords.x_start + i
            x2 = coords.x_end - 1 - i

            if y1 < image_height:
                fade_mask[y1, coords.x_start:coords.x_end] = min(
                    fade_mask[y1, coords.x_start], weight
                )
            if y2 >= 0:
                fade_mask[y2, coords.x_start:coords.x_end] = min(
                    fade_mask[y2, coords.x_start], weight
                )
            if x1 < image_width:
                fade_mask[coords.y_start:coords.y_end, x1] = np.minimum(
                    fade_mask[coords.y_start:coords.y_end, x1], weight
                )
            if x2 >= 0:
                fade_mask[coords.y_start:coords.y_end, x2] = np.minimum(
                    fade_mask[coords.y_start:coords.y_end, x2], weight
                )

    fade_3ch = np.stack([fade_mask] * 3, axis=-1)
    result = (direct.astype(np.float64) * fade_3ch +
              original_image.astype(np.float64) * (1.0 - fade_3ch))

    return np.clip(result, 0, 255).astype(np.uint8)


def blend_enhanced_zones(
    original_image: np.ndarray,
    enhanced_patches: List[np.ndarray],
    zone_coords: List[ZoneCoordinates],
    method: str = ecfg.BLEND_METHOD,
) -> np.ndarray:
    """
    Main entry point: blend all enhanced zone patches back into
    a full image using the configured blending method.
    """
    image_height, image_width = original_image.shape[:2]

    if method == "simple":
        return blend_zones_simple(image_height, image_width,
                                   enhanced_patches, zone_coords)

    if method == "crossfade":
        return blend_zones_crossfade(
            image_height, image_width, original_image,
            enhanced_patches, zone_coords
        )

    weight_masks = create_zone_weight_masks(
        image_height, image_width, zone_coords, method
    )
    normalized_masks = normalize_weight_masks(
        image_height, image_width, zone_coords, weight_masks
    )

    return blend_zones_weighted(
        image_height, image_width,
        enhanced_patches, zone_coords, normalized_masks
    )
