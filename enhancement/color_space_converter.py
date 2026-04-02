"""
Color Space Converter Module
-----------------------------
Provides BGR <-> LAB, HSV, YCrCb conversions with an in-memory
caching layer so repeated conversions on the same image are free.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict


class ColorSpaceCache:
    """
    Caches color space conversions for a single image to avoid
    redundant re-computation during the enhancement pipeline.
    """

    def __init__(self):
        self._bgr_image: Optional[np.ndarray] = None
        self._image_id: Optional[int] = None
        self._cache: Dict[str, np.ndarray] = {}

    def _get_image_id(self, image: np.ndarray) -> int:
        """Generate a quick identity hash for cache invalidation."""
        return id(image)

    def set_source(self, bgr_image: np.ndarray) -> None:
        """Set a new source BGR image and clear any stale cache entries."""
        new_id = self._get_image_id(bgr_image)
        if new_id != self._image_id:
            self._bgr_image = bgr_image
            self._image_id = new_id
            self._cache.clear()
            self._cache["bgr"] = bgr_image

    def get_cached(self, color_space: str) -> Optional[np.ndarray]:
        """Return a cached conversion if available."""
        return self._cache.get(color_space)

    def store(self, color_space: str, image: np.ndarray) -> None:
        """Store a conversion result in the cache."""
        self._cache[color_space] = image

    def clear(self) -> None:
        """Flush all cached conversions."""
        self._cache.clear()
        self._bgr_image = None
        self._image_id = None


_global_cache = ColorSpaceCache()


def reset_cache() -> None:
    """Reset the global conversion cache."""
    _global_cache.clear()


def set_source_image(bgr_image: np.ndarray) -> None:
    """Register the current working image with the cache."""
    _global_cache.set_source(bgr_image)


# -----------------------------------------------------------------------
# BGR -> other color spaces
# -----------------------------------------------------------------------

def bgr_to_lab(image: np.ndarray, use_cache: bool = True) -> np.ndarray:
    """Convert BGR image to CIE LAB color space."""
    if use_cache:
        cached = _global_cache.get_cached("lab")
        if cached is not None:
            return cached

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    if use_cache:
        _global_cache.store("lab", lab)
    return lab


def bgr_to_hsv(image: np.ndarray, use_cache: bool = True) -> np.ndarray:
    """Convert BGR image to HSV color space."""
    if use_cache:
        cached = _global_cache.get_cached("hsv")
        if cached is not None:
            return cached

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if use_cache:
        _global_cache.store("hsv", hsv)
    return hsv


def bgr_to_ycrcb(image: np.ndarray, use_cache: bool = True) -> np.ndarray:
    """Convert BGR image to YCrCb color space."""
    if use_cache:
        cached = _global_cache.get_cached("ycrcb")
        if cached is not None:
            return cached

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    if use_cache:
        _global_cache.store("ycrcb", ycrcb)
    return ycrcb


def bgr_to_gray(image: np.ndarray, use_cache: bool = True) -> np.ndarray:
    """Convert BGR image to single-channel grayscale."""
    if use_cache:
        cached = _global_cache.get_cached("gray")
        if cached is not None:
            return cached

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_cache:
        _global_cache.store("gray", gray)
    return gray


def bgr_to_rgb(image: np.ndarray, use_cache: bool = True) -> np.ndarray:
    """Convert BGR to RGB (useful for MTCNN which expects RGB)."""
    if use_cache:
        cached = _global_cache.get_cached("rgb")
        if cached is not None:
            return cached

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if use_cache:
        _global_cache.store("rgb", rgb)
    return rgb


# -----------------------------------------------------------------------
# Other color spaces -> BGR
# -----------------------------------------------------------------------

def lab_to_bgr(lab_image: np.ndarray) -> np.ndarray:
    """Convert LAB image back to BGR."""
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)


def hsv_to_bgr(hsv_image: np.ndarray) -> np.ndarray:
    """Convert HSV image back to BGR."""
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def ycrcb_to_bgr(ycrcb_image: np.ndarray) -> np.ndarray:
    """Convert YCrCb image back to BGR."""
    return cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)


def gray_to_bgr(gray_image: np.ndarray) -> np.ndarray:
    """Convert single-channel grayscale to 3-channel BGR."""
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


def rgb_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    """Convert RGB back to BGR."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------
# Channel splitting utilities
# -----------------------------------------------------------------------

def split_lab_channels(
    image: np.ndarray, use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert to LAB and return (L, A, B) channels."""
    lab = bgr_to_lab(image, use_cache=use_cache)
    l_channel, a_channel, b_channel = cv2.split(lab)
    return l_channel, a_channel, b_channel


def split_hsv_channels(
    image: np.ndarray, use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert to HSV and return (H, S, V) channels."""
    hsv = bgr_to_hsv(image, use_cache=use_cache)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    return h_channel, s_channel, v_channel


def split_ycrcb_channels(
    image: np.ndarray, use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert to YCrCb and return (Y, Cr, Cb) channels."""
    ycrcb = bgr_to_ycrcb(image, use_cache=use_cache)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    return y_channel, cr_channel, cb_channel


def split_bgr_channels(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split BGR image into (B, G, R) channels."""
    b_channel, g_channel, r_channel = cv2.split(image)
    return b_channel, g_channel, r_channel


# -----------------------------------------------------------------------
# Channel merging utilities
# -----------------------------------------------------------------------

def merge_lab_channels(
    l_channel: np.ndarray, a_channel: np.ndarray, b_channel: np.ndarray
) -> np.ndarray:
    """Merge L, A, B channels into a LAB image."""
    return cv2.merge([l_channel, a_channel, b_channel])


def merge_hsv_channels(
    h_channel: np.ndarray, s_channel: np.ndarray, v_channel: np.ndarray
) -> np.ndarray:
    """Merge H, S, V channels into an HSV image."""
    return cv2.merge([h_channel, s_channel, v_channel])


def merge_ycrcb_channels(
    y_channel: np.ndarray, cr_channel: np.ndarray, cb_channel: np.ndarray
) -> np.ndarray:
    """Merge Y, Cr, Cb channels into a YCrCb image."""
    return cv2.merge([y_channel, cr_channel, cb_channel])


def merge_bgr_channels(
    b_channel: np.ndarray, g_channel: np.ndarray, r_channel: np.ndarray
) -> np.ndarray:
    """Merge B, G, R channels back into a BGR image."""
    return cv2.merge([b_channel, g_channel, r_channel])


# -----------------------------------------------------------------------
# Convenience: replace single channel and convert back
# -----------------------------------------------------------------------

def replace_l_channel(image: np.ndarray, new_l: np.ndarray) -> np.ndarray:
    """Replace the L channel in a BGR image and return the updated BGR."""
    lab = bgr_to_lab(image, use_cache=False)
    _, a, b = cv2.split(lab)
    merged = merge_lab_channels(new_l, a, b)
    return lab_to_bgr(merged)


def replace_v_channel(image: np.ndarray, new_v: np.ndarray) -> np.ndarray:
    """Replace the V channel in a BGR image and return the updated BGR."""
    hsv = bgr_to_hsv(image, use_cache=False)
    h, s, _ = cv2.split(hsv)
    merged = merge_hsv_channels(h, s, new_v)
    return hsv_to_bgr(merged)


def replace_y_channel(image: np.ndarray, new_y: np.ndarray) -> np.ndarray:
    """Replace the Y channel in a BGR image and return the updated BGR."""
    ycrcb = bgr_to_ycrcb(image, use_cache=False)
    _, cr, cb = cv2.split(ycrcb)
    merged = merge_ycrcb_channels(new_y, cr, cb)
    return ycrcb_to_bgr(merged)
