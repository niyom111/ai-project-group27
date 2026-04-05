"""
Region Analyzer Module
----------------------
Splits an image into an NxN grid of zones and computes per-zone
statistics: mean brightness, contrast, standard deviation, histogram
skewness, gradient variance, and noise estimate.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import bgr_to_lab, bgr_to_gray


class ZoneCoordinates:
    """Stores the pixel coordinates of a single grid zone."""

    def __init__(self, row: int, col: int, y_start: int, y_end: int,
                 x_start: int, x_end: int):
        self.row = row
        self.col = col
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end
        self.width = x_end - x_start
        self.height = y_end - y_start

    def to_dict(self) -> Dict[str, int]:
        return {
            "row": self.row,
            "col": self.col,
            "y_start": self.y_start,
            "y_end": self.y_end,
            "x_start": self.x_start,
            "x_end": self.x_end,
            "width": self.width,
            "height": self.height,
        }


class ZoneStatistics:
    """Statistical summary for a single image zone."""

    def __init__(self):
        self.zone_id: int = 0
        self.coordinates: ZoneCoordinates = None
        self.mean_brightness: float = 0.0
        self.std_brightness: float = 0.0
        self.min_brightness: int = 0
        self.max_brightness: int = 255
        self.median_brightness: float = 0.0
        self.contrast: float = 0.0
        self.dynamic_range: int = 0
        self.histogram: np.ndarray = np.array([])
        self.histogram_skewness: float = 0.0
        self.histogram_kurtosis: float = 0.0
        self.gradient_variance: float = 0.0
        self.gradient_mean: float = 0.0
        self.noise_estimate: float = 0.0
        self.entropy: float = 0.0
        self.percentile_5: float = 0.0
        self.percentile_95: float = 0.0
        self.saturation_mean: float = 0.0
        self.saturation_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "coordinates": self.coordinates.to_dict() if self.coordinates else None,
            "mean_brightness": round(self.mean_brightness, 2),
            "std_brightness": round(self.std_brightness, 2),
            "min_brightness": self.min_brightness,
            "max_brightness": self.max_brightness,
            "median_brightness": round(self.median_brightness, 2),
            "contrast": round(self.contrast, 2),
            "dynamic_range": self.dynamic_range,
            "histogram_skewness": round(self.histogram_skewness, 3),
            "histogram_kurtosis": round(self.histogram_kurtosis, 3),
            "gradient_variance": round(self.gradient_variance, 3),
            "gradient_mean": round(self.gradient_mean, 3),
            "noise_estimate": round(self.noise_estimate, 3),
            "entropy": round(self.entropy, 3),
            "percentile_5": round(self.percentile_5, 2),
            "percentile_95": round(self.percentile_95, 2),
            "saturation_mean": round(self.saturation_mean, 2),
            "saturation_std": round(self.saturation_std, 2),
        }


def compute_grid_coordinates(
    image_height: int,
    image_width: int,
    grid_rows: int = ecfg.GRID_ROWS,
    grid_cols: int = ecfg.GRID_COLS,
) -> List[ZoneCoordinates]:
    """Divide the image into a grid and return the coordinates of each zone."""
    zone_height = image_height // grid_rows
    zone_width = image_width // grid_cols

    zones: List[ZoneCoordinates] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            y_start = r * zone_height
            y_end = (r + 1) * zone_height if r < grid_rows - 1 else image_height
            x_start = c * zone_width
            x_end = (c + 1) * zone_width if c < grid_cols - 1 else image_width

            zones.append(ZoneCoordinates(r, c, y_start, y_end, x_start, x_end))

    return zones


def extract_zone_patch(image: np.ndarray, coords: ZoneCoordinates) -> np.ndarray:
    """Extract the pixel region for a given zone from the image."""
    return image[coords.y_start : coords.y_end, coords.x_start : coords.x_end].copy()


def compute_zone_histogram(l_patch: np.ndarray) -> np.ndarray:
    """Compute the brightness histogram for a single-channel zone patch."""
    hist = cv2.calcHist(
        [l_patch], [0], None,
        [ecfg.HISTOGRAM_BINS],
        [ecfg.HISTOGRAM_RANGE_MIN, ecfg.HISTOGRAM_RANGE_MAX],
    )
    return hist.flatten()


def compute_histogram_skewness(histogram: np.ndarray) -> float:
    """Compute skewness of a brightness histogram."""
    total = np.sum(histogram)
    if total == 0:
        return 0.0
    bins = np.arange(len(histogram))
    mean_val = np.sum(bins * histogram) / total
    variance = np.sum(((bins - mean_val) ** 2) * histogram) / total
    std_val = np.sqrt(variance) if variance > 0 else 1.0
    skew = np.sum(((bins - mean_val) ** 3) * histogram) / (total * (std_val ** 3))
    return float(skew)


def compute_histogram_kurtosis(histogram: np.ndarray) -> float:
    """Compute excess kurtosis of a brightness histogram."""
    total = np.sum(histogram)
    if total == 0:
        return 0.0
    bins = np.arange(len(histogram))
    mean_val = np.sum(bins * histogram) / total
    variance = np.sum(((bins - mean_val) ** 2) * histogram) / total
    std_val = np.sqrt(variance) if variance > 0 else 1.0
    kurt = np.sum(((bins - mean_val) ** 4) * histogram) / (total * (std_val ** 4))
    return float(kurt - 3.0)


def compute_gradient_stats(gray_patch: np.ndarray) -> Tuple[float, float]:
    """Compute gradient magnitude mean and variance (Sobel-based)."""
    grad_x = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.mean(magnitude)), float(np.var(magnitude))


def compute_noise_estimate(gray_patch: np.ndarray) -> float:
    """
    Estimate noise level using the Laplacian method.
    Higher values indicate more noise.
    """
    laplacian = cv2.Laplacian(gray_patch, cv2.CV_64F)
    sigma = float(np.std(laplacian))
    return sigma


def compute_entropy(histogram: np.ndarray) -> float:
    """Compute Shannon entropy of the brightness histogram."""
    total = np.sum(histogram)
    if total == 0:
        return 0.0
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def compute_zone_statistics(
    bgr_patch: np.ndarray,
    zone_id: int,
    coords: ZoneCoordinates,
) -> ZoneStatistics:
    """Compute all statistics for a single zone patch."""
    stats = ZoneStatistics()
    stats.zone_id = zone_id
    stats.coordinates = coords

    lab_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB)
    l_channel = lab_patch[:, :, 0]

    stats.mean_brightness = float(np.mean(l_channel))
    stats.std_brightness = float(np.std(l_channel))
    stats.min_brightness = int(np.min(l_channel))
    stats.max_brightness = int(np.max(l_channel))
    stats.median_brightness = float(np.median(l_channel))
    stats.contrast = stats.std_brightness
    stats.dynamic_range = stats.max_brightness - stats.min_brightness

    stats.percentile_5 = float(np.percentile(l_channel, 5))
    stats.percentile_95 = float(np.percentile(l_channel, 95))

    stats.histogram = compute_zone_histogram(l_channel)
    stats.histogram_skewness = compute_histogram_skewness(stats.histogram)
    stats.histogram_kurtosis = compute_histogram_kurtosis(stats.histogram)
    stats.entropy = compute_entropy(stats.histogram)

    gray_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
    stats.gradient_mean, stats.gradient_variance = compute_gradient_stats(gray_patch)
    stats.noise_estimate = compute_noise_estimate(gray_patch)

    hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    s_channel = hsv_patch[:, :, 1]
    stats.saturation_mean = float(np.mean(s_channel))
    stats.saturation_std = float(np.std(s_channel))

    return stats


def analyze_all_zones(
    image: np.ndarray,
    grid_rows: int = ecfg.GRID_ROWS,
    grid_cols: int = ecfg.GRID_COLS,
) -> Tuple[List[ZoneCoordinates], List[ZoneStatistics]]:
    """
    Split the image into grid zones and compute statistics for each.
    Returns parallel lists of (coordinates, statistics).
    """
    height, width = image.shape[:2]
    zone_coords = compute_grid_coordinates(height, width, grid_rows, grid_cols)
    zone_stats: List[ZoneStatistics] = []

    for idx, coords in enumerate(zone_coords):
        patch = extract_zone_patch(image, coords)
        stats = compute_zone_statistics(patch, idx, coords)
        zone_stats.append(stats)

    return zone_coords, zone_stats


def get_zone_patches(
    image: np.ndarray,
    zone_coords: List[ZoneCoordinates],
) -> List[np.ndarray]:
    """Extract all zone patches from the image given precomputed coordinates."""
    patches = []
    for coords in zone_coords:
        patch = extract_zone_patch(image, coords)
        patches.append(patch)
    return patches


def compute_global_statistics(zone_stats: List[ZoneStatistics]) -> Dict[str, float]:
    """Aggregate per-zone stats into global image-level statistics."""
    if not zone_stats:
        return {}

    all_means = [z.mean_brightness for z in zone_stats]
    all_stds = [z.std_brightness for z in zone_stats]
    all_contrasts = [z.contrast for z in zone_stats]
    all_gradients = [z.gradient_variance for z in zone_stats]
    all_noise = [z.noise_estimate for z in zone_stats]

    return {
        "global_mean_brightness": float(np.mean(all_means)),
        "global_std_brightness": float(np.mean(all_stds)),
        "brightness_range_across_zones": float(max(all_means) - min(all_means)),
        "mean_contrast": float(np.mean(all_contrasts)),
        "mean_gradient_variance": float(np.mean(all_gradients)),
        "mean_noise_estimate": float(np.mean(all_noise)),
        "brightness_uniformity": 1.0 - (float(np.std(all_means)) / 128.0),
        "num_zones": len(zone_stats),
    }
