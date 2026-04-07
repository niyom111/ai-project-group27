"""
Quality Metrics Module
----------------------
Measures the quality of enhanced images: brightness uniformity,
Laplacian sharpness, face detectability score, and comparison
metrics between original and enhanced versions.
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Any

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import bgr_to_gray, bgr_to_lab
from enhancement.region_analyzer import (
    analyze_all_zones,
    compute_global_statistics,
)


class QualityReport:
    """Container for all quality metrics of an image."""

    def __init__(self):
        self.sharpness_score: float = 0.0
        self.brightness_mean: float = 0.0
        self.brightness_std: float = 0.0
        self.brightness_uniformity: float = 0.0
        self.contrast_score: float = 0.0
        self.dynamic_range: int = 0
        self.entropy: float = 0.0
        self.noise_estimate: float = 0.0
        self.face_count: int = 0
        self.face_detection_scores: List[float] = []
        self.overall_quality: float = 0.0
        self.zone_brightness_variance: float = 0.0
        self.color_cast_magnitude: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharpness_score": round(self.sharpness_score, 2),
            "brightness_mean": round(self.brightness_mean, 2),
            "brightness_std": round(self.brightness_std, 2),
            "brightness_uniformity": round(self.brightness_uniformity, 3),
            "contrast_score": round(self.contrast_score, 2),
            "dynamic_range": self.dynamic_range,
            "entropy": round(self.entropy, 3),
            "noise_estimate": round(self.noise_estimate, 2),
            "face_count": self.face_count,
            "overall_quality": round(self.overall_quality, 2),
            "zone_brightness_variance": round(self.zone_brightness_variance, 2),
            "color_cast_magnitude": round(self.color_cast_magnitude, 3),
        }


def compute_sharpness_laplacian(image: np.ndarray) -> float:
    """
    Compute sharpness using the variance of the Laplacian.
    Higher values indicate a sharper image.
    """
    gray = bgr_to_gray(image, use_cache=False)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(np.var(laplacian))
    return variance


def compute_sharpness_gradient(image: np.ndarray) -> float:
    """
    Compute sharpness using Sobel gradient magnitude.
    """
    gray = bgr_to_gray(image, use_cache=False)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.mean(magnitude))


def compute_sharpness_tenengrad(image: np.ndarray) -> float:
    """
    Tenengrad focus measure based on Sobel gradients.
    """
    gray = bgr_to_gray(image, use_cache=False)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(gx ** 2 + gy ** 2)
    return float(tenengrad)


def compute_brightness_stats(image: np.ndarray) -> Dict[str, float]:
    """Compute brightness mean and standard deviation."""
    gray = bgr_to_gray(image, use_cache=False)
    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "min": float(np.min(gray)),
        "max": float(np.max(gray)),
        "median": float(np.median(gray)),
    }


def compute_brightness_uniformity(image: np.ndarray) -> float:
    """
    Measure how uniform brightness is across the image by
    computing the coefficient of variation of zone-level means.
    1.0 = perfectly uniform, lower = less uniform.
    """
    _, zone_stats = analyze_all_zones(image)
    if not zone_stats:
        return 0.0

    zone_means = [z.mean_brightness for z in zone_stats]
    mean_of_means = np.mean(zone_means)

    if mean_of_means < 1.0:
        return 0.0

    cv = np.std(zone_means) / mean_of_means
    uniformity = max(0.0, 1.0 - cv)
    return float(uniformity)


def compute_zone_brightness_variance(image: np.ndarray) -> float:
    """Variance of mean brightness across zones."""
    _, zone_stats = analyze_all_zones(image)
    if not zone_stats:
        return 0.0
    zone_means = [z.mean_brightness for z in zone_stats]
    return float(np.var(zone_means))


def compute_contrast_score(image: np.ndarray) -> float:
    """
    Compute an overall contrast score using the RMS of pixel deviations.
    """
    gray = bgr_to_gray(image, use_cache=False).astype(np.float64)
    mean_val = np.mean(gray)
    rms_contrast = np.sqrt(np.mean((gray - mean_val) ** 2))
    return float(rms_contrast)


def compute_dynamic_range(image: np.ndarray) -> int:
    """Compute the dynamic range (max - min brightness)."""
    gray = bgr_to_gray(image, use_cache=False)
    return int(np.max(gray)) - int(np.min(gray))


def compute_entropy(image: np.ndarray) -> float:
    """Compute Shannon entropy of the grayscale histogram."""
    gray = bgr_to_gray(image, use_cache=False)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = np.sum(hist)
    if total == 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_noise_estimate(image: np.ndarray) -> float:
    """Estimate noise level via Laplacian std."""
    gray = bgr_to_gray(image, use_cache=False)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.std(laplacian))


def compute_color_cast(image: np.ndarray) -> float:
    """
    Measure color cast as the distance of mean A/B values
    from neutral (128, 128) in LAB space.
    """
    lab = bgr_to_lab(image, use_cache=False)
    a_mean = float(np.mean(lab[:, :, 1]))
    b_mean = float(np.mean(lab[:, :, 2]))
    distance = np.sqrt((a_mean - 128) ** 2 + (b_mean - 128) ** 2)
    return float(distance)


def count_faces(image: np.ndarray) -> int:
    """Count detectable faces. Imports face enhancer lazily."""
    try:
        from enhancement.face_region_enhancer import count_detectable_faces
        return count_detectable_faces(image)
    except Exception:
        return 0


def compute_overall_quality(report: QualityReport) -> float:
    """
    Compute a composite quality score (0-100) from individual metrics.
    Weighted combination favoring sharpness and brightness uniformity.
    """
    sharpness_norm = min(100.0, report.sharpness_score / 10.0)
    uniformity_norm = report.brightness_uniformity * 100.0

    brightness_dist = abs(report.brightness_mean - ecfg.BRIGHTNESS_TARGET_MEAN)
    brightness_norm = max(0.0, 100.0 - brightness_dist * 0.8)

    contrast_norm = min(100.0, report.contrast_score * 2.0)

    face_norm = min(100.0, report.face_count * 20.0)

    overall = (
        sharpness_norm * 0.25
        + uniformity_norm * 0.25
        + brightness_norm * 0.20
        + contrast_norm * 0.15
        + face_norm * 0.15
    )

    return max(0.0, min(100.0, overall))


def generate_quality_report(
    image: np.ndarray,
    detect_faces_flag: bool = True,
) -> QualityReport:
    """
    Generate a comprehensive quality report for an image.
    """
    report = QualityReport()

    report.sharpness_score = compute_sharpness_laplacian(image)

    brightness = compute_brightness_stats(image)
    report.brightness_mean = brightness["mean"]
    report.brightness_std = brightness["std"]

    report.brightness_uniformity = compute_brightness_uniformity(image)
    report.contrast_score = compute_contrast_score(image)
    report.dynamic_range = compute_dynamic_range(image)
    report.entropy = compute_entropy(image)
    report.noise_estimate = compute_noise_estimate(image)
    report.zone_brightness_variance = compute_zone_brightness_variance(image)
    report.color_cast_magnitude = compute_color_cast(image)

    if detect_faces_flag:
        report.face_count = count_faces(image)

    report.overall_quality = compute_overall_quality(report)

    return report


def compare_quality(
    original_report: QualityReport,
    enhanced_report: QualityReport,
) -> Dict[str, Any]:
    """
    Compare quality reports between original and enhanced images.
    Positive deltas mean improvement.
    """
    return {
        "sharpness_delta": round(
            enhanced_report.sharpness_score - original_report.sharpness_score, 2
        ),
        "brightness_mean_delta": round(
            enhanced_report.brightness_mean - original_report.brightness_mean, 2
        ),
        "brightness_uniformity_delta": round(
            enhanced_report.brightness_uniformity
            - original_report.brightness_uniformity, 3
        ),
        "contrast_delta": round(
            enhanced_report.contrast_score - original_report.contrast_score, 2
        ),
        "dynamic_range_delta": (
            enhanced_report.dynamic_range - original_report.dynamic_range
        ),
        "entropy_delta": round(
            enhanced_report.entropy - original_report.entropy, 3
        ),
        "noise_delta": round(
            enhanced_report.noise_estimate - original_report.noise_estimate, 2
        ),
        "face_count_delta": (
            enhanced_report.face_count - original_report.face_count
        ),
        "overall_quality_delta": round(
            enhanced_report.overall_quality - original_report.overall_quality, 2
        ),
        "zone_variance_delta": round(
            enhanced_report.zone_brightness_variance
            - original_report.zone_brightness_variance, 2
        ),
        "original_overall": round(original_report.overall_quality, 2),
        "enhanced_overall": round(enhanced_report.overall_quality, 2),
    }
