"""
Zone Classifier Module
----------------------
Classifies each image zone into one of five lighting categories:
dark, shadow, normal, bright, or overexposed.  The classification
drives which enhancement strategy is applied to the zone.
"""

import numpy as np
from typing import List, Dict, Tuple

from enhancement import enhancement_config as ecfg
from enhancement.region_analyzer import ZoneStatistics


ZONE_CATEGORY_DARK = "dark"
ZONE_CATEGORY_SHADOW = "shadow"
ZONE_CATEGORY_NORMAL = "normal"
ZONE_CATEGORY_BRIGHT = "bright"
ZONE_CATEGORY_OVEREXPOSED = "overexposed"

VALID_CATEGORIES = [
    ZONE_CATEGORY_DARK,
    ZONE_CATEGORY_SHADOW,
    ZONE_CATEGORY_NORMAL,
    ZONE_CATEGORY_BRIGHT,
    ZONE_CATEGORY_OVEREXPOSED,
]


class ZoneClassification:
    """Holds the classification result for a single zone."""

    def __init__(self, zone_id: int, category: str, confidence: float,
                 reasoning: str):
        self.zone_id = zone_id
        self.category = category
        self.confidence = confidence
        self.reasoning = reasoning

    def to_dict(self) -> Dict:
        return {
            "zone_id": self.zone_id,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
        }


def _is_dark_zone(stats: ZoneStatistics) -> Tuple[bool, float, str]:
    """Check if the zone qualifies as 'dark'."""
    mean_b = stats.mean_brightness

    if mean_b < ecfg.DARK_UPPER_THRESHOLD:
        confidence = 1.0 - (mean_b / ecfg.DARK_UPPER_THRESHOLD)
        confidence = max(0.5, min(1.0, confidence))
        reason = (
            f"Mean brightness {mean_b:.1f} < dark threshold "
            f"{ecfg.DARK_UPPER_THRESHOLD}"
        )
        return True, confidence, reason

    return False, 0.0, ""


def _is_shadow_zone(stats: ZoneStatistics) -> Tuple[bool, float, str]:
    """
    Check if the zone qualifies as 'shadow'.
    Shadow zones have moderate low brightness AND high gradient variance
    (indicating shadow edges).
    """
    mean_b = stats.mean_brightness
    grad_var = stats.gradient_variance
    std_b = stats.std_brightness

    in_brightness_range = (
        ecfg.DARK_UPPER_THRESHOLD <= mean_b < ecfg.SHADOW_UPPER_THRESHOLD
    )
    has_high_gradient = grad_var > ecfg.SHADOW_GRADIENT_VARIANCE_THRESHOLD
    has_high_std = std_b > ecfg.SHADOW_STD_DEV_MIN

    if in_brightness_range and (has_high_gradient or has_high_std):
        grad_score = min(
            1.0, grad_var / (ecfg.SHADOW_GRADIENT_VARIANCE_THRESHOLD * 3)
        )
        std_score = min(1.0, std_b / (ecfg.SHADOW_STD_DEV_MIN * 3))
        confidence = 0.5 * grad_score + 0.5 * std_score
        confidence = max(0.4, min(1.0, confidence))
        reason = (
            f"Brightness {mean_b:.1f} in shadow range, "
            f"gradient var {grad_var:.1f}, std {std_b:.1f}"
        )
        return True, confidence, reason

    if in_brightness_range and stats.histogram_skewness > 0.5:
        confidence = 0.5
        reason = (
            f"Brightness {mean_b:.1f} in shadow range with "
            f"positive skew {stats.histogram_skewness:.2f}"
        )
        return True, confidence, reason

    return False, 0.0, ""


def _is_bright_zone(stats: ZoneStatistics) -> Tuple[bool, float, str]:
    """Check if the zone qualifies as 'bright'."""
    mean_b = stats.mean_brightness

    if ecfg.NORMAL_UPPER_THRESHOLD <= mean_b < ecfg.BRIGHT_UPPER_THRESHOLD:
        ratio = (mean_b - ecfg.NORMAL_UPPER_THRESHOLD) / (
            ecfg.BRIGHT_UPPER_THRESHOLD - ecfg.NORMAL_UPPER_THRESHOLD
        )
        confidence = 0.5 + 0.5 * ratio
        reason = (
            f"Mean brightness {mean_b:.1f} in bright range "
            f"[{ecfg.NORMAL_UPPER_THRESHOLD}, {ecfg.BRIGHT_UPPER_THRESHOLD})"
        )
        return True, confidence, reason

    return False, 0.0, ""


def _is_overexposed_zone(stats: ZoneStatistics) -> Tuple[bool, float, str]:
    """Check if the zone qualifies as 'overexposed'."""
    mean_b = stats.mean_brightness

    if mean_b >= ecfg.BRIGHT_UPPER_THRESHOLD:
        excess = mean_b - ecfg.BRIGHT_UPPER_THRESHOLD
        confidence = min(1.0, 0.7 + 0.3 * (excess / 35.0))
        reason = (
            f"Mean brightness {mean_b:.1f} >= overexposed threshold "
            f"{ecfg.BRIGHT_UPPER_THRESHOLD}"
        )
        return True, confidence, reason

    if stats.percentile_95 > 250 and mean_b > ecfg.NORMAL_UPPER_THRESHOLD:
        confidence = 0.6
        reason = (
            f"95th percentile {stats.percentile_95:.0f} > 250 with "
            f"mean {mean_b:.1f}"
        )
        return True, confidence, reason

    return False, 0.0, ""


def _is_normal_zone(stats: ZoneStatistics) -> Tuple[bool, float, str]:
    """Check if the zone qualifies as 'normal'."""
    mean_b = stats.mean_brightness

    if ecfg.SHADOW_UPPER_THRESHOLD <= mean_b < ecfg.NORMAL_UPPER_THRESHOLD:
        center = (ecfg.SHADOW_UPPER_THRESHOLD + ecfg.NORMAL_UPPER_THRESHOLD) / 2
        distance_from_center = abs(mean_b - center)
        half_range = (ecfg.NORMAL_UPPER_THRESHOLD - ecfg.SHADOW_UPPER_THRESHOLD) / 2
        confidence = 1.0 - (distance_from_center / half_range) * 0.4
        confidence = max(0.5, min(1.0, confidence))
        reason = f"Mean brightness {mean_b:.1f} in normal range"
        return True, confidence, reason

    return False, 0.0, ""


def classify_single_zone(stats: ZoneStatistics) -> ZoneClassification:
    """
    Classify a single zone by checking categories in priority order:
    overexposed > dark > shadow > bright > normal (fallback).
    """
    is_over, conf_over, reason_over = _is_overexposed_zone(stats)
    if is_over:
        return ZoneClassification(stats.zone_id, ZONE_CATEGORY_OVEREXPOSED,
                                  conf_over, reason_over)

    is_dark, conf_dark, reason_dark = _is_dark_zone(stats)
    if is_dark:
        return ZoneClassification(stats.zone_id, ZONE_CATEGORY_DARK,
                                  conf_dark, reason_dark)

    is_shadow, conf_shadow, reason_shadow = _is_shadow_zone(stats)
    if is_shadow:
        return ZoneClassification(stats.zone_id, ZONE_CATEGORY_SHADOW,
                                  conf_shadow, reason_shadow)

    is_bright, conf_bright, reason_bright = _is_bright_zone(stats)
    if is_bright:
        return ZoneClassification(stats.zone_id, ZONE_CATEGORY_BRIGHT,
                                  conf_bright, reason_bright)

    is_normal, conf_normal, reason_normal = _is_normal_zone(stats)
    if is_normal:
        return ZoneClassification(stats.zone_id, ZONE_CATEGORY_NORMAL,
                                  conf_normal, reason_normal)

    return ZoneClassification(
        stats.zone_id, ZONE_CATEGORY_NORMAL, 0.5,
        f"Default fallback, mean brightness {stats.mean_brightness:.1f}"
    )


def classify_all_zones(
    zone_stats: List[ZoneStatistics],
) -> List[ZoneClassification]:
    """Classify every zone in the grid."""
    classifications = []
    for stats in zone_stats:
        classification = classify_single_zone(stats)
        classifications.append(classification)
    return classifications


def get_category_zone_indices(
    classifications: List[ZoneClassification],
) -> Dict[str, List[int]]:
    """Group zone indices by their category."""
    groups: Dict[str, List[int]] = {cat: [] for cat in VALID_CATEGORIES}
    for cls in classifications:
        groups[cls.category].append(cls.zone_id)
    return groups


def get_classification_summary(
    classifications: List[ZoneClassification],
) -> Dict[str, int]:
    """Count how many zones fall into each category."""
    summary: Dict[str, int] = {cat: 0 for cat in VALID_CATEGORIES}
    for cls in classifications:
        summary[cls.category] += 1
    return summary


def compute_classification_confidence_stats(
    classifications: List[ZoneClassification],
) -> Dict[str, float]:
    """Compute average confidence per category."""
    category_confidences: Dict[str, List[float]] = {cat: [] for cat in VALID_CATEGORIES}
    for cls in classifications:
        category_confidences[cls.category].append(cls.confidence)

    avg_conf: Dict[str, float] = {}
    for cat, confs in category_confidences.items():
        if confs:
            avg_conf[f"{cat}_avg_confidence"] = round(float(np.mean(confs)), 3)
        else:
            avg_conf[f"{cat}_avg_confidence"] = 0.0
    return avg_conf
