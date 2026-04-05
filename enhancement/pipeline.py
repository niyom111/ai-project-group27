"""
Master Enhancement Pipeline
----------------------------
Orchestrates the full image enhancement flow:
  1. Load & validate image
  2. Analyze zones and classify lighting
  3. Apply conditional per-zone enhancement
  4. Blend zones back together
  5. Global refinements (white balance, tone mapping, brightness norm)
  6. Face-region targeted enhancement
  7. Quality measurement
"""

import os
import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, Tuple, List

from enhancement import enhancement_config as ecfg
from enhancement.image_loader import (
    load_and_prepare_image,
    ImageMetadata,
)
from enhancement.color_space_converter import set_source_image, reset_cache
from enhancement.region_analyzer import (
    analyze_all_zones,
    get_zone_patches,
    compute_global_statistics,
    ZoneCoordinates,
    ZoneStatistics,
)
from enhancement.zone_classifier import (
    classify_all_zones,
    get_classification_summary,
    ZoneClassification,
)
from enhancement.conditional_pipeline import enhance_all_zone_patches
from enhancement.region_blender import blend_enhanced_zones
from enhancement.white_balance_corrector import correct_white_balance
from enhancement.tone_mapper import apply_blended_tone_map
from enhancement.brightness_normalizer import normalize_image_brightness
from enhancement.contrast_enhancer import enhance_contrast_full
from enhancement.face_region_enhancer import enhance_all_faces, load_face_detector
from enhancement.quality_metrics import generate_quality_report, QualityReport


class PipelineResult:
    """Container for the full pipeline execution result."""

    def __init__(self):
        self.enhanced_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.metadata: Optional[ImageMetadata] = None
        self.zone_classifications: List[ZoneClassification] = []
        self.zone_stats: List[ZoneStatistics] = []
        self.classification_summary: Dict[str, int] = {}
        self.global_stats: Dict[str, float] = {}
        self.face_count_before: int = 0
        self.face_count_after: int = 0
        self.quality_report: Optional[QualityReport] = None
        self.processing_time_ms: float = 0.0
        self.output_path: Optional[str] = None
        self.pipeline_log: List[str] = []

    def add_log(self, message: str) -> None:
        self.pipeline_log.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "classification_summary": self.classification_summary,
            "global_stats": self.global_stats,
            "face_count_before": self.face_count_before,
            "face_count_after": self.face_count_after,
            "quality_report": (
                self.quality_report.to_dict() if self.quality_report else None
            ),
            "processing_time_ms": round(self.processing_time_ms, 1),
            "output_path": self.output_path,
            "pipeline_log": self.pipeline_log,
        }


def _save_intermediate(
    image: np.ndarray,
    stage_name: str,
    base_name: str,
) -> None:
    """Save an intermediate result if configured to do so."""
    if not ecfg.PIPELINE_SAVE_INTERMEDIATES:
        return
    os.makedirs(ecfg.PIPELINE_INTERMEDIATE_DIR, exist_ok=True)
    filename = f"{os.path.splitext(base_name)[0]}_{stage_name}.jpg"
    output_path = os.path.join(ecfg.PIPELINE_INTERMEDIATE_DIR, filename)
    cv2.imwrite(output_path, image)


def run_zone_analysis(
    image: np.ndarray,
    result: PipelineResult,
) -> Tuple[List[ZoneCoordinates], List[ZoneStatistics], List[ZoneClassification]]:
    """
    Step 2: Split image into zones, compute stats, classify.
    """
    result.add_log("Analyzing zones...")
    zone_coords, zone_stats = analyze_all_zones(image)
    result.zone_stats = zone_stats
    result.global_stats = compute_global_statistics(zone_stats)

    result.add_log("Classifying zones...")
    classifications = classify_all_zones(zone_stats)
    result.zone_classifications = classifications
    result.classification_summary = get_classification_summary(classifications)

    summary = result.classification_summary
    result.add_log(
        f"Zone classification: dark={summary.get('dark', 0)}, "
        f"shadow={summary.get('shadow', 0)}, "
        f"normal={summary.get('normal', 0)}, "
        f"bright={summary.get('bright', 0)}, "
        f"overexposed={summary.get('overexposed', 0)}"
    )

    return zone_coords, zone_stats, classifications


def run_zone_enhancement(
    image: np.ndarray,
    zone_coords: List[ZoneCoordinates],
    zone_stats: List[ZoneStatistics],
    classifications: List[ZoneClassification],
    result: PipelineResult,
) -> np.ndarray:
    """
    Step 3-4: Enhance each zone with its category pipeline, then blend.
    """
    result.add_log("Extracting zone patches...")
    patches = get_zone_patches(image, zone_coords)

    result.add_log("Enhancing zones conditionally...")
    enhanced_patches = enhance_all_zone_patches(
        patches, classifications, zone_stats
    )

    result.add_log("Blending enhanced zones...")
    blended = blend_enhanced_zones(image, enhanced_patches, zone_coords)

    return blended


def run_global_refinements(
    image: np.ndarray,
    result: PipelineResult,
) -> np.ndarray:
    """
    Step 5: Apply global refinements - white balance, tone mapping,
    contrast, brightness normalization.
    """
    refined = image.copy()

    if ecfg.PIPELINE_ENABLE_WHITE_BALANCE:
        result.add_log("Applying white balance correction...")
        refined = correct_white_balance(refined)
        _save_intermediate(refined, "white_balance", result.metadata.file_name)

    if ecfg.PIPELINE_ENABLE_TONE_MAPPING:
        result.add_log("Applying tone mapping...")
        refined = apply_blended_tone_map(refined)
        _save_intermediate(refined, "tone_mapping", result.metadata.file_name)

    if ecfg.PIPELINE_ENABLE_CONTRAST_ENHANCEMENT:
        result.add_log("Enhancing contrast...")
        refined = enhance_contrast_full(
            refined,
            apply_wallis=False,
            apply_multiscale=True,
            apply_midtone=True,
        )
        _save_intermediate(refined, "contrast", result.metadata.file_name)

    if ecfg.PIPELINE_ENABLE_BRIGHTNESS_NORMALIZATION:
        result.add_log("Normalizing brightness...")
        refined = normalize_image_brightness(refined)
        _save_intermediate(refined, "brightness", result.metadata.file_name)

    return refined


def run_face_enhancement(
    image: np.ndarray,
    result: PipelineResult,
    detector=None,
) -> np.ndarray:
    """
    Step 6: Detect faces and apply targeted enhancement.
    """
    if not ecfg.PIPELINE_ENABLE_FACE_ENHANCEMENT:
        result.add_log("Face enhancement disabled, skipping.")
        return image.copy()

    result.add_log("Detecting and enhancing face regions...")
    enhanced, faces = enhance_all_faces(image, detector)

    result.face_count_after = len(faces)
    result.add_log(f"Enhanced {len(faces)} face region(s).")

    _save_intermediate(enhanced, "faces", result.metadata.file_name)

    return enhanced


def run_quality_assessment(
    image: np.ndarray,
    result: PipelineResult,
) -> None:
    """
    Step 7: Generate quality report for the final enhanced image.
    """
    if not ecfg.PIPELINE_ENABLE_QUALITY_METRICS:
        result.add_log("Quality metrics disabled, skipping.")
        return

    result.add_log("Computing quality metrics...")
    report = generate_quality_report(image, detect_faces_flag=False)
    report.face_count = result.face_count_after
    result.quality_report = report
    result.add_log(
        f"Quality: sharpness={report.sharpness_score:.1f}, "
        f"uniformity={report.brightness_uniformity:.3f}, "
        f"brightness={report.brightness_mean:.1f}, "
        f"contrast={report.contrast_score:.1f}"
    )


def run_pipeline(
    image_path: str,
    output_dir: str = None,
    do_resize: bool = True,
) -> PipelineResult:
    """
    Execute the full image enhancement pipeline.
    """
    start_time = time.time()
    result = PipelineResult()

    import config as app_config
    if output_dir is None:
        output_dir = app_config.ENHANCED_DIR

    # --- Step 1: Load image ---
    result.add_log(f"Loading image: {image_path}")
    image, metadata = load_and_prepare_image(image_path, do_resize=do_resize)

    if image is None or metadata is None:
        result.add_log("ERROR: Failed to load image.")
        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    result.original_image = image.copy()
    result.metadata = metadata
    set_source_image(image)

    _save_intermediate(image, "original", metadata.file_name)
    result.add_log(
        f"Image loaded: {metadata.original_width}x{metadata.original_height}, "
        f"mean brightness={metadata.mean_brightness:.1f}"
    )

    # --- Count faces before enhancement ---
    result.add_log("Counting faces in original image...")
    face_detector = load_face_detector()
    if face_detector is not None:
        try:
            from enhancement.face_region_enhancer import detect_faces
            original_faces = detect_faces(image, face_detector)
            result.face_count_before = len(original_faces)
        except Exception:
            result.face_count_before = 0
    result.add_log(f"Faces detected before enhancement: {result.face_count_before}")

    # --- Step 2: Zone analysis ---
    if ecfg.PIPELINE_ENABLE_ZONE_PROCESSING:
        zone_coords, zone_stats, classifications = run_zone_analysis(image, result)

        # --- Steps 3-4: Per-zone enhancement + blending ---
        enhanced = run_zone_enhancement(
            image, zone_coords, zone_stats, classifications, result
        )
        _save_intermediate(enhanced, "zone_blended", metadata.file_name)
    else:
        result.add_log("Zone processing disabled, using original image.")
        enhanced = image.copy()

    # --- Step 5: Global refinements ---
    enhanced = run_global_refinements(enhanced, result)

    # --- Step 6: Face enhancement ---
    enhanced = run_face_enhancement(enhanced, result, face_detector)

    # --- Step 7: Quality metrics ---
    run_quality_assessment(enhanced, result)

    # --- Save output ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, enhanced)
    result.output_path = output_path
    result.enhanced_image = enhanced

    elapsed = (time.time() - start_time) * 1000
    result.processing_time_ms = elapsed
    result.add_log(f"Pipeline complete in {elapsed:.0f}ms -> {output_path}")

    reset_cache()

    return result
