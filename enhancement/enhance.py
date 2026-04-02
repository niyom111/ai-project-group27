"""
Lighting enhancement module — adaptive, zone-aware version.

Public API:
    enhance_image(image_path) -> str | None

This thin wrapper delegates to the full pipeline in pipeline.py
while keeping the same interface that attendance_cycle.py expects.
"""

import os
import config
from enhancement.pipeline import run_pipeline, PipelineResult


def enhance_image(image_path: str) -> str | None:
    """
    Enhance a CCTV classroom image for face detection.

    Runs the full zone-aware enhancement pipeline:
      - Splits the image into a grid of zones
      - Classifies each zone (dark / shadow / normal / bright / overexposed)
      - Applies tailored enhancement per zone category
      - Blends zones back seamlessly
      - Applies global white balance, tone mapping, brightness normalization
      - Detects faces and applies targeted face-region enhancement
      - Measures output quality metrics

    Args:
        image_path: Path to the raw captured image.

    Returns:
        Path to the enhanced image on disk, or None on failure.
    """
    if not image_path or not os.path.isfile(image_path):
        print(f"enhance_image: invalid path '{image_path}'")
        return None

    result: PipelineResult = run_pipeline(
        image_path,
        output_dir=config.ENHANCED_DIR,
        do_resize=False,
    )

    if result.output_path and os.path.isfile(result.output_path):
        _print_summary(result)
        return result.output_path

    print("enhance_image: pipeline produced no output.")
    return None


def _print_summary(result: PipelineResult) -> None:
    """Print a compact summary of what the pipeline did."""
    summary = result.classification_summary
    faces_before = result.face_count_before
    faces_after = result.face_count_after
    elapsed = result.processing_time_ms

    zone_info = ", ".join(f"{k}={v}" for k, v in summary.items() if v > 0)
    print(
        f"[Enhancement] zones: [{zone_info}] | "
        f"faces: {faces_before}->{faces_after} | "
        f"{elapsed:.0f}ms -> {result.output_path}"
    )
