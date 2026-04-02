"""
Enhancement Pipeline Test Script
---------------------------------
Tests the full zone-aware enhancement pipeline on images in test_images/.
Produces before/after comparisons, quality metrics, and face detection counts.

Run from project root:
    python scripts/test_enhancement.py
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from enhancement.pipeline import run_pipeline, PipelineResult
from enhancement.image_loader import load_and_prepare_image
from enhancement.quality_metrics import (
    generate_quality_report,
    compare_quality,
    QualityReport,
)
from enhancement.face_region_enhancer import (
    detect_faces,
    load_face_detector,
    count_detectable_faces,
)
from enhancement.region_analyzer import analyze_all_zones, compute_global_statistics
from enhancement.zone_classifier import classify_all_zones, get_classification_summary

TEST_DIR = "test_images"
OUTPUT_DIR = "lighting_enhancement/output"
COMPARISON_DIR = "lighting_enhancement/comparisons"
REPORT_DIR = "lighting_enhancement/reports"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

SEPARATOR = "=" * 70
SUBSEP = "-" * 50


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def get_test_images():
    """Find all image files in the test directory."""
    if not os.path.isdir(TEST_DIR):
        print(f"ERROR: Test directory '{TEST_DIR}/' not found.")
        print("Please create it and add 2-3 classroom test images.")
        return []

    images = [
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith(EXTENSIONS)
    ]

    if not images:
        print(f"No images found in '{TEST_DIR}/'.")
        print(f"Add .jpg/.png images to '{TEST_DIR}/' and re-run.")
        return []

    return sorted(images)


def create_side_by_side(
    original: np.ndarray,
    enhanced: np.ndarray,
    filename: str,
) -> str:
    """Create a side-by-side comparison image with labels."""
    h1, w1 = original.shape[:2]
    h2, w2 = enhanced.shape[:2]

    target_h = max(h1, h2)
    target_w1 = int(w1 * target_h / h1) if h1 > 0 else w1
    target_w2 = int(w2 * target_h / h2) if h2 > 0 else w2

    orig_resized = cv2.resize(original, (target_w1, target_h))
    enh_resized = cv2.resize(enhanced, (target_w2, target_h))

    label_height = 40
    canvas_h = target_h + label_height
    canvas_w = target_w1 + target_w2 + 10
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    thickness = 2

    cv2.putText(canvas, "ORIGINAL", (10, 28), font, font_scale, font_color, thickness)
    cv2.putText(
        canvas, "ENHANCED", (target_w1 + 15, 28),
        font, font_scale, (0, 255, 0), thickness,
    )

    canvas[label_height:label_height + target_h, 0:target_w1] = orig_resized
    canvas[label_height:label_height + target_h, target_w1 + 10:target_w1 + 10 + target_w2] = enh_resized

    output_path = os.path.join(COMPARISON_DIR, f"comparison_{filename}")
    cv2.imwrite(output_path, canvas)
    return output_path


def create_zone_visualization(
    original: np.ndarray,
    classifications,
    zone_coords,
    filename: str,
) -> str:
    """Create a visualization showing zone classifications overlaid on the image."""
    viz = original.copy()

    color_map = {
        "dark": (0, 0, 180),
        "shadow": (0, 140, 180),
        "normal": (0, 180, 0),
        "bright": (0, 200, 255),
        "overexposed": (0, 0, 255),
    }

    for cls, coords in zip(classifications, zone_coords):
        color = color_map.get(cls.category, (128, 128, 128))

        overlay = viz.copy()
        cv2.rectangle(
            overlay,
            (coords.x_start, coords.y_start),
            (coords.x_end, coords.y_end),
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.25, viz, 0.75, 0, viz)

        cv2.rectangle(
            viz,
            (coords.x_start, coords.y_start),
            (coords.x_end, coords.y_end),
            color,
            2,
        )

        label = f"{cls.category[:3].upper()}"
        cx = coords.x_start + 5
        cy = coords.y_start + 20
        cv2.putText(
            viz, label, (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    output_path = os.path.join(COMPARISON_DIR, f"zones_{filename}")
    cv2.imwrite(output_path, viz)
    return output_path


def print_quality_report(report: QualityReport, label: str):
    """Print a quality report in a formatted way."""
    print(f"  {label}:")
    d = report.to_dict()
    for key, val in d.items():
        print(f"    {key:30s}: {val}")


def print_comparison(comparison: dict):
    """Print before/after comparison."""
    print("  Comparison (positive = improvement):")
    for key, val in comparison.items():
        indicator = "+" if isinstance(val, (int, float)) and val > 0 else ""
        print(f"    {key:35s}: {indicator}{val}")


def process_single_image(filename: str, face_detector) -> dict:
    """Run the full pipeline on a single test image and report results."""
    image_path = os.path.join(TEST_DIR, filename)
    print(f"\n{SUBSEP}")
    print(f"Processing: {filename}")
    print(SUBSEP)

    original = cv2.imread(image_path)
    if original is None:
        print(f"  ERROR: Could not load {image_path}")
        return {"filename": filename, "status": "failed", "error": "load_failed"}

    h, w = original.shape[:2]
    print(f"  Dimensions: {w}x{h}")

    print("\n  --- Original Image Analysis ---")
    orig_report = generate_quality_report(original, detect_faces_flag=False)

    orig_faces = 0
    if face_detector is not None:
        try:
            orig_faces = count_detectable_faces(original, face_detector)
        except Exception as e:
            print(f"  Warning: Face detection on original failed: {e}")
    orig_report.face_count = orig_faces

    print_quality_report(orig_report, "Original Quality")

    print(f"\n  --- Zone Analysis ---")
    zone_coords, zone_stats = analyze_all_zones(original)
    classifications = classify_all_zones(zone_stats)
    summary = get_classification_summary(classifications)
    for cat, count in summary.items():
        if count > 0:
            print(f"    {cat:15s}: {count} zone(s)")

    zone_viz_path = create_zone_visualization(
        original, classifications, zone_coords, filename
    )
    print(f"  Zone map saved: {zone_viz_path}")

    print(f"\n  --- Running Enhancement Pipeline ---")
    start_time = time.time()
    pipeline_result: PipelineResult = run_pipeline(
        image_path, output_dir=OUTPUT_DIR, do_resize=False,
    )
    elapsed = (time.time() - start_time) * 1000

    if pipeline_result.enhanced_image is None:
        print(f"  ERROR: Pipeline returned no output.")
        return {"filename": filename, "status": "failed", "error": "pipeline_failed"}

    print(f"  Pipeline completed in {elapsed:.0f}ms")
    for log_line in pipeline_result.pipeline_log:
        print(f"    > {log_line}")

    print(f"\n  --- Enhanced Image Analysis ---")
    enh_report = generate_quality_report(
        pipeline_result.enhanced_image, detect_faces_flag=False,
    )

    enh_faces = 0
    if face_detector is not None:
        try:
            enh_faces = count_detectable_faces(
                pipeline_result.enhanced_image, face_detector,
            )
        except Exception as e:
            print(f"  Warning: Face detection on enhanced failed: {e}")
    enh_report.face_count = enh_faces

    print_quality_report(enh_report, "Enhanced Quality")

    print(f"\n  --- Before vs After ---")
    comparison = compare_quality(orig_report, enh_report)
    print_comparison(comparison)

    print(f"\n  Faces detected: {orig_faces} (original) -> {enh_faces} (enhanced)")

    comp_path = create_side_by_side(
        original, pipeline_result.enhanced_image, filename,
    )
    print(f"  Side-by-side saved: {comp_path}")
    print(f"  Enhanced image saved: {pipeline_result.output_path}")

    return {
        "filename": filename,
        "status": "success",
        "original_quality": orig_report.to_dict(),
        "enhanced_quality": enh_report.to_dict(),
        "comparison": comparison,
        "faces_before": orig_faces,
        "faces_after": enh_faces,
        "processing_time_ms": round(elapsed, 1),
        "output_path": pipeline_result.output_path,
        "comparison_path": comp_path,
        "zone_map_path": zone_viz_path,
        "zone_summary": summary,
    }


def main():
    """Main entry point for the test script."""
    print(SEPARATOR)
    print("  CCTV Classroom Image Enhancement - Test Suite")
    print(SEPARATOR)

    ensure_directories()
    image_files = get_test_images()

    if not image_files:
        return

    print(f"\nFound {len(image_files)} test image(s) in '{TEST_DIR}/'")

    print("\nLoading MTCNN face detector...")
    face_detector = load_face_detector()
    if face_detector is None:
        print("WARNING: MTCNN not available. Face counts will be 0.")
    else:
        print("MTCNN loaded successfully.")

    all_results = []
    total_start = time.time()

    for filename in image_files:
        result = process_single_image(filename, face_detector)
        all_results.append(result)

    total_elapsed = (time.time() - total_start) * 1000

    print(f"\n{SEPARATOR}")
    print("  SUMMARY")
    print(SEPARATOR)

    successful = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] == "failed"]

    print(f"  Total images processed: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total time: {total_elapsed:.0f}ms")

    if successful:
        total_faces_before = sum(r["faces_before"] for r in successful)
        total_faces_after = sum(r["faces_after"] for r in successful)
        avg_time = np.mean([r["processing_time_ms"] for r in successful])

        print(f"\n  Faces detected (total): {total_faces_before} -> {total_faces_after}")
        print(f"  Average processing time: {avg_time:.0f}ms per image")

        avg_quality_before = np.mean(
            [r["original_quality"]["overall_quality"] for r in successful]
        )
        avg_quality_after = np.mean(
            [r["enhanced_quality"]["overall_quality"] for r in successful]
        )
        print(f"  Average quality score: {avg_quality_before:.1f} -> {avg_quality_after:.1f}")

    report_path = os.path.join(REPORT_DIR, "test_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full report saved: {report_path}")

    print(f"\n  Output directories:")
    print(f"    Enhanced images : {OUTPUT_DIR}/")
    print(f"    Comparisons     : {COMPARISON_DIR}/")
    print(f"    Reports         : {REPORT_DIR}/")

    print(f"\n{SEPARATOR}")
    print("  Done!")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
