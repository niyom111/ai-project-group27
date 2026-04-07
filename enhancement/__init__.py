# enhancement package — public API
# Person 3 calls: from enhancement import enhance_image

from .enhance import enhance_image

from .pipeline import run_pipeline, PipelineResult
from .image_loader import load_and_prepare_image, ImageMetadata
from .color_space_converter import (
    bgr_to_lab, bgr_to_hsv, bgr_to_gray, bgr_to_rgb,
    lab_to_bgr, hsv_to_bgr,
    split_lab_channels, split_hsv_channels,
)
from .region_analyzer import (
    analyze_all_zones, ZoneCoordinates, ZoneStatistics,
)
from .zone_classifier import (
    classify_all_zones, ZoneClassification,
    ZONE_CATEGORY_DARK, ZONE_CATEGORY_SHADOW, ZONE_CATEGORY_NORMAL,
    ZONE_CATEGORY_BRIGHT, ZONE_CATEGORY_OVEREXPOSED,
)
from .quality_metrics import generate_quality_report, QualityReport
from .face_region_enhancer import detect_faces, enhance_all_faces
