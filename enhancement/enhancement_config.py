"""
Enhancement Configuration Module
---------------------------------
Central configuration for the entire image enhancement pipeline.
All tunable parameters, thresholds, zone grid sizes, and per-category
processing settings are defined here.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Zone Grid Configuration
# ---------------------------------------------------------------------------
GRID_ROWS = 4
GRID_COLS = 4
TOTAL_ZONES = GRID_ROWS * GRID_COLS

ZONE_OVERLAP_PIXELS = 16
ZONE_FEATHER_RADIUS = 16

# ---------------------------------------------------------------------------
# Image Dimension Constraints
# ---------------------------------------------------------------------------
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MIN_IMAGE_WIDTH = 320
MIN_IMAGE_HEIGHT = 240
MAX_IMAGE_WIDTH = 3840
MAX_IMAGE_HEIGHT = 2160

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# ---------------------------------------------------------------------------
# Zone Classification Thresholds (on L-channel, 0-255 range)
# ---------------------------------------------------------------------------
DARK_UPPER_THRESHOLD = 60
SHADOW_UPPER_THRESHOLD = 100
NORMAL_UPPER_THRESHOLD = 180
BRIGHT_UPPER_THRESHOLD = 220

SHADOW_GRADIENT_VARIANCE_THRESHOLD = 35.0
SHADOW_STD_DEV_MIN = 20.0

ZONE_CONTRAST_LOW = 30.0
ZONE_CONTRAST_HIGH = 80.0

# ---------------------------------------------------------------------------
# Histogram Equalization Settings
# ---------------------------------------------------------------------------
HIST_EQ_BLEND_ORIGINAL_WEIGHT = 0.90
HIST_EQ_BLEND_EQUALIZED_WEIGHT = 0.10

HISTOGRAM_STRETCH_LOW_PERCENTILE = 2
HISTOGRAM_STRETCH_HIGH_PERCENTILE = 98

HISTOGRAM_BINS = 256
HISTOGRAM_RANGE_MIN = 0
HISTOGRAM_RANGE_MAX = 256

# ---------------------------------------------------------------------------
# CLAHE Settings Per Zone Category
# ---------------------------------------------------------------------------
CLAHE_TILE_GRID_SIZE = (8, 8)
CLAHE_TILE_GRID_SIZE_SMALL = (4, 4)
CLAHE_TILE_GRID_SIZE_LARGE = (16, 16)

CLAHE_SETTINGS = {
    "dark": {
        "clip_limit": 2.5,
        "tile_grid_size": CLAHE_TILE_GRID_SIZE,
        "iterations": 2,
        "blend_weight": 0.70,
    },
    "shadow": {
        "clip_limit": 2.5,
        "tile_grid_size": CLAHE_TILE_GRID_SIZE,
        "iterations": 1,
        "blend_weight": 0.80,
    },
    "normal": {
        "clip_limit": 1.5,
        "tile_grid_size": CLAHE_TILE_GRID_SIZE,
        "iterations": 1,
        "blend_weight": 0.70,
    },
    "bright": {
        "clip_limit": 1.0,
        "tile_grid_size": CLAHE_TILE_GRID_SIZE_SMALL,
        "iterations": 1,
        "blend_weight": 0.60,
    },
    "overexposed": {
        "clip_limit": 0.8,
        "tile_grid_size": CLAHE_TILE_GRID_SIZE_SMALL,
        "iterations": 1,
        "blend_weight": 0.50,
    },
}

# ---------------------------------------------------------------------------
# Gamma Correction Settings Per Zone Category
# ---------------------------------------------------------------------------
GAMMA_SETTINGS = {
    "dark": {"gamma": 0.75, "adaptive": True, "min_gamma": 0.6, "max_gamma": 0.85},
    "shadow": {"gamma": 0.85, "adaptive": True, "min_gamma": 0.7, "max_gamma": 0.9},
    "normal": {"gamma": 0.95, "adaptive": False, "min_gamma": 0.9, "max_gamma": 1.0},
    "bright": {"gamma": 1.2, "adaptive": True, "min_gamma": 1.1, "max_gamma": 1.4},
    "overexposed": {"gamma": 1.5, "adaptive": True, "min_gamma": 1.3, "max_gamma": 1.8},
}

GAMMA_LOOKUP_TABLE_SIZE = 256

# ---------------------------------------------------------------------------
# Noise Reduction Settings Per Zone Category
# ---------------------------------------------------------------------------
NOISE_REDUCTION_SETTINGS = {
    "dark": {
        "method": "nlm",
        "h_luminance": 6,
        "h_color": 6,
        "template_window": 7,
        "search_window": 21,
        "bilateral_d": 9,
        "bilateral_sigma_color": 75,
        "bilateral_sigma_space": 75,
        "median_ksize": 5,
        "apply_bilateral_pre": False,
        "apply_median_post": False,
    },
    "shadow": {
        "method": "bilateral",
        "h_luminance": 8,
        "h_color": 8,
        "template_window": 7,
        "search_window": 21,
        "bilateral_d": 7,
        "bilateral_sigma_color": 50,
        "bilateral_sigma_space": 50,
        "median_ksize": 3,
        "apply_bilateral_pre": True,
        "apply_median_post": False,
    },
    "normal": {
        "method": "light",
        "h_luminance": 4,
        "h_color": 4,
        "template_window": 7,
        "search_window": 15,
        "bilateral_d": 5,
        "bilateral_sigma_color": 30,
        "bilateral_sigma_space": 30,
        "median_ksize": 3,
        "apply_bilateral_pre": False,
        "apply_median_post": False,
    },
    "bright": {
        "method": "none",
        "h_luminance": 3,
        "h_color": 3,
        "template_window": 7,
        "search_window": 15,
        "bilateral_d": 5,
        "bilateral_sigma_color": 25,
        "bilateral_sigma_space": 25,
        "median_ksize": 3,
        "apply_bilateral_pre": False,
        "apply_median_post": False,
    },
    "overexposed": {
        "method": "none",
        "h_luminance": 3,
        "h_color": 3,
        "template_window": 7,
        "search_window": 15,
        "bilateral_d": 5,
        "bilateral_sigma_color": 20,
        "bilateral_sigma_space": 20,
        "median_ksize": 3,
        "apply_bilateral_pre": False,
        "apply_median_post": False,
    },
}

# ---------------------------------------------------------------------------
# Sharpening Settings Per Zone Category
# ---------------------------------------------------------------------------
SHARPENING_SETTINGS = {
    "dark": {
        "method": "unsharp",
        "unsharp_sigma": 1.5,
        "unsharp_strength": 0.6,
        "unsharp_threshold": 5,
        "laplacian_weight": 0.15,
        "apply_laplacian": False,
    },
    "shadow": {
        "method": "unsharp",
        "unsharp_sigma": 1.2,
        "unsharp_strength": 0.5,
        "unsharp_threshold": 3,
        "laplacian_weight": 0.10,
        "apply_laplacian": False,
    },
    "normal": {
        "method": "unsharp",
        "unsharp_sigma": 1.0,
        "unsharp_strength": 0.3,
        "unsharp_threshold": 2,
        "laplacian_weight": 0.05,
        "apply_laplacian": False,
    },
    "bright": {
        "method": "none",
        "unsharp_sigma": 0.8,
        "unsharp_strength": 0.2,
        "unsharp_threshold": 2,
        "laplacian_weight": 0.03,
        "apply_laplacian": False,
    },
    "overexposed": {
        "method": "none",
        "unsharp_sigma": 0.5,
        "unsharp_strength": 0.1,
        "unsharp_threshold": 1,
        "laplacian_weight": 0.02,
        "apply_laplacian": False,
    },
}

# ---------------------------------------------------------------------------
# Shadow Detection & Removal
# ---------------------------------------------------------------------------
SHADOW_MORPH_KERNEL_SIZE = 7
SHADOW_MORPH_ITERATIONS = 3
SHADOW_GAUSSIAN_BLUR_KSIZE = 21
SHADOW_THRESHOLD_BLOCK_SIZE = 51
SHADOW_THRESHOLD_C = 10
SHADOW_DILATION_KERNEL_SIZE = 5
SHADOW_DILATION_ITERATIONS = 2

SHADOW_ILLUMINATION_BLUR_KSIZE = 51
SHADOW_COMPENSATION_STRENGTH = 0.5
SHADOW_COMPENSATION_BLEND = 0.5
SHADOW_REMOVAL_GAMMA = 0.9

# ---------------------------------------------------------------------------
# White Balance
# ---------------------------------------------------------------------------
WHITE_BALANCE_METHOD = "gray_world"
WHITE_BALANCE_GRAY_WORLD_POWER = 6
WHITE_BALANCE_MAX_WHITE_PERCENTILE = 95
WHITE_BALANCE_RETINEX_SIGMA_LIST = [15, 80, 250]
WHITE_BALANCE_RETINEX_GAIN = 1.0
WHITE_BALANCE_RETINEX_OFFSET = 0.0
WHITE_BALANCE_BLEND_STRENGTH = 0.7

# ---------------------------------------------------------------------------
# Contrast Enhancement
# ---------------------------------------------------------------------------
CONTRAST_LOCAL_BLOCK_SIZE = 64
CONTRAST_WALLIS_TARGET_MEAN = 127.0
CONTRAST_WALLIS_TARGET_STD = 60.0
CONTRAST_WALLIS_BRIGHTNESS_CONSTANT = 0.8
CONTRAST_WALLIS_CONTRAST_CONSTANT = 0.8
CONTRAST_WALLIS_MAX_STD = 80.0

CONTRAST_MULTISCALE_CLAHE_SCALES = [
    {"clip_limit": 1.0, "tile_grid_size": (4, 4), "weight": 0.3},
    {"clip_limit": 2.0, "tile_grid_size": (8, 8), "weight": 0.4},
    {"clip_limit": 3.0, "tile_grid_size": (16, 16), "weight": 0.3},
]

# ---------------------------------------------------------------------------
# Tone Mapping
# ---------------------------------------------------------------------------
TONE_MAP_METHOD = "reinhard"
TONE_MAP_REINHARD_INTENSITY = 0.0
TONE_MAP_REINHARD_LIGHT_ADAPT = 0.8
TONE_MAP_REINHARD_COLOR_ADAPT = 0.0

TONE_MAP_DRAGO_SATURATION = 1.0
TONE_MAP_DRAGO_BIAS = 0.85
TONE_MAP_DRAGO_GAMMA = 1.0

TONE_MAP_BLEND_STRENGTH = 0.2

# ---------------------------------------------------------------------------
# Brightness Normalization
# ---------------------------------------------------------------------------
BRIGHTNESS_TARGET_MEAN = 130.0
BRIGHTNESS_TARGET_STD = 40.0
BRIGHTNESS_ADJUSTMENT_STRENGTH = 0.4
BRIGHTNESS_MAX_SHIFT = 30
BRIGHTNESS_PRESERVE_HIGHLIGHTS = True
BRIGHTNESS_HIGHLIGHT_THRESHOLD = 240
BRIGHTNESS_PRESERVE_SHADOWS = True
BRIGHTNESS_SHADOW_THRESHOLD = 15

# ---------------------------------------------------------------------------
# Edge-Preserving Filtering
# ---------------------------------------------------------------------------
EDGE_BILATERAL_D = 9
EDGE_BILATERAL_SIGMA_COLOR = 75
EDGE_BILATERAL_SIGMA_SPACE = 75
EDGE_GUIDED_RADIUS = 8
EDGE_GUIDED_EPS = 0.01
EDGE_PRESERVE_BLEND = 0.6

# ---------------------------------------------------------------------------
# Face Region Enhancement
# ---------------------------------------------------------------------------
FACE_DETECTION_CONFIDENCE = 0.80
FACE_DETECTION_MIN_SIZE = (30, 30)
FACE_BBOX_PADDING_RATIO = 0.25
FACE_ENHANCE_CLAHE_CLIP = 2.0
FACE_ENHANCE_CLAHE_GRID = (4, 4)
FACE_ENHANCE_SHARPEN_SIGMA = 1.0
FACE_ENHANCE_SHARPEN_STRENGTH = 0.4
FACE_ENHANCE_DENOISE_H = 6
FACE_ENHANCE_DENOISE_TEMPLATE = 7
FACE_ENHANCE_DENOISE_SEARCH = 15
FACE_ENHANCE_BRIGHTNESS_TARGET = 140.0
FACE_ENHANCE_GAMMA = 0.9
FACE_ENHANCE_FEATHER_RADIUS = 12
FACE_ENHANCE_BLEND_STRENGTH = 0.75

# ---------------------------------------------------------------------------
# Region Blending
# ---------------------------------------------------------------------------
BLEND_FEATHER_SIZE = 16
BLEND_GAUSSIAN_SIGMA = 8.0
BLEND_OVERLAP_PIXELS = 16
BLEND_METHOD = "gaussian"

# ---------------------------------------------------------------------------
# Quality Metrics
# ---------------------------------------------------------------------------
QUALITY_SHARPNESS_THRESHOLD = 100.0
QUALITY_BRIGHTNESS_UNIFORMITY_THRESHOLD = 0.7
QUALITY_MIN_FACE_CONFIDENCE = 0.80
QUALITY_TARGET_BRIGHTNESS_RANGE = (80, 200)

# ---------------------------------------------------------------------------
# Pipeline Control
# ---------------------------------------------------------------------------
PIPELINE_ENABLE_ZONE_PROCESSING = True
PIPELINE_ENABLE_SHADOW_REMOVAL = True
PIPELINE_ENABLE_WHITE_BALANCE = True
PIPELINE_ENABLE_TONE_MAPPING = True
PIPELINE_ENABLE_FACE_ENHANCEMENT = True
PIPELINE_ENABLE_QUALITY_METRICS = True
PIPELINE_ENABLE_NOISE_REDUCTION = True
PIPELINE_ENABLE_SHARPENING = True
PIPELINE_ENABLE_CONTRAST_ENHANCEMENT = True
PIPELINE_ENABLE_BRIGHTNESS_NORMALIZATION = True
PIPELINE_ENABLE_EDGE_PRESERVING = True

PIPELINE_LOG_LEVEL = "INFO"
PIPELINE_SAVE_INTERMEDIATES = False
PIPELINE_INTERMEDIATE_DIR = "lighting_enhancement/intermediates"
