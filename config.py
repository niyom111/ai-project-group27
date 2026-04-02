IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAPTURE_INTERVAL_SECONDS = 180

DATASET_DIR = "dataset"
CAPTURE_DIR = "capture/images"
ENHANCED_DIR = "lighting_enhancement/output"

MATCH_THRESHOLD = 0.7  #If similarity ≥ 70%, accept the match

ATTENDANCE_OUTPUT = "attendance/attendance.json"

# --- Enhancement Pipeline Settings ---
ENHANCEMENT_GRID_ROWS = 4
ENHANCEMENT_GRID_COLS = 4
FACE_DETECTION_CONFIDENCE = 0.8
