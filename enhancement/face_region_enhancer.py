"""
Face Region Enhancer Module
----------------------------
Detects face regions using MTCNN and applies localized enhancement
(contrast, sharpening, denoising, brightness normalization) to each
detected face bounding box with feathered blending.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from enhancement import enhancement_config as ecfg
from enhancement.color_space_converter import (
    bgr_to_rgb,
    rgb_to_bgr,
    split_lab_channels,
    merge_lab_channels,
    lab_to_bgr,
)
from enhancement.histogram_processor import apply_clahe
from enhancement.sharpening_engine import apply_unsharp_mask
from enhancement.noise_reducer import apply_nlm_denoise_color
from enhancement.gamma_corrector import apply_gamma_lut


class FaceDetectionResult:
    """Stores information about a single detected face."""

    def __init__(self):
        self.bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.confidence: float = 0.0
        self.keypoints: Dict[str, Tuple[int, int]] = {}
        self.padded_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.face_brightness: float = 0.0
        self.face_contrast: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 3),
            "padded_bbox": self.padded_bbox,
            "face_brightness": round(self.face_brightness, 2),
            "face_contrast": round(self.face_contrast, 2),
        }


def load_face_detector():
    """
    Load the MTCNN face detector.
    Returns None if MTCNN is not installed.
    """
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        return detector
    except ImportError:
        print("Warning: MTCNN not installed. Face enhancement will be skipped.")
        return None
    except Exception as e:
        print(f"Warning: Failed to load MTCNN: {e}")
        return None


def detect_faces(
    image: np.ndarray,
    detector=None,
    min_confidence: float = ecfg.FACE_DETECTION_CONFIDENCE,
) -> List[FaceDetectionResult]:
    """
    Detect faces in a BGR image using MTCNN.
    Returns a list of FaceDetectionResult objects.
    """
    if detector is None:
        detector = load_face_detector()
    if detector is None:
        return []

    rgb_image = bgr_to_rgb(image, use_cache=False)

    try:
        raw_detections = detector.detect_faces(rgb_image)
    except Exception as e:
        print(f"Warning: Face detection failed: {e}")
        return []

    results: List[FaceDetectionResult] = []
    height, width = image.shape[:2]

    for detection in raw_detections:
        confidence = detection.get("confidence", 0.0)
        if confidence < min_confidence:
            continue

        x, y, w, h = detection["box"]
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        face_result = FaceDetectionResult()
        face_result.bbox = (x, y, w, h)
        face_result.confidence = confidence
        face_result.keypoints = detection.get("keypoints", {})

        pad_x = int(w * ecfg.FACE_BBOX_PADDING_RATIO)
        pad_y = int(h * ecfg.FACE_BBOX_PADDING_RATIO)
        px1 = max(0, x - pad_x)
        py1 = max(0, y - pad_y)
        px2 = min(width, x + w + pad_x)
        py2 = min(height, y + h + pad_y)
        face_result.padded_bbox = (px1, py1, px2 - px1, py2 - py1)

        face_roi = image[y:y+h, x:x+w]
        if face_roi.size > 0:
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_result.face_brightness = float(np.mean(gray_roi))
            face_result.face_contrast = float(np.std(gray_roi))

        results.append(face_result)

    return results


def create_face_feather_mask(
    face_width: int,
    face_height: int,
    feather_radius: int = ecfg.FACE_ENHANCE_FEATHER_RADIUS,
) -> np.ndarray:
    """
    Create a feathered (soft-edged) mask for blending face enhancement
    smoothly into the surrounding area.
    """
    mask = np.ones((face_height, face_width), dtype=np.float64)

    for i in range(feather_radius):
        weight = (i + 1) / feather_radius
        if i < face_height and i < face_width:
            mask[i, :] = min(mask[i, 0], weight)
            mask[face_height - 1 - i, :] = np.minimum(
                mask[face_height - 1 - i, :], weight
            )
            mask[:, i] = np.minimum(mask[:, i], weight)
            mask[:, face_width - 1 - i] = np.minimum(
                mask[:, face_width - 1 - i], weight
            )

    return mask


def enhance_face_clahe(
    face_roi: np.ndarray,
    clip_limit: float = ecfg.FACE_ENHANCE_CLAHE_CLIP,
    grid_size: Tuple[int, int] = ecfg.FACE_ENHANCE_CLAHE_GRID,
) -> np.ndarray:
    """Apply CLAHE specifically to a face region's L channel."""
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_face_sharpen(
    face_roi: np.ndarray,
    sigma: float = ecfg.FACE_ENHANCE_SHARPEN_SIGMA,
    strength: float = ecfg.FACE_ENHANCE_SHARPEN_STRENGTH,
) -> np.ndarray:
    """Apply gentle unsharp masking to a face region."""
    return apply_unsharp_mask(face_roi, sigma, strength, threshold=2)


def enhance_face_denoise(
    face_roi: np.ndarray,
    h: float = ecfg.FACE_ENHANCE_DENOISE_H,
    template_window: int = ecfg.FACE_ENHANCE_DENOISE_TEMPLATE,
    search_window: int = ecfg.FACE_ENHANCE_DENOISE_SEARCH,
) -> np.ndarray:
    """Apply light denoising to a face region."""
    return apply_nlm_denoise_color(face_roi, h, h, template_window, search_window)


def enhance_face_brightness(
    face_roi: np.ndarray,
    target_brightness: float = ecfg.FACE_ENHANCE_BRIGHTNESS_TARGET,
    gamma: float = ecfg.FACE_ENHANCE_GAMMA,
) -> np.ndarray:
    """
    Adjust face brightness toward a target and apply gentle gamma.
    """
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    current_mean = float(np.mean(l_channel))
    if current_mean < target_brightness - 20:
        shift = min(40, (target_brightness - current_mean) * 0.5)
        l_adjusted = np.clip(
            l_channel.astype(np.float64) + shift, 0, 255
        ).astype(np.uint8)
    elif current_mean > target_brightness + 20:
        shift = min(30, (current_mean - target_brightness) * 0.4)
        l_adjusted = np.clip(
            l_channel.astype(np.float64) - shift, 0, 255
        ).astype(np.uint8)
    else:
        l_adjusted = l_channel

    l_gamma = apply_gamma_lut(l_adjusted, gamma)

    merged = cv2.merge([l_gamma, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_single_face(
    face_roi: np.ndarray,
    face_info: FaceDetectionResult,
) -> np.ndarray:
    """
    Apply the full face enhancement pipeline to a single face ROI:
    1. Brightness normalization
    2. CLAHE contrast boost
    3. Denoising
    4. Sharpening
    """
    result = face_roi.copy()

    result = enhance_face_brightness(result)
    result = enhance_face_clahe(result)
    result = enhance_face_denoise(result)
    result = enhance_face_sharpen(result)

    return result


def blend_enhanced_face(
    image: np.ndarray,
    enhanced_roi: np.ndarray,
    x: int, y: int, w: int, h: int,
    blend_strength: float = ecfg.FACE_ENHANCE_BLEND_STRENGTH,
) -> np.ndarray:
    """
    Blend an enhanced face ROI back into the full image using
    feathered masking.
    """
    result = image.copy()
    feather_mask = create_face_feather_mask(w, h)
    feather_mask_3ch = np.stack([feather_mask] * 3, axis=-1)

    effective_mask = feather_mask_3ch * blend_strength

    original_roi = result[y:y+h, x:x+w].astype(np.float64)
    enhanced_float = enhanced_roi.astype(np.float64)

    blended = original_roi * (1.0 - effective_mask) + enhanced_float * effective_mask
    result[y:y+h, x:x+w] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


def enhance_all_faces(
    image: np.ndarray,
    detector=None,
) -> Tuple[np.ndarray, List[FaceDetectionResult]]:
    """
    Detect all faces and apply enhancement to each.
    Returns the enhanced image and the list of detected faces.
    """
    faces = detect_faces(image, detector)

    if not faces:
        return image.copy(), faces

    result = image.copy()

    for face in faces:
        px, py, pw, ph = face.padded_bbox

        if pw < 10 or ph < 10:
            continue

        face_roi = result[py:py+ph, px:px+pw].copy()
        if face_roi.size == 0:
            continue

        enhanced_roi = enhance_single_face(face_roi, face)
        result = blend_enhanced_face(result, enhanced_roi, px, py, pw, ph)

    return result, faces


def count_detectable_faces(
    image: np.ndarray,
    detector=None,
    min_confidence: float = ecfg.FACE_DETECTION_CONFIDENCE,
) -> int:
    """Count the number of detectable faces in an image."""
    faces = detect_faces(image, detector, min_confidence)
    return len(faces)
