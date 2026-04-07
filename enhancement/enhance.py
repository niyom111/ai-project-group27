"""
Lighting enhancement module — adaptive, artifact-free version.
"""
import cv2
import os
import numpy as np
import config


def enhance_image(image_path: str) -> str | None:
    image = cv2.imread(image_path)
    if image is None:
        return None

    # --- Convert to LAB ---
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # --- VERY LIGHT histogram equalization (almost bypassed) ---
    l_eq = cv2.equalizeHist(l)
    l_soft = cv2.addWeighted(l, 0.95, l_eq, 0.05, 0)  # only 5% effect

    # --- CLAHE (main enhancement) ---
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_soft)

    # --- Merge back ---
    lab_enhanced = cv2.merge([l_clahe, a, b])
    image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # --- Gamma correction (mild) ---
    gamma = 0.95  # safe value
    image = image / 255.0
    image = np.power(image, gamma)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # --- Save output ---
    os.makedirs(config.ENHANCED_DIR, exist_ok=True)
    output_path = os.path.join(config.ENHANCED_DIR, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

    return output_path