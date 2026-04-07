import cv2
import os
import config
from datetime import datetime

def get_capture():
    # Gets source from environment variable, defaults to "0" (local webcam)
    source = os.environ.get("CAMERA_SOURCE", "0")
    
    # If source is just digits (like "0"), convert it to an int for OpenCV
    if source.isdigit():
        source = int(source)
        
    cap = cv2.VideoCapture(source)
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return frame
    return None

def save_frame(cap):
    # Ensure the capture directory exists
    os.makedirs(config.CAPTURE_DIR, exist_ok=True)
    
    frame = get_frame(cap)
    if frame is not None:
        # Build filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        full_path = os.path.join(config.CAPTURE_DIR, filename)
        
        # Save image
        cv2.imwrite(full_path, frame)
        return full_path
    
    return None