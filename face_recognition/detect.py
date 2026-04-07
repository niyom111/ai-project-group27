import cv2
import config
from mtcnn import MTCNN

# Initialize MTCNN detector. 
# (Default handles most reasonable sizes)
detector = MTCNN()

def detect_faces(image_path):
    """
    Detect faces using MTCNN and return bounding boxes.
    Filters weak detections based on config.FACE_DETECTION_CONFIDENCE.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    # MTCNN expects RGB images
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    
    # Get the confidence threshold from config, defaulting to 0.8
    confidence_thresh = getattr(config, 'FACE_DETECTION_CONFIDENCE', 0.8)
    
    # Filter and return faces with high confidence
    valid_faces = [f for f in faces if f['confidence'] >= confidence_thresh]
    return valid_faces
