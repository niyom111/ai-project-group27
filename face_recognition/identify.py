import os
import cv2
import pickle
import numpy as np
import config
from deepface import DeepFace
from face_recognition.detect import detect_faces

def process_image_for_attendance(enhanced_image_path):
    """
    Identifies faces in an image and returns a list of matching student IDs.
    Uses MTCNN for face detection and FaceNet embeddings for recognition.
    """
    db_path = os.path.join(config.DATASET_DIR, "face_db.pkl")
    if not os.path.exists(db_path):
        print("Face DB not found. Run enroll.py first.")
        return []
        
    with open(db_path, "rb") as f:
        face_db = pickle.load(f)
        
    if not face_db:
        return []
        
    faces = detect_faces(enhanced_image_path)
    img = cv2.imread(enhanced_image_path)
    if img is None:
        return []
        
    recognized_ids = set()
    
    for f in faces:
        x, y, w, h = f['box']
        # Extract and pad the face
        pad = int(max(w, h) * 0.1)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        try:
            # Get FaceNet embedding for the cropped face
            result = DeepFace.represent(img_path=crop, model_name="Facenet", enforce_detection=False)
            if not result:
                continue
                
            emb = np.array(result[0]["embedding"])
            
            best_match = None
            best_score = -1
            
            # Compare with all students in DB
            for student_id, student_embs in face_db.items():
                for known_emb in student_embs:
                    known_emb = np.array(known_emb)
                    # Cosine similarity: (A dot B) / (||A|| * ||B||)
                    sim = np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
                    
                    if sim > best_score:
                        best_score = sim
                        best_match = student_id
                        
            if best_score >= config.MATCH_THRESHOLD:
                recognized_ids.add(best_match)
                
        except Exception as e:
            print(f"Error during identification of a face: {e}")
            continue
            
    return list(recognized_ids)
