import os
import cv2
import pickle
import numpy as np
import config
from deepface import DeepFace
from face_recognition.detect import detect_faces

def build_face_database():
    dataset_dir = config.DATASET_DIR
    os.makedirs(dataset_dir, exist_ok=True)
    embeddings_db = {}
    
    print(f"Scanning dataset in: {dataset_dir}")
    
    for student_id in os.listdir(dataset_dir):
        student_path = os.path.join(dataset_dir, student_id)
        if not os.path.isdir(student_path):
            continue
            
        student_embs = []
        
        for img_name in os.listdir(student_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(student_path, img_name)
            print(f"Processing image for {student_id}: {img_name}")
            
            # Use our custom detect_faces to ensure robustness (like small faces)
            faces = detect_faces(img_path)
            img = cv2.imread(img_path)
            
            # Fallback to DeepFace full image representation if custom detector found nothing
            if not faces:
                print(f"No face detected by custom MTCNN in {img_name}, trying DeepFace directly...")
                try:
                    result = DeepFace.represent(img_path=img_path, model_name="Facenet", detector_backend="mtcnn")
                    for r in result:
                        student_embs.append(r["embedding"])
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
                continue
                
            for f in faces:
                x, y, w, h = f['box']
                # Expand bounding box slightly for better features
                pad = int(max(w, h) * 0.1)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
                
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                try:
                    # Face is already cropped, enforce_detection=False avoids redundant detection errors
                    result = DeepFace.represent(img_path=crop, model_name="Facenet", enforce_detection=False)
                    if result:
                        student_embs.append(result[0]["embedding"])
                except Exception as e:
                    print(f"Error extracting embedding from cropped face in {img_path}: {e}")
                    
        if student_embs:
            embeddings_db[student_id] = student_embs
            print(f"-> Enrolled {student_id} with {len(student_embs)} embeddings.")
        else:
            print(f"-> Failed to find any face embeddings for {student_id}.")
            
    db_path = os.path.join(dataset_dir, "face_db.pkl")
    with open(db_path, "wb") as f:
        pickle.dump(embeddings_db, f)
    print(f"Face database saved to {db_path} with {len(embeddings_db)} students.")

if __name__ == "__main__":
    build_face_database()
