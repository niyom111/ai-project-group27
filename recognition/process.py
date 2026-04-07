import os
from deepface import DeepFace
import config

def process_image_for_attendance(image_path: str) -> list:
    """Finds faces in the image and matches them to our dataset using MTCNN and FaceNet."""
    
    # If there's no dataset folder or it's empty, we can't match anyone
    if not os.path.exists(config.DATASET_DIR) or not os.listdir(config.DATASET_DIR):
        print("Warning: Dataset folder is missing or empty. Nobody is enrolled!")
        return []
        
    student_ids = []
    
    try:
        # DeepFace.find compares the camera image against all images in the dataset folder
        # MTCNN draws the box around the face, FaceNet does the identity matching
        results = DeepFace.find(
            img_path=image_path,
            db_path=config.DATASET_DIR,
            model_name="Facenet",
            detector_backend="mtcnn",
            enforce_detection=False, # Don't crash if nobody is standing in front of the camera
            silent=True # Keeps the terminal clean
        )
        
        # 'results' is a list of Pandas DataFrames (one for each face found in the camera image)
        for df in results:
            if not df.empty:
                # The 'identity' column contains the file path of the matched face
                best_match_path = df.iloc[0]['identity']
                
                # Extract the folder name (which is the student ID, e.g., 'ST001')
                student_id = os.path.basename(os.path.dirname(best_match_path))
                
                if student_id not in student_ids:
                    student_ids.append(student_id)
                    print(f"✅ Matched face: {student_id}")
                    
    except Exception as e:
        print(f"Recognition error: {e}")
        
    return student_ids