import os
import json
from datetime import datetime
import config
from capture_camera.camera import get_capture, save_frame

# --- HANDOFF IMPORTS ---
# Person 1's work (This matches your screenshot perfectly)
from enhancement import enhance_image 

# Person 2's work (Updated to match your 'face_recognition' folder)
# NOTE: Make sure Person 2 actually names their function 'process_image_for_attendance'
try:
    from face_recognition import process_image_for_attendance 
except ImportError:
    # Fallback mock just in case Person 2 hasn't pushed their code yet
    print("Warning: Could not import Person 2's code. Using mock data.")
    def process_image_for_attendance(path):
        return ["ST001"] # Mock student ID

def run_attendance_cycle(class_id, team):
    # 1. Capture one frame
    cap = get_capture()
    raw_path = save_frame(cap)
    cap.release()

    if not raw_path:
        print("Error: Failed to capture image from camera.")
        return {}

    # 2. Call Person 1's enhancement pipeline
    enhanced_path = enhance_image(raw_path)
    if not enhanced_path:
        print("Warning: Enhancement pipeline failed. Falling back to raw capture.")
        enhanced_path = raw_path

    # 3. Call Person 2's recognition core
    student_ids = process_image_for_attendance(enhanced_path)

    # 4. Load and update attendance.json
    os.makedirs(os.path.dirname(config.ATTENDANCE_OUTPUT), exist_ok=True)
    
    attendance_data = {}
    if os.path.exists(config.ATTENDANCE_OUTPUT) and os.path.getsize(config.ATTENDANCE_OUTPUT) > 0:
        with open(config.ATTENDANCE_OUTPUT, 'r') as f:
            try:
                attendance_data = json.load(f)
            except json.JSONDecodeError:
                attendance_data = {}

    today_str = datetime.now().strftime("%d-%m-%Y")

    # If the file is empty or it's a new day, reset the structure
    if not attendance_data or attendance_data.get("date") != today_str:
        attendance_data = {
            "date": today_str,
            "class_id": class_id,
            "team": team,
            "attendance": []
        }

    # Append new student IDs (avoiding duplicates)
    current_attendance = attendance_data.get("attendance", [])
    for sid in student_ids:
        if sid not in current_attendance:
            current_attendance.append(sid)

    attendance_data["attendance"] = current_attendance
    attendance_data["class_id"] = class_id 
    attendance_data["team"] = team         

    # 5. Save the updated JSON back to disk
    with open(config.ATTENDANCE_OUTPUT, "w") as f:
        json.dump(attendance_data, f, indent=2)

    return attendance_data