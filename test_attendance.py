import os
import sys
import json
from capture_camera.attendance_cycle import run_attendance_cycle

# Mocking CAMERA_SOURCE to use a test image
os.environ["CAMERA_SOURCE"] = "test_images/class.jpeg"

def main():
    print("Starting full attendance cycle test...")
    # These are defaults in app.py
    class_id = "CSE_A"
    team = "Team2"
    
    results = run_attendance_cycle(class_id, team)
    
    if results:
        print("\nPipeline execution SUCCESSFUL!")
        print(f"Date: {results.get('date')}")
        print(f"Class: {results.get('class_id')}")
        print(f"Students found: {len(results.get('attendance', []))}")
        print(f"Student IDs: {results.get('attendance')}")
        
        # Save results to a separate file for inspection
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        print("\nPipeline execution FAILED.")

if __name__ == "__main__":
    main()
