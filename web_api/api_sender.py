import json
import os
import requests
import sys

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# A dummy webhook URL (you can replace this with a real server later)
EXTERNAL_API_URL = "https://jsonplaceholder.typicode.com/posts"

def push_attendance_data():
    """Reads the local attendance.json and pushes it to an external server."""
    if not os.path.exists(config.ATTENDANCE_OUTPUT):
        print("Error: No attendance data found to send.")
        return

    with open(config.ATTENDANCE_OUTPUT, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Could not read attendance.json")
            return

    print(f"Pushing attendance data for class {data.get('class_id')}...")
    
    try:
        # PUSH the data to the external service
        response = requests.post(EXTERNAL_API_URL, json=data)
        
        if response.status_code in [200, 201]:
            print("Success! Data pushed to external service.")
            print("Server Response:", response.json())
        else:
            print(f"Failed to push data. Status Code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    push_attendance_data()