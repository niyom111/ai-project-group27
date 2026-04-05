import os
import json
from flask import Flask, jsonify, request, render_template # <-- Added render_template
import config
from capture_camera.attendance_cycle import run_attendance_cycle

app = Flask(__name__)

# Step 6 requirement: Ensure directories exist at startup so nothing crashes
os.makedirs(config.CAPTURE_DIR, exist_ok=True)
os.makedirs(config.ENHANCED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.ATTENDANCE_OUTPUT), exist_ok=True)

# --- ADDED THIS NEW ROUTE ---
@app.route('/')
def home():
    """Serves the main dashboard HTML page."""
    return render_template('index.html')
# ----------------------------

@app.route('/attendance', methods=['GET'])
def get_attendance():
    """Reads and returns the current attendance JSON."""
    if os.path.exists(config.ATTENDANCE_OUTPUT) and os.path.getsize(config.ATTENDANCE_OUTPUT) > 0:
        with open(config.ATTENDANCE_OUTPUT, 'r') as f:
            try:
                data = json.load(f)
                return jsonify(data), 200
            except json.JSONDecodeError:
                return jsonify({"error": "Failed to parse attendance file."}), 500
    else:
        return jsonify({"message": "No attendance data found yet."}), 404

@app.route('/attendance/run', methods=['POST'])
def run_attendance():
    """Triggers one full capture -> enhance -> recognize cycle."""
    # Allow optional JSON body for class_id and team
    req_data = request.get_json(silent=True) or {}
    class_id = req_data.get("class_id", "CSE_A")
    team = req_data.get("team", "Team2")

    # Run the cycle
    updated_data = run_attendance_cycle(class_id, team)

    if not updated_data:
        return jsonify({"error": "Failed to run attendance cycle. Check the camera source."}), 500

    return jsonify(updated_data), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Simple check to ensure the server is running."""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # Run the app locally on port 5000, but turn off the auto-reloader!
    app.run(debug=True, use_reloader=False, port=5000)