from flask import Flask, render_template, jsonify
import threading
import logging
import os

# Create template folder if not exists
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
if not os.path.exists(template_dir):
    os.makedirs(template_dir)

app = Flask(__name__, template_folder=template_dir)

# Global state to share between main thread and Flask
vehicle_state = {
    "speed": 0,
    "gps": "12.927 N, 80.221 E",
    "driver_status": "NORMAL",
    "emergency_active": False,
    "blood_type": "O+",
    "allergies": "Penicillin",
    "hospital_alert": "Pending",
    "guardian_alert": "Pending",
    "police_alert": "Pending",
    "v2v_alert": "Pending"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    return jsonify(vehicle_state)

def start_dashboard():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def launch_dashboard_thread():
    thread = threading.Thread(target=start_dashboard, daemon=True)
    thread.start()
    print("[SYSTEM] Rescue Dashboard started at http://localhost:5000")
    return vehicle_state
