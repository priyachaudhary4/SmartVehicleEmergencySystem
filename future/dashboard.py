from flask import Flask, render_template, jsonify, Response
import threading
import logging
import os
import cv2
import time

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

# Global frames for streaming
latest_env_frame = None
latest_driver_frame = None

def generate_frames(frame_type):
    global latest_env_frame, latest_driver_frame
    while True:
        if frame_type == 'env' and latest_env_frame is not None:
            frame = latest_env_frame
        elif frame_type == 'driver' and latest_driver_frame is not None:
            frame = latest_driver_frame
        else:
            time.sleep(0.05)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    return jsonify(vehicle_state)

@app.route('/video_feed/env')
def video_feed_env():
    return Response(generate_frames('env'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/driver')
def video_feed_driver():
    return Response(generate_frames('driver'), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_dashboard():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def launch_dashboard_thread():
    thread = threading.Thread(target=start_dashboard, daemon=True)
    thread.start()
    print("[SYSTEM] Rescue Dashboard started at http://localhost:5000")
    return vehicle_state
