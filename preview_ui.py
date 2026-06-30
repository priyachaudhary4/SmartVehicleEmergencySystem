import sys
import os
import cv2
import numpy as np
import time
from main import render_dashboard_panel, render_left_panel, render_bottom_panel

vehicle_state = {
    "speed": 42,
    "gear": "D",
    "throttle": 35,
    "brake": 0,
    "steer": 2,
    "rpm": 1200,
    "adas_lane": "ACTIVE",
    "adas_obstacle": "ACTIVE",
    "adas_speed": "ACTIVE",
    "adas_traffic": "ACTIVE",
    "adas_eb": "READY",
    "adas_hazard": "OFF",
    "system_status": "SYSTEM STABLE",
    "gps": "11.0168 N, 76.9558 E",
    "driver_status": "NORMAL",
    "emergency_active": False,
    "hospital_alert": "Pending",
    "police_alert": "Pending",
    "guardian_alert": "Pending",
    "v2v_alert": "Pending",
    "det_cars": 8,
    "det_peds": 3,
    "det_moto": 1,
    "det_tl": 2,
    "det_signs": 4,
    "det_bikes": 0,
    "det_em": 0,
    "driver_stats": {
        "ear": 0.28,
        "mar": 0.02,
        "blinks": 3,
        "yawns": 0,
        "head_pose": "Forward",
        "fatigue": 0
    },
    "alert_logs": [
        ("14:25:36", "Driver Unresponsive Detected", "critical"),
        ("14:25:35", "Vehicle Speed Reduced", "warning"),
        ("14:25:34", "Hazard Lights Activated", "info"),
        ("14:25:33", "Traffic Light RED", "critical"),
        ("14:25:32", "Pedestrian Detected", "warning")
    ],
    "sys_alerts": [
        ("Driver Unresponsive", "HIGH", (48, 59, 255)),
        ("Obstacle Ahead", "MEDIUM", (0, 179, 255)),
        ("Traffic Light: RED", "LOW", (118, 230, 0))
    ],
    "comms": [
        ("Emergency Contacts", "SENT", (118, 230, 0)),
        ("Nearby Vehicles (V2V)", "SENT", (118, 230, 0)),
        ("Hospital", "SENT", (118, 230, 0)),
        ("Police", "PENDING", (0, 179, 255))
    ]
}

dash_panel = render_dashboard_panel(vehicle_state)
left_panel = render_left_panel(vehicle_state, width=360, height=900)
bottom_panel = render_bottom_panel(vehicle_state, width=1200, height=300)

driver_frame_disp = np.zeros((300, 400, 3), dtype=np.uint8)
env_frame_disp = np.zeros((600, 1200, 3), dtype=np.uint8)
cv2.putText(env_frame_disp, "CARLA ENVIRONMENT PLACEHOLDER (1200x600)", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
cv2.putText(driver_frame_disp, "WEBCAM", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

right_col = np.vstack((driver_frame_disp, dash_panel))
center_col = np.vstack((env_frame_disp, bottom_panel))
final_ui = np.hstack((left_panel, center_col, right_col))

final_ui_resized = cv2.resize(final_ui, (1600, 735))

output_path = r"C:\Users\Priya_Chaudhary\.gemini\antigravity-ide\brain\720f66f7-a14b-4a5c-b53e-18f3a92892f9\opencv_dashboard_preview.jpg"
cv2.imwrite(output_path, final_ui_resized)
print(f"Saved preview to {output_path}")
