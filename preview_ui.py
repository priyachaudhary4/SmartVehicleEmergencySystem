import sys
import os
import cv2
import numpy as np
import time
from main import render_dashboard_panel

vehicle_state = {
    "speed": 42,
    "gps": "11.0168 N, 76.9558 E",
    "driver_status": "UNRESPONSIVE",
    "emergency_active": True,
    "hospital_alert": "Pending",
    "police_alert": "Sent",
    "guardian_alert": "Sent",
    "v2v_alert": "Pending",
    "driver_stats": {
        "ear": 0.18,
        "mar": 0.62,
        "blinks": 15,
        "yawns": 3,
        "head_pose": "Forward",
        "fatigue": 92
    }
}

panel = render_dashboard_panel(vehicle_state)
output_path = r"C:\Users\Priya_Chaudhary\.gemini\antigravity-ide\brain\9780b992-f944-4f3d-a84d-b9bcc4e919e4\opencv_dashboard_preview.jpg"
cv2.imwrite(output_path, panel)
print(f"Saved preview to {output_path}")
