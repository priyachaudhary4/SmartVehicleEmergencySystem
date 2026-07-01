import carla
import cv2
import numpy as np
import random
import time
import threading

# Import custom AI modules
from ai.driver_monitor import DriverMonitor
from ai.yolo_detector import YoloDetector
from future.vehicle_controller import VehicleController
from future.emergency_manager import EmergencyManager
from future.notification_system import NotificationSystem
from future.dashboard import launch_dashboard_thread, vehicle_state

def add_alert_log(msg, level="info"):
    now = time.strftime("%H:%M:%S")
    logs = vehicle_state.setdefault("alert_logs", [])
    if not logs or logs[0][1] != msg:
        logs.insert(0, (now, msg, level))
        if len(logs) > 6:
            logs.pop()

# ==========================================
# SMART VEHICLE EMERGENCY RESPONSE SYSTEM
# MAIN ORCHESTRATION MODULE
# ==========================================

# Global variables for CARLA camera stream
vehicle_frame = None

def process_carla_camera(image):
    global vehicle_frame
    img = np.array(image.raw_data)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3] # Remove alpha channel
    vehicle_frame = img.copy()

def draw_rounded_glass_panel(img, pt1, pt2, bg_color, border_color, radius=12, alpha=0.85):
    """Draws a rounded glassmorphism panel on the image."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    overlay = img.copy()
    mask = np.zeros_like(img, dtype=np.uint8)
    
    # 4 corner circles
    cv2.circle(mask, (x1 + radius, y1 + radius), radius, (255, 255, 255), -1)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, (255, 255, 255), -1)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, (255, 255, 255), -1)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, (255, 255, 255), -1)
    
    # Rectangles connecting them
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), (255, 255, 255), -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), (255, 255, 255), -1)
    
    # Create colored panel
    colored_panel = np.zeros_like(img, dtype=np.uint8)
    colored_panel[:] = bg_color
    
    mask_bool = mask > 0
    overlay[mask_bool] = colored_panel[mask_bool]
    
    # Apply alpha blending
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw neon border lines
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border_color, 1)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border_color, 1)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border_color, 1)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border_color, 1)
    
    # Border arcs
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border_color, 1)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border_color, 1)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border_color, 1)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border_color, 1)

def draw_glowing_text(img, text, pos, font, scale, color, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale, (10, 10, 10), thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

def draw_brain_head_icon(img, pos, size, color, thickness=1):
    cx, cy = pos
    cr = int(size * 0.45)
    cv2.ellipse(img, (cx, cy - int(size*0.1)), (cr, cr), 0, 120, 420, color, thickness, cv2.LINE_AA)
    pts = np.array([
        [cx + int(cr * 0.5), cy - int(size*0.1) + int(cr * 0.866)],
        [cx + int(size * 0.4), cy], 
        [cx + int(size * 0.25), cy + int(size * 0.1)], 
        [cx + int(size * 0.35), cy + int(size * 0.15)], 
        [cx + int(size * 0.3), cy + int(size * 0.25)], 
        [cx + int(size * 0.1), cy + int(size * 0.4)], 
        [cx - int(size * 0.25), cy + int(size * 0.4)], 
        [cx - int(cr * 0.5), cy - int(size*0.1) + int(cr * 0.866)]
    ], np.int32)
    cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)
    bx, by, br = cx - int(size * 0.05), cy - int(size * 0.15), int(size * 0.25)
    cv2.circle(img, (bx, by), br, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (bx, by), (br - 3, int(br * 0.5)), 0, 0, 180, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (bx, by), (int(br * 0.4), br - 3), 0, 90, 270, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (bx - int(br * 0.3), by + int(br * 0.8)), int(br * 0.3), color, thickness, cv2.LINE_AA)

def render_dashboard_panel(vehicle_state, driver_frame):
    # Total right column height 900, width 400
    panel = np.zeros((900, 400, 3), dtype=np.uint8)
    bg_color = (5, 5, 5) # #0D1117 (BGR)
    card_bg = (15, 15, 15) # #111827 (BGR)
    panel[:] = bg_color
    
    font_main = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX
    
    c_cyan = (255, 212, 0)
    c_green = (102, 255, 0)
    c_yellow = (0, 215, 255)
    c_orange = (0, 140, 255)
    c_red = (68, 68, 255)
    c_purple = (255, 0, 150)
    c_white = (255, 255, 255)
    c_gray = (150, 150, 150)

    # Helper function to draw cards
    def draw_card(y1, y2, title, bg_col, border_col):
        draw_rounded_glass_panel(panel, (15, y1), (385, y2), bg_col, border_col)
        draw_glowing_text(panel, title, (25, y1 + 25), font_bold, 0.45, c_cyan, 1)

    stats = vehicle_state.get('driver_stats', {})
    ear = stats.get('ear', 0.0)
    mar = stats.get('mar', 0.0)
    blinks = stats.get('blinks', 0)
    yawns = stats.get('yawns', 0)
    head = stats.get('head_pose', 'Forward')
    driver_state_str = vehicle_state.get("driver_status", "NORMAL")

    # ==========================================
    # SECTION 1: DRIVER MONITOR (10 to 330)
    # ==========================================
    y1 = 10
    y2 = 330
    draw_card(y1, y2, "DRIVER MONITOR", (92, 42, 16), (150, 80, 40))
    
    if driver_frame is not None:
        try:
            h, w = driver_frame.shape[:2]
            target_w, target_h = 320, 240
            
            if w/h > target_w/target_h:
                new_w = int(h * target_w / target_h)
                start_x = (w - new_w) // 2
                cropped = driver_frame[:, start_x:start_x+new_w]
            else:
                new_h = int(w * target_h / target_w)
                start_y = (h - new_h) // 2
                cropped = driver_frame[start_y:start_y+new_h, :]
                
            resized_cam = cv2.resize(cropped, (target_w, target_h))
            cam_x = 15 + (370 - target_w) // 2
            panel[y1+40:y1+40+target_h, cam_x:cam_x+target_w] = resized_cam
            cv2.rectangle(panel, (cam_x, y1+40), (cam_x+target_w, y1+40+target_h), (100, 100, 100), 1)
        except Exception:
            pass

    stat_y = y1 + 295
    
    draw_glowing_text(panel, f"EAR: {ear:.2f}", (30, stat_y), font_main, 0.45, c_white, 1)
    ear_col = c_red if ear < 0.22 else c_green
    cv2.circle(panel, (115, stat_y - 4), 4, ear_col, -1)
    
    draw_glowing_text(panel, f"MAR: {mar:.2f}", (30, stat_y + 20), font_main, 0.45, c_white, 1)
    mar_col = c_yellow if mar > 0.6 else c_green
    cv2.circle(panel, (115, stat_y + 16), 4, mar_col, -1)
    
    draw_glowing_text(panel, f"Blinks: {blinks}", (145, stat_y), font_main, 0.45, c_white, 1)
    draw_glowing_text(panel, f"Yawns: {yawns}", (145, stat_y + 20), font_main, 0.45, c_white, 1)
    
    draw_glowing_text(panel, "Pose:", (255, stat_y), font_main, 0.45, c_white, 1)
    hp_col = c_green if head == "Forward" else c_purple
    draw_glowing_text(panel, head, (255, stat_y + 20), font_bold, 0.45, hp_col, 1)

    # ==========================================
    # SECTION 2: DRIVER STATUS & TAKEOVER
    # ==========================================
    y1 = 340
    y2 = 440
    if driver_state_str == "NORMAL": state_col = c_green
    elif driver_state_str == "ATTENTIVE": state_col = c_cyan
    elif driver_state_str == "BLINK": state_col = c_yellow
    elif "DROWSY" in driver_state_str: state_col = c_orange
    elif driver_state_str == "DISTRACTED": state_col = c_purple
    else: state_col = c_red
    
    bg_col_ds = (0, 78, 107) # Amber Gold
    draw_rounded_glass_panel(panel, (15, y1), (385, y2), bg_col_ds, state_col)
    
    # Draw Brain/Head icon on the left
    draw_brain_head_icon(panel, (75, y1 + 35), size=60, color=state_col, thickness=2)
    
    # Shift text to the right to accommodate the icon
    draw_glowing_text(panel, "DRIVER STATUS", (160, y1 + 25), font_bold, 0.45, c_cyan, 1)
    
    ts = cv2.getTextSize(driver_state_str, font_bold, 1.0, 2)[0]
    tx = 255 - (ts[0] // 2) # Center it in the right half
    draw_glowing_text(panel, driver_state_str, (tx, y1 + 55), font_bold, 1.0, state_col, 2)
    
    # Autonomous Takeover Status
    cv2.line(panel, (30, y1 + 65), (370, y1 + 65), state_col, 1)
    ai_status = "Monitoring"
    if vehicle_state.get("emergency_active", False):
        ai_status = "EMERGENCY TAKEOVER ACTIVE"
    elif driver_state_str in ["DROWSY", "DISTRACTED"]:
        ai_status = "Warning Mode"
    elif driver_state_str == "NORMAL":
        ai_status = "Driver Control"
        
    draw_glowing_text(panel, f"AI CONTROL: {ai_status}", (25, y1 + 85), font_main, 0.45, state_col, 1)

    # ==========================================
    # SECTION 3: FATIGUE LEVEL (450 to 580)
    # ==========================================
    y1 = 450
    y2 = 580
    draw_card(y1, y2, "FATIGUE LEVEL", (21, 0, 90), (40, 0, 150))
    
    fatigue = stats.get('fatigue', 0)
    fatigue_pct = min(100, max(0, int(fatigue)))
    
    if fatigue_pct <= 30: f_col = c_green
    elif fatigue_pct <= 60: f_col = c_yellow
    elif fatigue_pct <= 80: f_col = c_orange
    else: f_col = c_red
    
    center = (75, y1 + 80)
    radius = 35
    cv2.circle(panel, center, radius, (40, 40, 40), 4)
    end_angle = int(360 * (fatigue_pct / 100.0))
    if end_angle > 0:
        cv2.ellipse(panel, center, (radius, radius), -90, 0, end_angle, f_col, 4)
    
    pct_str = f"{fatigue_pct}%"
    p_size = cv2.getTextSize(pct_str, font_main, 0.6, 1)[0]
    draw_glowing_text(panel, pct_str, (75 - p_size[0]//2, y1 + 85), font_main, 0.6, c_white, 1)

    graph_x = 130
    graph_y = y1 + 45
    graph_w = 240
    graph_h = 70
    cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (25, 25, 25), -1)
    cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (80, 80, 80), 1)
    
    if not hasattr(render_dashboard_panel, "ecg_pts"):
        render_dashboard_panel.ecg_pts = [0] * 60
    
    render_dashboard_panel.ecg_pts.pop(0)
    noise = int(np.random.normal(0, 2)) if fatigue_pct > 10 else 0
    val = fatigue_pct + noise
    render_dashboard_panel.ecg_pts.append(val)
    
    pts = []
    step_x = graph_w / len(render_dashboard_panel.ecg_pts)
    for i, p_val in enumerate(render_dashboard_panel.ecg_pts):
        px = int(graph_x + i * step_x)
        py = graph_y + graph_h - int(min(100, max(0, p_val)) / 100.0 * graph_h)
        pts.append([px, py])
    cv2.polylines(panel, [np.array(pts, np.int32)], False, f_col, 1)

    # ==========================================
    # SECTION 4: EMERGENCY INFORMATION (590 to 765)
    # ==========================================
    y1 = 590
    y2 = 765
    draw_card(y1, y2, "EMERGENCY INFORMATION", (0, 0, 74), (0, 0, 150))
    
    is_emergency = vehicle_state.get("emergency_active", False)
    
    current_y = y1 + 45
    if is_emergency:
        cv2.circle(panel, (30, current_y - 5), 6, c_red, -1)
        draw_glowing_text(panel, "Driver Unresponsive Detected", (45, current_y), font_bold, 0.45, c_red, 1)
        current_y += 25
        
    draw_glowing_text(panel, "Actions Taken:", (25, current_y), font_main, 0.45, c_cyan, 1)
    
    actions = [
        ("Speed Reduced", is_emergency),
        ("Hazard Lights ON", is_emergency),
        ("Lane Keeping Active", is_emergency),
        ("Vehicle Stopping Safely", is_emergency),
        ("Emergency Alerts Sent", is_emergency)
    ]
    
    ay = current_y + 25
    for txt, active in actions:
        chk_col = c_green if active else (120, 120, 120)
        cv2.circle(panel, (30, ay - 4), 6, chk_col, -1 if active else 1)
        if active:
            draw_glowing_text(panel, "v", (27, ay - 1), font_main, 0.3, (0, 0, 0), 1)
        draw_glowing_text(panel, txt, (45, ay), font_main, 0.45, c_white if active else (210, 210, 210), 1)
        ay += 23

    # ==========================================
    # SECTION 5: DRIVER HEALTH DETAILS (775 to 885)
    # ==========================================
    y1 = 775
    y2 = 885
    draw_card(y1, y2, "DRIVER HEALTH DETAILS", (92, 63, 0), (150, 100, 0))
    
    # Driver Profile
    profile = vehicle_state.get("profile", {
        "name": "Jane Doe",
        "blood": "O+",
        "allergies": "Penicillin, Peanuts",
        "conditions": "Type 1 Diabetes"
    })
    
    draw_glowing_text(panel, f"Driver: {profile['name']}   |   Blood Type: {profile['blood']}", (25, y1 + 45), font_main, 0.48, c_white, 1)
    draw_glowing_text(panel, f"Allergies: {profile['allergies']}", (25, y1 + 70), font_main, 0.48, c_white, 1)
    draw_glowing_text(panel, f"Conditions: {profile['conditions']}", (25, y1 + 95), font_main, 0.48, c_white, 1)
    
    # Vitals removed per user request
    
    return panel

def draw_progress_bar(img, x, y, w, h, progress, max_val, label, val_text, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_glowing_text(img, label, (x, y + 10), font, 0.45, (255, 255, 255), 1)
    draw_glowing_text(img, val_text, (x + w - 40, y + 10), font, 0.45, (255, 255, 255), 1)
    # Background bar
    cv2.rectangle(img, (x + 80, y + 2), (x + w - 50, y + h), (50, 50, 50), -1)
    # Fill bar
    fill_w = int((progress / max_val) * (w - 130))
    if fill_w > 0:
        cv2.rectangle(img, (x + 80, y + 2), (x + 80 + fill_w, y + h), color, -1)

def render_left_panel(state, width=360, height=900):
    # Background: #0D1117 (BGR: 23, 17, 13)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (5, 5, 5)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Card background: #111827 (BGR: 39, 24, 17)
    card_bg = (15, 15, 15)
    border_color = (150, 50, 0) # Subtle neon blue glow (OpenCV BGR)

    # -----------------------------------------
    # SECTION 1: VEHICLE STATUS (0 to 350)
    # -----------------------------------------
    bg_col_1 = (102, 42, 10)
    draw_rounded_glass_panel(panel, (15, 15), (width - 15, 340), bg_col_1, (255, 212, 0))
    draw_glowing_text(panel, "VEHICLE STATUS", (30, 40), font, 0.55, (255, 212, 0), 1)
    
    # Semi-Circular Speedometer (180 degrees)
    center = (width // 2, 170)
    radius = 110
    speed = state.get("speed", 0)
    
    # Colors (BGR)
    col_teal = (168, 229, 0) # #00E5A8
    col_gray = (58, 47, 42)  # #2A2F3A
    col_orange = (0, 165, 255)# #FFA500
    col_red = (48, 59, 255)  # #FF3B30
    
    thickness_bg = 10
    thickness_fg = 16
    
    # Draw Track Segments (0->180 mapped to 180->360 in OpenCV)
    # 0-60: 180 to 234 deg
    cv2.ellipse(panel, center, (radius, radius), 0, 180, 234, col_teal, thickness_bg)
    # 60-140: 234 to 306 deg
    cv2.ellipse(panel, center, (radius, radius), 0, 234, 306, col_gray, thickness_bg)
    # 140-160: 306 to 324 deg
    cv2.ellipse(panel, center, (radius, radius), 0, 306, 324, col_orange, thickness_bg)
    # 160-200: 324 to 360 deg
    cv2.ellipse(panel, center, (radius, radius), 0, 324, 360, col_red, thickness_bg)
    
    # Active Speed Overlay
    end_angle = 180 + min(180, int((speed / 200.0) * 180))
    if speed <= 60:
        active_col = col_teal
    elif speed <= 140:
        active_col = (200, 200, 200) # bright gray/white for active
    elif speed <= 160:
        active_col = col_orange
    else:
        active_col = col_red
        
    if speed > 0:
        cv2.ellipse(panel, center, (radius, radius), 0, 180, end_angle, active_col, thickness_fg)
        # Subtle glow
        cv2.ellipse(panel, center, (radius, radius), 0, 180, end_angle, active_col, thickness_fg + 8, lineType=cv2.LINE_AA)
        # Redraw core to overwrite glow center
        cv2.ellipse(panel, center, (radius, radius), 0, 180, end_angle, (255, 255, 255), 4)

    # Tick marks & Labels
    labels = [0, 20, 40, 60, 100, 140, 160, 200]
    import math
    for val in labels:
        ang_deg = 180 + (val / 200.0) * 180
        ang_rad = math.radians(ang_deg)
        # Ticks
        outer_r = radius + 20
        inner_r = radius + 8
        x_out = int(center[0] + outer_r * math.cos(ang_rad))
        y_out = int(center[1] + outer_r * math.sin(ang_rad))
        x_in = int(center[0] + inner_r * math.cos(ang_rad))
        y_in = int(center[1] + inner_r * math.sin(ang_rad))
        cv2.line(panel, (x_in, y_in), (x_out, y_out), (255, 255, 255), 2)
        
        # Labels
        lbl_r = radius - 25
        x_lbl = int(center[0] + lbl_r * math.cos(ang_rad))
        y_lbl = int(center[1] + lbl_r * math.sin(ang_rad))
        # Center text alignment roughly
        text_size = cv2.getTextSize(str(val), font, 0.45, 1)[0]
        cv2.putText(panel, str(val), (x_lbl - text_size[0]//2, y_lbl + text_size[1]//2), font, 0.45, (255, 255, 255), 1)

    # Center Speed Text
    text_speed = str(int(speed))
    size_speed = cv2.getTextSize(text_speed, font, 1.8, 3)[0]
    cv2.putText(panel, text_speed, (center[0] - size_speed[0]//2, center[1] - 10), font, 1.8, (255, 255, 255), 3)
    
    text_unit = "km/h"
    size_unit = cv2.getTextSize(text_unit, font, 0.5, 1)[0]
    col_subtext = (179, 172, 165) # #A5ACB3
    cv2.putText(panel, text_unit, (center[0] - size_unit[0]//2, center[1] + 15), font, 0.5, col_subtext, 1)

    # Metrics Below Speedometer
    y_offset = 230
    gear = state.get("gear", "D")
    draw_glowing_text(panel, f"Gear        {gear}", (40, y_offset + 10), font, 0.45, (255, 255, 255), 1)
    y_offset += 20
    draw_progress_bar(panel, 40, y_offset, 280, 6, state.get("throttle", 0), 100, "Throttle", f"{state.get('throttle', 0)}%", (0, 255, 0))
    y_offset += 20
    draw_progress_bar(panel, 40, y_offset, 280, 6, state.get("brake", 0), 100, "Brake", f"{state.get('brake', 0)}%", (0, 0, 255))
    y_offset += 20
    # Steering: map -45 to 45 to 0 to 90 for progress, center is 45.
    steer = state.get("steer", 0)
    steer_prog = steer + 45
    draw_progress_bar(panel, 40, y_offset, 280, 6, steer_prog, 90, "Steering", f"{steer} deg", (255, 255, 0))
    y_offset += 20
    draw_progress_bar(panel, 40, y_offset, 280, 6, state.get("rpm", 800), 6000, "RPM", f"{state.get('rpm', 800)}", (0, 200, 255))

    # -----------------------------------------
    # SECTION 2: AI VEHICLE CONTROL (360 to 650)
    # -----------------------------------------
    bg_col_2 = (77, 77, 0)
    draw_rounded_glass_panel(panel, (15, 360), (width - 15, 630), bg_col_2, (255, 212, 0))
    draw_glowing_text(panel, "AI VEHICLE CONTROL", (30, 385), font, 0.55, (255, 212, 0), 1)
    
    is_auto = state.get("autopilot", True)
    mode = "AUTO MODE" if is_auto else "MANUAL MODE"
    mode_col = (102, 255, 0) if is_auto else (0, 215, 255)
    
    draw_glowing_text(panel, mode, (135, 415), font, 0.55, mode_col, 1) # Mode indicator
    
    adas_stat = "ACTIVE" if is_auto else "DISABLED"
    adas_col = (0, 255, 0) if is_auto else (100, 100, 100)
    tl_stat = "ON" if is_auto else "OFF"
    
    features = [
        ("Lane Keeping", state.get("adas_lane", adas_stat), adas_col),
        ("Obstacle Avoidance", state.get("adas_obstacle", adas_stat), adas_col),
        ("Speed Control", state.get("adas_speed", adas_stat), adas_col),
        ("Traffic Light Assist", state.get("adas_traffic", tl_stat), adas_col),
        ("Emergency Braking", state.get("adas_eb", "READY"), (0, 255, 255)),
        ("Hazard Lights", state.get("adas_hazard", "OFF"), (0, 0, 255) if state.get("adas_hazard", "OFF") == "ON" else (100, 100, 100))
    ]
    
    y_feat = 450
    for name, stat, col in features:
        cv2.circle(panel, (40, y_feat - 4), 4, col, -1)
        draw_glowing_text(panel, name, (55, y_feat), font, 0.45, (255, 255, 255), 1)
        draw_glowing_text(panel, stat, (width - 100, y_feat), font, 0.45, col, 1)
        y_feat += 25
        
    # System Status Button
    sys_status = state.get("system_status", "SYSTEM STABLE")
    sys_col = (102, 255, 0) if sys_status == "SYSTEM STABLE" else (68, 68, 255)
    cv2.rectangle(panel, (30, 590), (width - 30, 615), (20, 40, 20), -1)
    cv2.rectangle(panel, (30, 590), (width - 30, 615), sys_col, 1)
    draw_glowing_text(panel, sys_status, (width//2 - 60, 607), font, 0.45, sys_col, 1)

    # -----------------------------------------
    # SECTION 3: ALERT LOGS (650 to 885)
    # -----------------------------------------
    bg_col_3 = (69, 10, 43)
    draw_rounded_glass_panel(panel, (15, 650), (width - 15, 885), bg_col_3, (255, 212, 0))
    draw_glowing_text(panel, "ALERT LOGS", (30, 675), font, 0.55, (255, 212, 0), 1)
    
    logs = state.get("alert_logs", [])
    y_log = 705
    for tm, msg, lvl in logs:
        # Color: Critical=Red, Warning=Yellow, Info=Green
        if lvl == "critical": col = (68, 68, 255)
        elif lvl == "warning": col = (0, 215, 255)
        else: col = (102, 255, 0)
        
        draw_glowing_text(panel, tm, (30, y_log), font, 0.4, (255, 255, 255), 1)
        # Truncate msg if too long
        if len(msg) > 30: msg = msg[:28] + ".."
        draw_glowing_text(panel, msg, (90, y_log), font, 0.4, col, 1)
        y_log += 25

    return panel

def render_bottom_panel(state, width=1200, height=300):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (5, 5, 5) # #0D1117 background
    
    card_bg = (15, 15, 15) # #111827
    border_col = (150, 50, 0) # subtle blue glow
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX
    
    cyan = (255, 212, 0)
    green = (102, 255, 0)
    yellow = (0, 215, 255)
    red = (68, 68, 255)
    purple = (255, 0, 150)
    white = (255, 255, 255)
    
    # --- 1. DETECTED OBJECTS SUMMARY (Top Row) ---
    bg_det = (96, 45, 0) # Navy gradient solid mapping
    draw_rounded_glass_panel(panel, (15, 10), (width-15, 90), bg_det, cyan)
    draw_glowing_text(panel, "DETECTED OBJECTS SUMMARY", (25, 30), font, 0.45, cyan, 1)
    
    cats = [
        ("Cars", state.get("det_cars", 0), cyan, "C"),
        ("Pedestrians", state.get("det_peds", 0), yellow, "P"),
        ("Motorcycles", state.get("det_moto", 0), green, "M"),
        ("Traffic Lights", state.get("det_tl", 0), red, "T"),
        ("Road Signs", state.get("det_signs", 0), purple, "R"),
        ("Bicycles", state.get("det_bikes", 0), yellow, "B"),
        ("Heavy Veh", state.get("det_trucks", 0), red, "H")
    ]
    
    step = (width - 40) // len(cats)
    for i, (name, count, col, icon) in enumerate(cats):
        x = 30 + i * step
        if i > 0: cv2.line(panel, (x-15, 35), (x-15, 80), (60, 40, 30), 1)
        draw_glowing_text(panel, icon, (x, 70), font_bold, 1.2, col, 2)
        draw_glowing_text(panel, name, (x+45, 50), font, 0.45, white, 1)
        draw_glowing_text(panel, str(count), (x+45, 75), font_bold, 0.9, white, 1)

    # --- 2. BOTTOM PANELS (3 Cards) ---
    c_w = (width - 50) // 3
    y_c = 100
    h_c = 150
    
    # CARD A: SYSTEM ALERTS
    x_a = 15
    bg_sys_alert = (37, 66, 0)
    draw_rounded_glass_panel(panel, (x_a, y_c), (x_a+c_w, y_c+h_c), bg_sys_alert, cyan)
    draw_glowing_text(panel, "SYSTEM ALERTS", (x_a+10, y_c+20), font, 0.45, cyan, 1)
    
    alerts = []
    if state.get("emergency_active"):
        alerts.append(("Driver Unresponsive", "HIGH", red))
    elif state.get("driver_status") in ["DROWSY", "DISTRACTED"]:
        alerts.append((f"Driver {state.get('driver_status')}", "MEDIUM", yellow))
    else:
        alerts.append(("Driver Alert", "LOW", green))
        
    if state.get("det_peds", 0) > 0:
        alerts.append(("Pedestrians Ahead", "MEDIUM", yellow))
    if state.get("det_tl", 0) > 0:
        alerts.append(("Traffic Light Ahead", "LOW", green))
        
    alerts = alerts[:3]
    
    step_y = 120 // 3
    for i, (msg, sev, col) in enumerate(alerts):
        ay = y_c + 45 + i * step_y
        cv2.circle(panel, (x_a+25, ay-5), 8, col, 1)
        draw_glowing_text(panel, "!", (x_a+23, ay-1), font_bold, 0.45, col, 1)
        draw_glowing_text(panel, msg, (x_a+45, ay), font, 0.45, white, 1)
        draw_glowing_text(panel, f"Severity: {sev}", (x_a+45, ay+15), font, 0.4, col, 1)

    # CARD B: COMMUNICATION STATUS
    x_b = x_a + c_w + 10
    bg_comm = (102, 0, 61)
    draw_rounded_glass_panel(panel, (x_b, y_c), (x_b+c_w, y_c+h_c), bg_comm, cyan)
    draw_glowing_text(panel, "COMMUNICATION STATUS", (x_b+10, y_c+20), font, 0.45, cyan, 1)
    
    is_em = state.get("emergency_active", False)
    stat = "SENT" if is_em else "READY"
    col_stat = green if is_em else yellow
    
    comms = state.get("comms", [
        ("Contacts", stat, col_stat),
        ("V2V Radio", stat, col_stat),
        ("Hospital", stat, col_stat),
        ("Police", stat, col_stat)
    ])
    
    cx1, cx2 = x_b + 15, x_b + c_w//2 + 5
    cy1, cy2 = y_c + 50, y_c + 105
    coords = [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]
    
    for i, (msg, stat, col) in enumerate(comms):
        cx, cy = coords[i]
        cv2.rectangle(panel, (cx, cy-15), (cx+25, cy+10), (50, 40, 30), 1)
        draw_glowing_text(panel, "@", (cx+4, cy+3), font, 0.6, cyan, 1)
        if "(" in msg:
            p1, p2 = msg.split(" (")
            p2 = "(" + p2
            draw_glowing_text(panel, p1, (cx+35, cy-5), font, 0.45, white, 1)
            draw_glowing_text(panel, p2, (cx+35, cy+10), font, 0.45, white, 1)
            draw_glowing_text(panel, stat, (cx+35, cy+25), font_bold, 0.45, col, 1)
        else:
            draw_glowing_text(panel, msg, (cx+35, cy-2), font, 0.45, white, 1)
            draw_glowing_text(panel, stat, (cx+35, cy+15), font_bold, 0.45, col, 1)

    # CARD C: GPS & ROUTE
    x_c = x_b + c_w + 10
    bg_gps = (102, 51, 0)
    draw_rounded_glass_panel(panel, (x_c, y_c), (x_c+c_w, y_c+h_c), bg_gps, cyan)
    draw_glowing_text(panel, "GPS & ROUTE", (x_c+10, y_c+20), font, 0.45, cyan, 1)
    
    map_x, map_y = x_c + 15, y_c + 35
    map_w, map_h = 110, 100
    cv2.rectangle(panel, (map_x, map_y), (map_x+map_w, map_y+map_h), (25, 35, 25), -1)
    for i in range(1, 4):
        cv2.line(panel, (map_x, map_y + i*25), (map_x+map_w, map_y + i*25), (45, 55, 45), 1)
        cv2.line(panel, (map_x + i*27, map_y), (map_x + i*27, map_y+map_h), (45, 55, 45), 1)
    pts = np.array([[map_x+10, map_y+80], [map_x+40, map_y+50], [map_x+70, map_y+40], [map_x+100, map_y+20]], np.int32)
    cv2.polylines(panel, [pts], False, cyan, 2)
    cv2.circle(panel, (map_x+10, map_y+80), 5, green, -1)
    cv2.circle(panel, (map_x+100, map_y+20), 5, red, -1)
    
    info_x = map_x + map_w + 15
    draw_glowing_text(panel, "Current Location", (info_x, map_y+10), font, 0.35, white, 1)
    draw_glowing_text(panel, "You are here", (info_x, map_y+25), font, 0.4, green, 1)
    
    draw_glowing_text(panel, "Location", (info_x, map_y+45), font, 0.35, white, 1)
    loc_str = f"{state.get('loc_x', 11.01)} N, {state.get('loc_y', 76.95)} E"
    draw_glowing_text(panel, loc_str, (info_x, map_y+60), font, 0.4, white, 1)
    
    draw_glowing_text(panel, f"Dist: {state.get('distance_km', 2.4)} km", (info_x, map_y+80), font, 0.4, white, 1)
    draw_glowing_text(panel, f"ETA: {state.get('eta_min', 4)} min", (info_x, map_y+95), font, 0.4, white, 1)

    # --- 3. FOOTER STATUS BAR ---
    y_f = 270
    cv2.circle(panel, (25, y_f-4), 5, green, -1)
    draw_glowing_text(panel, "CARLA Connected", (38, y_f), font, 0.45, green, 1)
    
    draw_glowing_text(panel, "Server: localhost:2000", (180, y_f), font, 0.45, white, 1)
    draw_glowing_text(panel, "FPS: 28", (350, y_f), font, 0.45, white, 1)
    draw_glowing_text(panel, "Latency: 42 ms", (450, y_f), font, 0.45, white, 1)
    
    cv2.circle(panel, (600, y_f-4), 5, red, -1)
    draw_glowing_text(panel, "Recording", (613, y_f), font, 0.45, white, 1)
    
    draw_glowing_text(panel, "[Logs]", (750, y_f), font, 0.45, white, 1)
    draw_glowing_text(panel, "[Settings]", (850, y_f), font, 0.45, white, 1)

    return panel

def main():
    print("="*50)
    print("AI EMERGENCY CO-PILOT INITIALIZATION")
    print("="*50)
    
    # -----------------------------
    # 1. Initialize Driver Profile & Notifier
    # -----------------------------
    driver_profile = {
        "blood_type": "O+",
        "allergies": ["Penicillin"],
        "medical_conditions": ["None"],
        "emergency_contacts": [
            {"name": "Jane Doe", "phone": "+1-555-0101"}
        ]
    }
    notification_system = NotificationSystem(driver_profile)

    # -----------------------------
    # 2. Connect to CARLA
    # -----------------------------
    print("[SYSTEM] Connecting to CARLA Simulator...")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        print("[SYSTEM] Loading Town05 (This may take a few seconds)...")
        world = client.load_world('Town05')
        print("[SYSTEM] Connected to CARLA successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to connect to CARLA: {e}")
        print("Please ensure the CARLA server is running on localhost:2000.")
        return

    blueprint_library = world.get_blueprint_library()
    
    # -----------------------------
    # 3. Spawn Traffic & Ego Vehicle
    # -----------------------------
    def spawn_traffic(client, world, num_vehicles=40):
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        print(f"[SYSTEM] Spawning {num_vehicles} background traffic vehicles...")
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles: break
            bp = random.choice(blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            bp.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(bp, transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))
        client.apply_batch_sync(batch, True)
    
    spawn_traffic(client, world, num_vehicles=40)
    
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    # Try multiple times to spawn ego vehicle without collision
    vehicle = None
    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle: break
        
    if not vehicle:
        print("[ERROR] Failed to spawn Ego vehicle. Try restarting CARLA.")
        return
    print("[SYSTEM] Tesla Model 3 spawned successfully!")

    # -----------------------------
    # 4. Attach RGB Camera
    # -----------------------------
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', '0.05') # 20 FPS to save resources
    
    # Third-person camera view behind the car
    camera_transform = carla.Transform(carla.Location(x=-6.5, z=3.2), carla.Rotation(pitch=-15))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera.listen(lambda image: process_carla_camera(image))
    print("[SYSTEM] Front RGB Camera attached and streaming!")

    # -----------------------------
    # 5. Initialize Controllers & AI Modules
    # -----------------------------
    vehicle_controller = VehicleController(vehicle, world)
    emergency_manager = EmergencyManager(vehicle_controller, notification_system)
    
    # Initialize Computer Vision AI
    print("[SYSTEM] Initializing AI Models (MediaPipe & YOLOv8)...")
    driver_monitor = DriverMonitor()
    
    # Configure Traffic Manager to keep vehicle driving straight and safely
    tm = client.get_trafficmanager(8000)
    tm.auto_lane_change(vehicle, False)
    tm.vehicle_percentage_speed_difference(vehicle, 0.0) # Drive at 100% of speed limit (approx 30 km/h)
    tm.distance_to_leading_vehicle(vehicle, 8.0) # Maintain 8m distance
    tm.ignore_lights_percentage(vehicle, 0.0) # Strictly follow traffic lights
    tm.ignore_signs_percentage(vehicle, 0.0)  # Strictly follow stop signs
    tm.ignore_vehicles_percentage(vehicle, 0.0) # Never ignore vehicles
    
    # Critical fixes to prevent hazardous swerving into curbs/sidewalks
    try:
        tm.keep_right_rule_percentage(vehicle, 0.0)
    except AttributeError:
        if hasattr(tm, 'keep_slow_lane_rule_percentage'):
            tm.keep_slow_lane_rule_percentage(vehicle, 0.0)
            
    try:
        tm.random_left_lanechange_percentage(vehicle, 0.0)
        tm.random_right_lanechange_percentage(vehicle, 0.0)
    except AttributeError:
        pass
    
    # Run normal drive initially
    vehicle_controller.normal_drive(tm_port=8000)
    
    # Start Webcam for Driver Monitor
    webcam = cv2.VideoCapture(0)
    
    try:
        yolo_detector = YoloDetector(model_name="yolov8n.pt")
    except Exception as e:
        print(f"[WARNING] YOLO initialization failed: {e}. Skipping YOLO.")
        yolo_detector = None

    # Start Web Dashboard
    launch_dashboard_thread()
    
    print("\n" + "="*50)
    print("SYSTEM ACTIVE. PRESS 'ESC' IN CV2 WINDOW TO EXIT.")
    print("="*50 + "\n")

    # -----------------------------
    # 6. Main Simulation Loop
    # -----------------------------
    autopilot_enabled = True
    
    # --- ENABLE SYNCHRONOUS MODE ---
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    
    try:
        while True:
            # Sync with CARLA world
            world.tick()

            # --- PROCESS WEBCAM (DRIVER MONITOR) ---
            webcam_success, webcam_frame = webcam.read()
            driver_frame_disp = np.zeros((300, 400, 3), dtype=np.uint8)
            if webcam_success:
                webcam_frame = cv2.flip(webcam_frame, 1)
                driver_frame, driver_state, driver_stats = driver_monitor.process_frame(webcam_frame)
                vehicle_state["driver_status"] = driver_state
                vehicle_state["driver_stats"] = driver_stats
                if driver_state == "UNRESPONSIVE":
                    add_alert_log("Driver Unresponsive Detected", "critical")
                elif driver_state == "DROWSY":
                    add_alert_log("Driver Drowsiness Detected", "warning")
                
                emergency_manager.evaluate_driver_state(driver_state)
                driver_frame_disp = driver_frame.copy()
            
            # --- PROCESS CARLA CAMERA (YOLO ENVIRONMENT) ---
            global vehicle_frame
            env_frame_disp = np.zeros((900, 1200, 3), dtype=np.uint8)
            if vehicle_frame is not None:
                if yolo_detector:
                    env_frame, detected = yolo_detector.process_frame(vehicle_frame, vehicle_state.get("emergency_active", False))
                    vehicle_state["det_cars"] = detected.get("car", 0)
                    vehicle_state["det_peds"] = detected.get("person", 0)
                    vehicle_state["det_moto"] = detected.get("motorcycle", 0)
                    vehicle_state["det_bikes"] = detected.get("bicycle", 0)
                    vehicle_state["det_trucks"] = detected.get("truck", 0) + detected.get("bus", 0)
                    vehicle_state["det_tl"] = detected.get("traffic_light", 0)
                    vehicle_state["det_signs"] = detected.get("stop_sign", 0)
                else:
                    env_frame = vehicle_frame
                    
                speed = vehicle_controller.get_speed()
                vehicle_state["speed"] = int(speed)
                
                control = vehicle.get_control()
                vehicle_state["throttle"] = int(control.throttle * 100)
                vehicle_state["brake"] = int(control.brake * 100)
                vehicle_state["steer"] = int(control.steer * 45)
                
                if control.reverse:
                    vehicle_state["gear"] = "R"
                elif speed < 0.5 and control.throttle == 0:
                    vehicle_state["gear"] = "P"
                else:
                    vehicle_state["gear"] = "D"
                    
                vehicle_state["autopilot"] = autopilot_enabled
                
                # Add dynamic environmental logs
                if vehicle_state.get("det_peds", 0) > 0:
                    add_alert_log("Pedestrian detected ahead", "warning")
                if vehicle_state.get("det_signs", 0) > 0:
                    add_alert_log("Stop sign detected", "info")
                    
                loc = vehicle.get_transform().location
                vehicle_state["loc_x"] = round(loc.x, 2)
                vehicle_state["loc_y"] = round(loc.y, 2)
                dist_m = ((loc.x)**2 + (loc.y)**2)**0.5
                vehicle_state["distance_km"] = round(dist_m / 1000.0, 1)
                vehicle_state["eta_min"] = max(1, int(dist_m / 500.0))
                vehicle_state["rpm"] = min(6000, max(800, int(abs(speed) * 120 + 800)))
                
                if vehicle_controller.emergency_mode_active:
                    vehicle_state["system_status"] = "EMERGENCY OVERRIDE"
                    vehicle_state["adas_eb"] = "ACTIVE"
                    vehicle_state["adas_hazard"] = "ON"
                else:
                    vehicle_state["system_status"] = "SYSTEM STABLE"
                    vehicle_state["adas_eb"] = "READY"
                    vehicle_state["adas_hazard"] = "OFF"
                
                h, w, _ = env_frame.shape
                cv2.putText(env_frame, f"Speed: {int(speed)} km/h", (w - 200, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                env_frame_disp = cv2.resize(env_frame, (1000, 600))
                            
            # --- PUSH FRAMES TO WEB DASHBOARD ---
            import future.dashboard
            future.dashboard.latest_driver_frame = driver_frame_disp
            future.dashboard.latest_env_frame = env_frame_disp

            # --- RENDER UNIFIED CENTRAL CONSOLE ---
            right_col = render_dashboard_panel(vehicle_state, driver_frame_disp)
            left_panel = render_left_panel(vehicle_state, width=360, height=900)
            bottom_panel = render_bottom_panel(vehicle_state, width=1000, height=300)
            
            center_col = np.vstack((env_frame_disp, bottom_panel))
            
            # Combine 3 columns: Left Panel (360) + Center Col (1000) + Right Col (400) = Total 1760x900
            final_ui = np.hstack((left_panel, center_col, right_col))
            
            # Show the newly enhanced massive UI
            # We use high-quality INTER_AREA interpolation to keep text perfectly sharp!
            final_ui_resized = cv2.resize(final_ui, (1530, 782), interpolation=cv2.INTER_AREA)
            cv2.imshow("GuardianDrive Enhanced Pro Dashboard", final_ui_resized)

            key = cv2.waitKey(1) & 0xFF
            
            if key == 27: # ESC key
                print("[SYSTEM] Shutdown signal received.")
                break
            elif key == ord('q'):
                # Toggle Autopilot
                autopilot_enabled = not autopilot_enabled
                vehicle.set_autopilot(autopilot_enabled)
                print(f"[SYSTEM] Autopilot changed to: {autopilot_enabled}")
            elif key in [ord('w'), ord('a'), ord('s'), ord('d'), ord('x')]:
                autopilot_enabled = False
                vehicle.set_autopilot(False)
                control = vehicle.get_control()
                
                if key == ord('w'):
                    control.throttle = 0.8
                    control.reverse = False
                    control.brake = 0.0
                elif key == ord('s'):
                    control.throttle = 0.6
                    control.reverse = True
                    control.brake = 0.0
                elif key == ord('x'):
                    control.throttle = 0.0
                    control.brake = 1.0
                elif key == ord('a'):
                    control.steer = -0.5
                elif key == ord('d'):
                    control.steer = 0.5
                
                vehicle.apply_control(control)
            elif key == ord('c'):
                # Center steering manually
                control = vehicle.get_control()
                control.steer = 0.0
                vehicle.apply_control(control)
            elif key == 255 and not vehicle_controller.emergency_mode_active:
                # No key pressed: smoothly auto-center steering and decay throttle
                control = vehicle.get_control()
                if not autopilot_enabled: # If autopilot is OFF
                    if control.steer > 0.1: control.steer -= 0.1
                    elif control.steer < -0.1: control.steer += 0.1
                    else: control.steer = 0.0
                    
                    if control.throttle > 0.05: control.throttle -= 0.05
                    else: control.throttle = 0.0
                    
                    vehicle.apply_control(control)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard Interrupt.")
        
    finally:
        print("\n[SYSTEM] Cleaning up resources...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        webcam.release()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("[SYSTEM] Shutdown complete.")

if __name__ == '__main__':
    main()
