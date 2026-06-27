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

def render_dashboard_panel(vehicle_state):
    # Dark modern background (Height 600, Width 400)
    panel = np.zeros((600, 400, 3), dtype=np.uint8)
    panel[:] = (15, 15, 20)
    
    font_main = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. DRIVER MONITOR METRICS
    cv2.putText(panel, "DRIVER MONITOR METRICS", (20, 30), font_main, 0.6, (255, 210, 0), 1)
    cv2.line(panel, (20, 40), (380, 40), (50, 50, 50), 1)
    
    stats = vehicle_state.get('driver_stats', {})
    ear = stats.get('ear', 0)
    mar = stats.get('mar', 0)
    blinks = stats.get('blinks', 0)
    yawns = stats.get('yawns', 0)
    head = stats.get('head_pose', 'Forward')
    
    cv2.putText(panel, f"EAR         {ear:.2f}", (30, 70), font_main, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"MAR         {mar:.2f}", (30, 100), font_main, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"Blink Count {blinks}", (30, 130), font_main, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"Yawn Count  {yawns}", (30, 160), font_main, 0.55, (200, 200, 200), 1)
    
    cv2.putText(panel, "Head Pose", (240, 70), font_main, 0.5, (150, 150, 150), 1)
    head_color = (0, 255, 0) if head == "Forward" else (0, 150, 255)
    cv2.putText(panel, head, (240, 95), font_main, 0.55, head_color, 1)

    # 2. DRIVER STATUS BANNER
    driver_state = vehicle_state.get("driver_status", "NORMAL")
    if driver_state in ["NORMAL"]:
        box_color = (15, 30, 15)
        text_color = (0, 255, 0)
    elif driver_state in ["UNRESPONSIVE", "CRITICAL ALERT", "DROWSY"]:
        box_color = (0, 0, 80)
        text_color = (0, 0, 255)
    else:
        box_color = (0, 50, 80)
        text_color = (0, 165, 255)
        
    cv2.rectangle(panel, (20, 200), (380, 280), box_color, -1)
    cv2.rectangle(panel, (20, 200), (380, 280), (50, 50, 50), 1)
    cv2.putText(panel, "DRIVER STATUS", (140, 225), font_main, 0.45, (150, 150, 150), 1)
    
    text_size = cv2.getTextSize(driver_state, font_main, 1.0, 2)[0]
    tx = 200 - (text_size[0] // 2)
    cv2.putText(panel, driver_state, (tx, 260), font_main, 1.0, text_color, 2)
    
    # 3. FATIGUE LEVEL GAUGE
    cv2.putText(panel, "FATIGUE LEVEL", (20, 310), font_main, 0.5, (255, 50, 50), 1)
    fatigue = stats.get('fatigue', 0)
    fatigue_pct = min(100, int((fatigue / 100.0) * 100))
    
    center = (70, 360)
    radius = 35
    cv2.circle(panel, center, radius, (40, 40, 40), 4)
    end_angle = int(360 * (fatigue_pct / 100.0))
    if end_angle > 0:
        cv2.ellipse(panel, center, (radius, radius), -90, 0, end_angle, (0, 0, 255), 4)
    
    pct_str = f"{fatigue_pct}%"
    p_size = cv2.getTextSize(pct_str, font_main, 0.5, 1)[0]
    cv2.putText(panel, pct_str, (70 - p_size[0]//2, 365), font_main, 0.5, (200,200,200), 1)
    
    pts = []
    base_y = 360
    for i in range(120, 380, 10):
        y_offset = int(np.sin(i*0.2 + time.time()*10) * 15 * (1 if i%30==0 else 0.2))
        pts.append([i, base_y + y_offset])
    cv2.polylines(panel, [np.array(pts, np.int32)], False, (0, 0, 200), 1)

    # 4. EMERGENCY INFO & GPS
    cv2.putText(panel, "EMERGENCY ACTIONS", (20, 430), font_main, 0.55, (255, 210, 0), 1)
    cv2.line(panel, (20, 440), (380, 440), (50, 50, 50), 1)
    
    alert_y = 470
    for key, label in [("hospital_alert", "Hospital Alert"), ("police_alert", "Police Alert"), ("guardian_alert", "Guardian SMS")]:
        status = vehicle_state.get(key, "Pending")
        color = (0, 255, 0) if status == "Sent" else (100, 100, 100)
        cv2.circle(panel, (30, alert_y-4), 6, color, -1 if status == "Sent" else 1)
        cv2.putText(panel, label, (45, alert_y), font_main, 0.5, (200,200,200), 1)
        alert_y += 25
        
    cv2.putText(panel, "GPS", (220, 470), font_main, 0.45, (150,150,150), 1)
    cv2.putText(panel, vehicle_state.get('gps', 'Loading...'), (220, 495), font_main, 0.45, (255,255,255), 1)
    
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
    
    # Run normal drive initially
    vehicle_controller.normal_drive()
    
    # Initialize Computer Vision AI
    print("[SYSTEM] Initializing AI Models (MediaPipe & YOLOv8)...")
    driver_monitor = DriverMonitor()
    
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
    try:
        while True:
            # Sync with CARLA world
            world.wait_for_tick()

            # --- PROCESS WEBCAM (DRIVER MONITOR) ---
            webcam_success, webcam_frame = webcam.read()
            driver_frame_disp = np.zeros((300, 400, 3), dtype=np.uint8)
            if webcam_success:
                webcam_frame = cv2.flip(webcam_frame, 1)
                driver_frame, driver_state, driver_stats = driver_monitor.process_frame(webcam_frame)
                vehicle_state["driver_status"] = driver_state
                vehicle_state["driver_stats"] = driver_stats
                emergency_manager.evaluate_driver_state(driver_state)
                driver_frame_disp = cv2.resize(driver_frame, (400, 300))
            
            # --- PROCESS CARLA CAMERA (YOLO ENVIRONMENT) ---
            global vehicle_frame
            env_frame_disp = np.zeros((900, 1200, 3), dtype=np.uint8)
            if vehicle_frame is not None:
                if yolo_detector:
                    env_frame, detected = yolo_detector.process_frame(vehicle_frame, vehicle_state.get("emergency_active", False))
                else:
                    env_frame = vehicle_frame
                    
                speed = vehicle_controller.get_speed()
                vehicle_state["speed"] = int(speed)
                h, w, _ = env_frame.shape
                cv2.putText(env_frame, f"Speed: {int(speed)} km/h", (w - 200, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                env_frame_disp = cv2.resize(env_frame, (1200, 900))
                            
            # --- PUSH FRAMES TO WEB DASHBOARD ---
            import future.dashboard
            future.dashboard.latest_driver_frame = driver_frame_disp
            future.dashboard.latest_env_frame = env_frame_disp

            # --- RENDER UNIFIED CENTRAL CONSOLE ---
            dash_panel = render_dashboard_panel(vehicle_state)
            right_col = np.vstack((driver_frame_disp, dash_panel))
            final_ui = np.hstack((env_frame_disp, right_col))
            
            # Show the newly enhanced massive UI
            # Resize it to 1280x720 to ensure it fits perfectly on all laptop screens without clipping
            final_ui_resized = cv2.resize(final_ui, (1280, 720))
            cv2.imshow("GuardianDrive Enhanced Pro Dashboard", final_ui_resized)

            key = cv2.waitKey(1) & 0xFF
            
            if key == 27: # ESC key
                print("[SYSTEM] Shutdown signal received.")
                break
            elif key == ord('q'):
                # Toggle Autopilot
                autopilot_enabled = vehicle.get_control().manual_gear_shift
                vehicle.set_autopilot(not autopilot_enabled)
                print("[SYSTEM] Toggled Autopilot!")
            elif key in [ord('w'), ord('a'), ord('s'), ord('d'), ord('x')]:
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
                if not control.manual_gear_shift: # If autopilot is OFF
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
        webcam.release()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("[SYSTEM] Shutdown complete.")

if __name__ == '__main__':
    main()
