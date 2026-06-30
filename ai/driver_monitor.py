import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None
import time
import math
import numpy as np
from scipy.spatial import distance

# ==========================================
# SMART VEHICLE DRIVER MONITOR (Advanced)
# ==========================================

class DriverMonitor:
    def __init__(self):
        # -----------------------------
        # MediaPipe Setup
        # -----------------------------
        if mp is not None:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            print("[WARNING] MediaPipe not installed. Driver Monitor disabled.")
            self.mp_face_mesh = None
            self.face_mesh = None

        # -----------------------------
        # Landmarks Configuration
        # -----------------------------
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.LEFT_MOUTH = 78
        self.RIGHT_MOUTH = 308

        # For head pose estimation (Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth)
        self.FACE_3D = []
        self.FACE_2D = []

        # -----------------------------
        # Thresholds & State Variables
        # -----------------------------
        self.EAR_THRESHOLD = 0.20
        self.MAR_THRESHOLD = 0.60
        
        self.ear_history = []
        
        self.driver_state = "NORMAL"
        self.fatigue_score = 0
        
        # Blink tracking
        self.blink_counter = 0
        self.closed_start_time = None
        
        # Yawning tracking
        self.yawn_counter = 0
        self.yawn_active = False
        
        # Head pose tracking
        self.head_pose_status = "Forward"
        self.distracted_start_time = None

        # FPS calculation
        self.prev_time = 0

    def eye_aspect_ratio(self, eye_points, landmarks, width, height):
        pts = [(int(landmarks[idx].x * width), int(landmarks[idx].y * height)) for idx in eye_points]
        
        # Vertical distances
        A = distance.euclidean(pts[1], pts[5])
        B = distance.euclidean(pts[2], pts[4])
        
        # Horizontal distance
        C = distance.euclidean(pts[0], pts[3])
        
        if C == 0: return 0
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, landmarks, width, height):
        top_lip = (int(landmarks[self.UPPER_LIP].x * width), int(landmarks[self.UPPER_LIP].y * height))
        bottom_lip = (int(landmarks[self.LOWER_LIP].x * width), int(landmarks[self.LOWER_LIP].y * height))
        left_lip = (int(landmarks[self.LEFT_MOUTH].x * width), int(landmarks[self.LEFT_MOUTH].y * height))
        right_lip = (int(landmarks[self.RIGHT_MOUTH].x * width), int(landmarks[self.RIGHT_MOUTH].y * height))
        
        vertical_dist = distance.euclidean(top_lip, bottom_lip)
        horizontal_dist = distance.euclidean(left_lip, right_lip)
        
        if horizontal_dist == 0: return 0
        return vertical_dist / horizontal_dist

    def estimate_head_pose(self, face_landmarks, frame_w, frame_h):
        face_3d = []
        face_2d = []
        
        # Key landmarks for head pose: Nose(1), Chin(152), Left Eye Left(33), Right Eye Right(263), Left Mouth(61), Right Mouth(291)
        key_landmarks = [1, 152, 33, 263, 61, 291]
        
        for idx in key_landmarks:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * frame_w), int(lm.y * frame_h)
            
            # Get 2D Coordinates
            face_2d.append([x, y])
            # Get 3D Coordinates (z is approximated)
            face_3d.append([x, y, lm.z])
            
        # Convert to numpy arrays
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera matrix
        focal_length = 1 * frame_w
        cam_matrix = np.array([[focal_length, 0, frame_h / 2],
                               [0, focal_length, frame_w / 2],
                               [0, 0, 1]])

        # Distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        # Get rotational matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        x = angles[0] * 360
        y = angles[1] * 360
        
        # Determine Head Pose
        if y < -15:
            pose = "Looking Left"
        elif y > 15:
            pose = "Looking Right"
        elif x < -15:
            pose = "Looking Down"
        elif x > 25:
            pose = "Looking Up"
        else:
            pose = "Forward"
            
        return pose

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.face_mesh is None:
            cv2.putText(frame, "DRIVER MONITOR DISABLED (Missing MediaPipe)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            stats = {"ear": 0, "mar": 0, "blinks": 0, "yawns": 0, "head_pose": "Unknown", "fatigue": 0}
            return frame, "DISABLED", stats

        results = self.face_mesh.process(rgb)

        # Default state and color
        color = (0, 255, 0)
        current_time = time.time()
        
        # Calculate FPS
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            
            # --- Draw Face Bounding Box ---
            x_min = w; y_min = h; x_max = 0; y_max = 0
            for lm in face.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y
            # Glowing cyan bounding box (#00B8FF in BGR)
            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 184, 0), 1)
            
            # --- Draw Eye Landmarks (Green Dots) ---
            for idx in self.LEFT_EYE + self.RIGHT_EYE:
                x, y = int(face.landmark[idx].x * w), int(face.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # --- Draw Mouth Landmarks (Yellow Dots) ---
            mouth_indices = [self.UPPER_LIP, self.LOWER_LIP, self.LEFT_MOUTH, self.RIGHT_MOUTH]
            for idx in mouth_indices:
                x, y = int(face.landmark[idx].x * w), int(face.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            # --- Calculate EAR, MAR, Head Pose ---
            left_ear = self.eye_aspect_ratio(self.LEFT_EYE, face.landmark, w, h)
            right_ear = self.eye_aspect_ratio(self.RIGHT_EYE, face.landmark, w, h)
            ear = (left_ear + right_ear) / 2.0
            
            self.ear_history.append(ear)
            if len(self.ear_history) > 5:
                self.ear_history.pop(0)
            smoothed_ear = sum(self.ear_history) / len(self.ear_history)
            
            mar = self.mouth_aspect_ratio(face.landmark, w, h)
            self.head_pose_status = self.estimate_head_pose(face, w, h)
            
            # --- Draw Head Pose Indicator ---
            cx = x_min + (x_max - x_min) // 2
            arrow_color = (0, 255, 0) if self.head_pose_status == "Forward" else (0, 165, 255)
            cv2.putText(frame, f"POSE: {self.head_pose_status.upper()}", (cx - 40, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color, 1)

            # --- 1. Yawning Logic ---
            if mar > 0.6: # Configured MAR threshold
                if not self.yawn_active:
                    self.yawn_active = True
                    self.yawn_counter += 1
            else:
                self.yawn_active = False

            # --- 2. Distraction / Head Pose Logic ---
            if self.head_pose_status != "Forward":
                if self.distracted_start_time is None:
                    self.distracted_start_time = current_time
                distracted_duration = current_time - self.distracted_start_time
            else:
                self.distracted_start_time = None
                distracted_duration = 0

            # --- 3. Eye Closure / Blink Logic ---
            # Configured EAR threshold
            if smoothed_ear < 0.22:
                if self.closed_start_time is None:
                    self.closed_start_time = current_time
                closed_duration = current_time - self.closed_start_time
            else:
                if self.closed_start_time is not None:
                    duration = current_time - self.closed_start_time
                    if duration < 0.30:
                        self.blink_counter += 1
                self.closed_start_time = None
                closed_duration = 0

            # --- 4. Driver State Logic ---
            if closed_duration >= 4.0:
                self.driver_state = "UNRESPONSIVE"
            elif distracted_duration >= 5.0:
                self.driver_state = "FAINTED"
            elif closed_duration >= 2.0:
                self.driver_state = "DROWSY"
            elif distracted_duration > 3.0:
                self.driver_state = "DISTRACTED"
            elif closed_duration >= 0.30:
                self.driver_state = "BLINK"
            else:
                if self.head_pose_status == "Forward":
                    self.driver_state = "ATTENTIVE"
                else:
                    self.driver_state = "NORMAL"

            # --- 5. Fatigue Score Calculation ---
            # 30% Eye Closure (closed_duration: 2 sec = 30%)
            f_eye = min(30, (closed_duration / 2.0) * 30)
            # 30% Yawning (yawn_counter: 5 yawns = 30%)
            f_yawn = min(30, (self.yawn_counter / 5.0) * 30)
            # 20% Blink Frequency (blink_counter: 30 blinks = 20%)
            f_blink = min(20, (self.blink_counter / 30.0) * 20)
            # 20% Head Movement (distracted_duration: 3 sec = 20%)
            f_head = min(20, (distracted_duration / 3.0) * 20)
            
            raw_fatigue = f_eye + f_yawn + f_blink + f_head
            
            # State-based Overrides
            if self.driver_state in ["UNRESPONSIVE", "FAINTED"]:
                self.fatigue_score = 100.0
            elif self.driver_state == "DROWSY":
                self.fatigue_score = max(80.0, raw_fatigue)
            else:
                self.fatigue_score = raw_fatigue
            
        else:
            self.driver_state = "NO FACE DETECTED"
            cv2.putText(frame, self.driver_state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            smoothed_ear = 0
            mar = 0

        stats = {
            "ear": smoothed_ear if 'smoothed_ear' in locals() else 0,
            "mar": mar if 'mar' in locals() else 0,
            "blinks": self.blink_counter,
            "yawns": self.yawn_counter,
            "head_pose": self.head_pose_status,
            "fatigue": self.fatigue_score
        }
        return frame, self.driver_state, stats

    # This method allows standalone testing with a webcam
    def run_standalone(self):
        cap = cv2.VideoCapture(0)
        self.prev_time = time.time()
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            processed_frame, state, stats = self.process_frame(frame)
            
            cv2.imshow('Driver Monitor', processed_frame)
            if cv2.waitKey(5) & 0xFF == 27: # ESC key
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = DriverMonitor()
    monitor.run_standalone()