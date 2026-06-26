import cv2
import mediapipe as mp
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
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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
        if y < -10:
            pose = "Looking Left"
        elif y > 10:
            pose = "Looking Right"
        elif x < -10:
            pose = "Looking Down"
        elif x > 15:
            pose = "Looking Up"
        else:
            pose = "Forward"
            
        return pose

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        # Default state and color
        color = (0, 255, 0)
        current_time = time.time()
        
        # Calculate FPS
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            
            # --- Draw Landmarks (Eyes) ---
            for idx in self.LEFT_EYE:
                x, y = int(face.landmark[idx].x * w), int(face.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for idx in self.RIGHT_EYE:
                x, y = int(face.landmark[idx].x * w), int(face.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

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
            
            # --- 1. Yawning Logic ---
            if mar > self.MAR_THRESHOLD:
                if not self.yawn_active:
                    self.yawn_active = True
                    self.yawn_counter += 1
                    self.fatigue_score += 10 # Increase fatigue score
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
            if smoothed_ear < self.EAR_THRESHOLD:
                if self.closed_start_time is None:
                    self.closed_start_time = current_time
                
                closed_duration = current_time - self.closed_start_time
                
                if closed_duration < 0.30:
                    self.driver_state = "BLINK"
                    color = (0, 255, 255)
                elif closed_duration < 3:
                    self.driver_state = "EYES CLOSED"
                    color = (0, 165, 255)
                elif closed_duration < 5:
                    self.driver_state = "WARNING"
                    color = (0, 140, 255)
                    self.fatigue_score += 5
                elif closed_duration < 7:
                    self.driver_state = "CRITICAL ALERT"
                    color = (0, 50, 255)
                else:
                    self.driver_state = "UNRESPONSIVE"
                    color = (0, 0, 255)
            else:
                if self.closed_start_time is not None:
                    duration = current_time - self.closed_start_time
                    if duration < 0.30:
                        self.blink_counter += 1
                self.closed_start_time = None
                
                # Check for distraction if eyes are open
                if distracted_duration > 3:
                    self.driver_state = "DISTRACTED"
                    color = (0, 165, 255)
                    self.fatigue_score += 2
                elif self.fatigue_score > 50:
                    self.driver_state = "DROWSY"
                    color = (0, 140, 255)
                else:
                    self.driver_state = "NORMAL"
                    color = (0, 255, 0)
                    # Slowly decrease fatigue score when normal
                    if self.fatigue_score > 0:
                        self.fatigue_score -= 0.1

            # --- HUD Overlay ---
            metrics = [
                (f"FPS: {int(fps)}", (w - 150, 40), (255,255,255)),
                (f"EAR: {smoothed_ear:.2f}", (20, 40), (255,255,255)),
                (f"MAR: {mar:.2f}", (20, 80), (255,255,0)),
                (f"Blinks: {self.blink_counter}", (20, 120), (255,255,255)),
                (f"Yawns: {self.yawn_counter}", (20, 160), (255,255,0)),
                (f"Head: {self.head_pose_status}", (20, 200), (200,200,200)),
                (f"State: {self.driver_state}", (20, 250), color)
            ]
            
            for text, pos, txt_color in metrics:
                cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)
                
            # If unresponsive, draw big alert
            if self.driver_state == "UNRESPONSIVE":
                cv2.putText(frame, "EMERGENCY: DRIVER UNRESPONSIVE", (int(w*0.1), h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        else:
            self.driver_state = "NO FACE DETECTED"
            cv2.putText(frame, self.driver_state, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame, self.driver_state

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
            processed_frame, state = self.process_frame(frame)
            
            cv2.imshow('Driver Monitor', processed_frame)
            if cv2.waitKey(5) & 0xFF == 27: # ESC key
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = DriverMonitor()
    monitor.run_standalone()