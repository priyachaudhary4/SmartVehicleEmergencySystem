import cv2
from ultralytics import YOLO
import numpy as np

# ==========================================
# SMART VEHICLE YOLO ENVIRONMENT DETECTOR
# ==========================================

class YoloDetector:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initializes the YOLOv8 model for environment detection.
        Uses yolov8n.pt (Nano) by default for faster real-time inference.
        """
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        
        # Classes we care about from COCO dataset:
        # 0: person (Pedestrians)
        # 1: bicycle
        # 2: car
        # 3: motorcycle
        # 5: bus
        # 7: truck
        # 9: traffic light
        # 11: stop sign
        self.target_classes = [0, 1, 2, 3, 5, 7, 9, 11]

    def process_frame(self, frame, emergency_active=False):
        """
        Runs YOLO inference on a single frame.
        Returns the annotated frame and a summary of detected objects.
        """
        if frame is None:
            return None, {}

        # Run inference (lower confidence to detect small distant traffic lights in CARLA)
        results = self.model(frame, conf=0.05, imgsz=800, classes=self.target_classes, verbose=False)
        
        # Extract annotated frame
        annotated_frame = results[0].plot(conf=True, line_width=1, font_size=1)

        # Extract counts of detected objects
        detected_summary = {
            "person": 0,
            "bicycle": 0,
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
            "traffic_light": 0,
            "stop_sign": 0
        }

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            
            if cls_id == 0:
                detected_summary["person"] += 1
            elif cls_id == 1:
                detected_summary["bicycle"] += 1
            elif cls_id == 2:
                detected_summary["car"] += 1
            elif cls_id == 3:
                detected_summary["motorcycle"] += 1
            elif cls_id == 5:
                detected_summary["bus"] += 1
            elif cls_id == 7:
                detected_summary["truck"] += 1
            elif cls_id == 9:
                detected_summary["traffic_light"] += 1
            elif cls_id == 11:
                detected_summary["stop_sign"] += 1
                
            if cls_id in [1, 2, 3, 5, 7] and emergency_active:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.putText(annotated_frame, "V2V ALERT RECEIVED", (int(x1), max(0, int(y1)-15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add an overlay for detected counts
        h, w, _ = annotated_frame.shape
        vehicle_count = detected_summary["car"] + detected_summary["truck"] + detected_summary["bus"] + detected_summary["motorcycle"]
        overlay_text = f"Vehicles: {vehicle_count} | Pedestrians: {detected_summary['person']} | Lights: {detected_summary['traffic_light']}"
        cv2.putText(annotated_frame, overlay_text, (20, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return annotated_frame, detected_summary

    def run_standalone(self):
        """
        Runs a standalone webcam test for the YOLO detector.
        """
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
                
            annotated_frame, summary = self.process_frame(frame)
            
            cv2.imshow("YOLO Environment Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = YoloDetector()
    detector.run_standalone()
