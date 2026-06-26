import cv2
import mediapipe as mp
from scipy.spatial import distance

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Open Webcam
cap = cv2.VideoCapture(0)

# Left Eye Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right Eye Landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(eye_points, landmarks, width, height):
    points = []

    for idx in eye_points:
        x = int(landmarks[idx].x * width)
        y = int(landmarks[idx].y * height)
        points.append((x, y))

    # Vertical distances
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])

    # Horizontal distance
    C = distance.euclidean(points[0], points[3])

    ear = (A + B) / (2.0 * C)

    return ear


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            # Left Eye EAR
            left_ear = eye_aspect_ratio(
                LEFT_EYE,
                face_landmarks.landmark,
                w,
                h
            )

            # Right Eye EAR
            right_ear = eye_aspect_ratio(
                RIGHT_EYE,
                face_landmarks.landmark,
                w,
                h
            )

            # Average EAR
            ear = (left_ear + right_ear) / 2
            print(f"EAR = {ear:.3f}")

            # Draw eye landmark points
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            # Eye Status
            if ear > 0.22:
                status = "Eyes OPEN"
                color = (0, 255, 0)
            else:
                status = "Eyes CLOSED"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                f"EAR : {ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            cv2.putText(
                frame,
                status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        cv2.imshow("EAR Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()