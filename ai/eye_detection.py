import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 133, 159, 145, 160, 144]
RIGHT_EYE = [362, 263, 386, 374, 387, 373]

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

            face = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            # Left Eye
            for idx in LEFT_EYE:
                x = int(face.landmark[idx].x * w)
                y = int(face.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0,255,0), -1)

            # Right Eye
            for idx in RIGHT_EYE:
                x = int(face.landmark[idx].x * w)
                y = int(face.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0,0,255), -1)

            cv2.putText(frame,
                        "Eye Landmarks Detected",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

        cv2.imshow("Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()