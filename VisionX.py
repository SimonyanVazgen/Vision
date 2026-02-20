import datetime
import cv2
import mediapipe as mp
from ultralytics import YOLO
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate('vision-pluse-firebase-adminsdk-fbsvc-95fee91a53.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://vision-pluse-default-rtdb.firebaseio.com/'  
})

ref = db.reference('restricted_access/secret_document')
ref.set({
    "message": "Hello from VisionPlus!"
})

data = ref.get()
print("[FIREBASE DATA]", data)

mp_hands = mp.solutions.hands
mp_faces = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

model = YOLO("yolov8n.pt")

FLOWER_CLASS_IDS = [58]

flower_names = {
    58: "Potted Plant"
}

SAVE_DIR = "flowers_detected"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.9,
                    min_tracking_confidence=0.7) as hands, \
     mp_faces.FaceDetection(min_detection_confidence=0.7) as faces:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame, verbose=False)

        hand_results = hands.process(rgb)
        face_results = faces.process(rgb)

        action_allowed = False

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                if cls in FLOWER_CLASS_IDS and conf > 0.4:
                    action_allowed = True
                    flower_name = flower_names.get(cls, "Unknown Flower")

                    filename = f"{SAVE_DIR}/{flower_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    flower_crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(filename, flower_crop)
                    print(f"[INFO] Saved: {filename}")
                    print(f"[INFO] Flower detected: {flower_name}")

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                action_allowed = True

        if face_results.detections:
            for detection in face_results.detections:
                mp_draw.draw_detection(frame, detection)
                action_allowed = True

        if action_allowed:
            cv2.putText(frame, "Flower detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No flower detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Hands + Face + Flower Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()