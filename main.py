import cv2
import time
import requests
from collections import Counter

WEBHOOK_URL = "http://127.0.0.1:8502/emotion"


def send_emotion(emotion):
    payload = {"emotion": emotion}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
        print(f"Sent: {payload} | Response: {response.status_code}")
    except Exception as e:
        print("Error sending data:", e)


def run_yolo():
    from ultralytics import YOLO

    model = YOLO(r"C:\Users\chitt\Downloads\Facial Emotion Detection\Facial Emotion Detection\last.pt")
    cap = cv2.VideoCapture(0)
    interval = 5
    last_sent = time.time()
    detection_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, show=True, conf=0.5, verbose=False)

        for result in results:
            if result.names:
                for c in result.boxes.cls.tolist():
                    emotion = result.names[int(c)]
                    detection_buffer.append(emotion)

        current_time = time.time()
        if current_time - last_sent >= interval:
            most_common = Counter(detection_buffer).most_common(1)[0][0] if detection_buffer else "no_detection"
            send_emotion(most_common)
            detection_buffer.clear()
            last_sent = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_fer():
    from emotion_detection_app import EmotionDetectionApp

    app = EmotionDetectionApp(webhook_url=WEBHOOK_URL)
    app.run()


if __name__ == "__main__":
    try:
        run_yolo()
    except ModuleNotFoundError as error:
        if error.name == "ultralytics":
            print("ultralytics is not installed in this environment; using the FER-based webcam app instead.")
            run_fer()
        else:
            raise
