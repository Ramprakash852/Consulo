import cv2
import numpy as np
import time
import json
import requests
from collections import Counter
from fer import FER

class EmotionDetectionApp:
    def __init__(self, webhook_url="http://127.0.0.1:8502/emotion"):
        # Initialize the webcam. DirectShow is often more reliable on Windows.
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Initialize FER detector
        print("Loading emotion detection model...")
        self.emotion_detector = FER(mtcnn=True)  # Using MTCNN for better face detection
        
        # Variables for emotion tracking
        self.emotions_buffer = []
        self.last_summary_time = time.time()
        
        # API endpoint for sending emotion data
        self.webhook_url = webhook_url
        
        print("Model loaded successfully!")
        
    def detect_emotions(self, frame):
        """Detect faces and emotions in a frame."""
        # Make a copy to avoid modifying the original frame
        result_frame = frame.copy()
        
        # Detect emotions
        detected_faces = self.emotion_detector.detect_emotions(frame)
        detected_emotions = []
        
        # Process each detected face
        for face in detected_faces:
            # Get face coordinates
            x, y, w, h = face['box']
            
            # Draw rectangle around face
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Get emotions
            emotions = face['emotions']
            
            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda item: item[1])[0]
            detected_emotions.append(dominant_emotion)
            
            # Display emotion on frame
            cv2.putText(result_frame, dominant_emotion, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display emotion scores
            y_offset = y + h + 15
            for emotion, score in emotions.items():
                text = f"{emotion}: {score:.2f}"
                cv2.putText(result_frame, text, (x, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                y_offset += 15
        
        return result_frame, detected_emotions
    
    def send_emotion(self, emotion):
        """Send emotion data to API endpoint."""
        payload = {"emotion": emotion}
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.webhook_url, json=payload, headers=headers)
            print(f"Sent: {payload} | Response: {response.status_code}")
            return True
        except Exception as e:
            print("Error sending data:", e)
            return False
    
    def run(self):
        """Run the emotion detection app."""
        print(f"Starting emotion detection. Press 'q' to quit.")
        print(f"Sending emotion data to: {self.webhook_url}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect emotions in current frame
            start_time = time.time()
            result_frame, current_emotions = self.detect_emotions(frame)
            processing_time = time.time() - start_time
            
            # Add detected emotions to buffer
            self.emotions_buffer.extend(current_emotions)
            
            # Check if 5 seconds have passed
            current_time = time.time()
            elapsed_time = current_time - self.last_summary_time
            
            # Show processing time
            cv2.putText(result_frame, f"Processing: {processing_time:.2f}s", 
                      (10, result_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.6, (255, 0, 0), 1)
            
            if elapsed_time >= 5:
                # Calculate most frequent emotion in buffer
                if self.emotions_buffer:
                    emotion_counts = Counter(self.emotions_buffer)
                    most_common_emotion = emotion_counts.most_common(1)[0][0]
                    print(f"Dominant emotion in last 5 seconds: {most_common_emotion}")
                    
                    # Send emotion data to API
                    self.send_emotion(most_common_emotion)
                    
                    # Display most common emotion as text overlay
                    cv2.putText(result_frame, f"Dominant: {most_common_emotion}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2)
                else:
                    print("No emotions detected in the last 5 seconds")
                    cv2.putText(result_frame, "No emotions detected", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2)
                    # Send no_detection to API
                    self.send_emotion("no_detection")
                    
                # Reset for next interval
                self.emotions_buffer = []
                self.last_summary_time = current_time
            
            # Display remaining time until next summary
            remaining = max(0, 5 - elapsed_time)
            cv2.putText(result_frame, f"Next update: {remaining:.1f}s", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
            # Display the resulting frame
            cv2.imshow('Emotion Detection', result_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

# Run the app
if __name__ == "__main__":
    try:
        # API endpoint - change this to your actual endpoint
        WEBHOOK_URL = "http://127.0.0.1:8502/emotion"
        
        # Start the app
        app = EmotionDetectionApp(webhook_url=WEBHOOK_URL)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")