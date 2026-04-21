import cv2

def test_webcam():
    # Try to open the default camera (usually 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check connections and permissions.")
        return False
    
    print("Webcam opened successfully!")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame from webcam.")
        cap.release()
        return False
    
    print("Successfully read a frame from the webcam!")
    
    # Display the frame for 3 seconds
    cv2.imshow('Webcam Test', frame)
    cv2.waitKey(3000)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    test_webcam()