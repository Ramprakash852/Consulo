import cv2

for i in range(3):  # Try indexes 0 to 2
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is working!")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Camera {i} not available.")