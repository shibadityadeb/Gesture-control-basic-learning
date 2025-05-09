import cv2
from hand_gesture_detector import detect_hand_gesture

# Initialize the video capture (you can use a video file or a webcam)
cap = cv2.VideoCapture(0)  # Use 'your_video.mp4' if you want to play a video
is_playing = True  # Initially, play the video

while True:
    # Read the video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform hand gesture detection
    is_open, processed_frame = detect_hand_gesture(frame)

    # Control video playback based on hand gesture
    if is_open:
        is_playing = True  # Open palm, play the video
    else:
        is_playing = False  # Closed palm, pause the video

    # Show the appropriate frame
    if is_playing:
        cv2.imshow('Video Player', processed_frame)
    else:
        cv2.imshow('Video Player - Paused', processed_frame)

    # Display a message on the screen (optional)
    cv2.putText(processed_frame, 'Open Palm = Play, Closed Palm = Pause', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
