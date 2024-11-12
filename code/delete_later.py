import cv2
import numpy as np

# Open the video file
video_path = '/home/xhe71/Downloads/1017_flag_test/1027_1/_Color.mp4'
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create masks for red and green
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours for red and green frames
    contours_red, contours_red_mat = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_green, contours_green_mat = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours_red_mat[0]), len(contours_green_mat[0]))
    # Draw contours on the frame
    cv2.drawContours(frame, contours_red, -1, (0, 0, 255), 2)  # Red contours
    cv2.drawContours(frame, contours_green, -1, (0, 255, 0), 2)  # Green contours

    # Display the frame
    cv2.imshow('Video Frame with Red and Green Frames', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
