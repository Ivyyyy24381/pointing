import cv2
import mediapipe as mp
import numpy as np
import time
from gesture_util import *
import sys
import argparse
import pandas as pd

class PointingGestureDetector:
    __WRIST_INDEX_COLOR = (255, 153, 51)
    __ELBOW_WRIST_COLOR = (51, 255, 153)
    __SHOULDER_WRIST_COLOR = (51, 153, 255)
    __EYE_WRIST_COLOR = (153, 51, 255)
    __NOSE_WRIST_COLOR = (255, 51, 153)
    __GAZE_COLOR = (160, 160, 160)

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence = 0.7, smooth_landmarks = True, model_complexity = 2)
        self.pointing = False
        self.frame = 0
        self.gesture_start_time = time.time()
        self.previous_time = time.time() 
        self.gesture_duration = 0.0
        self.pointing_weighted_sum = {
            "eye_to_wrist": np.array([0, 0, 0]),
            "shoulder_to_wrist": np.array([0, 0, 0]),
            "elbow_to_wrist": np.array([0, 0, 0]),
            "nose_to_wrist": np.array([0, 0, 0]),
        }
        self.weighted_pointing_vector = self.pointing_weighted_sum
        self.weighted_pointing_cone = self.pointing_weighted_sum
        self.pointing_hand_handedness = ''
        self.pointing_count = 0

        # Data storage (using pandas DataFrame)
        self.data = pd.DataFrame(columns=[
            'frame', 'gesture_duration', 'pointing_count', 'pointing_arm',
            'eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist',
            'wrist_location', 'landmarks', 'landmarks_3d'
        ])

        
    def clear_values(self):
        self.pointing = False
        self.gesture_start_time = time.time()
        self.previous_time = time.time() 
        self.gesture_duration = 0.0
        self.pointing_weighted_sum = {
            "eye_to_wrist": np.array([0, 0, 0]),
            "shoulder_to_wrist": np.array([0, 0, 0]),
            "elbow_to_wrist": np.array([0, 0, 0]),
            "nose_to_wrist": np.array([0, 0, 0]),
        }
        self.weighted_pointing_vector = self.pointing_weighted_sum
        self.weighted_pointing_cone = self.pointing_weighted_sum

    

    def update_weighted_vector(self, vectors, duration):
        """
        Update the weighted sum for each vector in the dictionary of vectors based on duration.

        Args:
            vectors: A dictionary of vectors, e.g., {'eye_to_wrist': np.array([...]), 'shoulder_to_wrist': np.array([...])}
            duration: Duration over which the vector is being updated.
        """
        if vectors is not None and duration is not None:
            for key, vector in vectors.items():
                print(key, vector)
                if vector is not None:  # Only update if the vector is valid
                    self.pointing_weighted_sum[key] = self.pointing_weighted_sum[key] + np.array(vector) * duration
            
            # Update total gesture duration
            self.gesture_duration += duration
            
            # Compute the weighted pointing vector for each vector type
            self.weighted_pointing_vector = {
                key: self.pointing_weighted_sum[key] / self.gesture_duration if self.gesture_duration > 0 else np.array([0, 0, 0])
                for key in self.pointing_weighted_sum
            } 
        
    # def is_pointing_hand(self, hand_landmarks, handedness):
    #     """
    #     Determines whether a hand is pointing.
    #     Returns:
    #         - handedness: 'Right' or 'Left'
    #         - confidence: float (confidence score of pointing)
    #         - is_pointing: bool (whether hand is pointing)
    #     """
    #     try:
    #         # Calculate the vector from the wrist to the index finger tip
    #         wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
    #         index_finger_tip = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    #         index_finger_pip = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    #         index_finger_mcp = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            
    #         index_finger_lower = calculate_vector(index_finger_pip, index_finger_mcp)
    #         index_finger_higher = calculate_vector(index_finger_pip, index_finger_tip)
    #         index_finger_bend_angle = np.rad2deg(angle_between(index_finger_lower,index_finger_higher))
    #         index_finger_vector = calculate_vector(wrist, index_finger_tip)

    #         # Check if the index finger is extended by comparing its vector's magnitude to the others
    #         middle_finger_tip = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    #         ring_finger_tip = hand_landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    #         pinky_tip = hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]

    #         middle_finger_vector = calculate_vector(wrist, middle_finger_tip)
    #         ring_finger_vector = calculate_vector(wrist, ring_finger_tip)
    #         pinky_vector = calculate_vector(wrist, pinky_tip)

    #         index_finger_extended = (
    #             np.linalg.norm(index_finger_vector) > 1.2 * np.linalg.norm(middle_finger_vector) and
    #             np.linalg.norm(index_finger_vector) > 1.2 * np.linalg.norm(ring_finger_vector) and
    #             np.linalg.norm(index_finger_vector) > 1.2 * np.linalg.norm(pinky_vector) and
    #             index_finger_bend_angle > 130
    #         )
            
    #         confidence = index_finger_bend_angle / 180  #confidence score based on bend angle
    #         is_pointing = index_finger_extended

    #         return handedness, confidence, is_pointing
        
    #     except (IndexError, TypeError, ZeroDivisionError):
    #         return handedness, 0.0, False

    # def is_pointing_arm(self, pose_landmarks):
    #     """
    #     Determines whether the arm is raised and extended.
    #     Returns:
    #         - handedness: 'Right' or 'Left'
    #         - confidence: float (confidence score based on arm posture)
    #         - is_pointing: bool (whether arm is raised and extended)
    #     """
    #     arm_handedness = 'Left'
    #     is_pointing = False
    #     try:
    #         # Get shoulder, elbow, and wrist landmarks based on handedness
    #         r_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    #         r_wrist = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    #         r_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

            
    #         l_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    #         l_wrist = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    #         l_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    
    #         # Calculate vectors & angle
    #         r_shoulder_to_wrist = calculate_vector(r_shoulder, r_wrist)
    #         r_shoulder_to_hip = calculate_vector(r_shoulder, r_hip)
    #         r_arm_raise_angle = np.rad2deg(angle_between(r_shoulder_to_hip, r_shoulder_to_wrist))
            
    #         l_shoulder_to_wrist = calculate_vector(l_shoulder, l_wrist)
    #         l_shoulder_to_hip = calculate_vector(l_shoulder, l_hip)
    #         l_arm_raise_angle = np.rad2deg(angle_between(l_shoulder_to_hip, l_shoulder_to_wrist))
            
    #         if (l_arm_raise_angle > r_arm_raise_angle):
    #             arm_handedness = 'Left'
    #             # confidence is calculated by left arm to right arm raising angle ratio
    #             confidence = l_arm_raise_angle / (l_arm_raise_angle + r_arm_raise_angle)
    #         else:
    #             arm_handedness = 'Right'
    #             confidence = r_arm_raise_angle / (l_arm_raise_angle + r_arm_raise_angle)
                
    #         is_pointing = confidence > 0.6

    #         return arm_handedness, confidence, is_pointing

    #     except (IndexError, AttributeError):
    #         return arm_handedness, 0.0, False
        
    def save_gesture_data(self, frame_num, gesture_duration, pointing_arm, vectors, wrist_location, landmarks_2d, landmarks_3d):
        """
        Save the detected pointing gesture data into the dataframe.
        
        Args:
            gesture_duration: Duration of the gesture.
            pointing_arm: The arm used for pointing (left or right).
            vectors: A dictionary of pointing vectors.
            wrist_location: The 3D location of the wrist.
        """
        # Add a new row to the dataframe with all the relevant data
        if vectors is not None:
            new_row = {
                'frame': frame_num,
                'gesture_duration': gesture_duration,
                'pointing_count': self.pointing_count,
                'pointing_arm': pointing_arm,
                'eye_to_wrist': [vectors['eye_to_wrist']],
                'shoulder_to_wrist': [vectors['shoulder_to_wrist']],
                'elbow_to_wrist': [vectors['elbow_to_wrist']],
                'nose_to_wrist': [vectors['nose_to_wrist']],
                'wrist_location': wrist_location, 
                'landmarks': landmarks_2d,
                'landmarks_3d': landmarks_3d
            }
        else:
            new_row = {
                'frame': frame_num,
                'gesture_duration': [""],
                'pointing_count': self.pointing_count,
                'pointing_arm': [""],
                'eye_to_wrist': [""],
                'shoulder_to_wrist': [""],
                'elbow_to_wrist': [""],
                'nose_to_wrist': [""],
                'wrist_location': [""], 
                'landmarks': landmarks_2d,
                'landmarks_3d': landmarks_3d
            }
        new_row = pd.DataFrame(new_row)
        
        self.data = pd.concat([self.data, new_row], ignore_index=True)

            
    def process_frame(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(image_rgb)
        pose_results = self.pose.process(image_rgb)
        
        pointing_hand = False
        is_hand_pointing = False
        is_arm_pointing = False
        hand_handedness = None
        prev_arm = self.pointing_hand_handedness
        
        # draw pose landmarks
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(image, hand_results.multi_hand_landmarks[idx], mp.solutions.hands.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        print('-----')        
        # for close proximity detect hand gestures
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = hand_results.multi_handedness[idx].classification[0].label
                # Check if the hand is pointing
                hand_handedness, hand_confidence, is_hand_pointing = is_pointing_hand(hand_landmarks.landmark, handedness)
                if is_hand_pointing:
                    print(f"Pointing detected with {hand_handedness} hand, confidence: {hand_confidence}")
                    self.pointing = True
                    self.pointing_hand_handedness = hand_handedness
                    pointing_confidence = hand_confidence
        
        landmarks = pose_results.pose_world_landmarks
        landmarks_2d = pose_results.pose_landmarks
        # for far proximity detect arm gestures
        if landmarks:
            # Check if the arm is raised and extended
            arm_handedness, arm_confidence, is_arm_pointing = is_pointing_arm(landmarks.landmark)
            if is_arm_pointing:
                print(f"Pointing detected with {arm_handedness} arm, confidence: {arm_confidence}")
                self.pointing = True
                self.pointing_hand_handedness = arm_handedness
                pointing_confidence = arm_confidence
        else:
            self.save_gesture_data(self.frame, None, None, None, None, landmarks_2d,landmarks)
            self.frame += 1
            return image
        if not (is_hand_pointing or is_arm_pointing):
            self.pointing = False
            self.pointing_hand_handedness = None
            pointing_confidence = 0

        # if pointing is detected, update and store weighted vectors
        if self.pointing:
            
            pointing_hand = self.pointing_hand_handedness
            vectors = self.find_vectors(pointing_hand, landmarks)
            print(vectors)
            joints = self.find_joint_locations(pointing_hand, landmarks)
            
            
            current_time = time.time()
            self.gesture_duration = time.time() - self.gesture_start_time
            self.previous_time = current_time
            if self.pointing_hand_handedness != prev_arm:
                self.pointing_count += 1
            prev_arm = self.pointing_hand_handedness
                
            # self.update_weighted_vector(vectors, duration)

            # Save gesture data
            self.save_gesture_data(self.frame, self.gesture_duration, self.pointing_hand_handedness, vectors, joints['wrist'],landmarks_2d, landmarks)
            
            vectors_2d = self.find_vectors(pointing_hand, landmarks_2d)
            joints_2d = self.find_joint_locations(pointing_hand, landmarks_2d)
            self.display_visualization(image, joints_2d, vectors_2d)
            self.display_info(image, self.pointing_hand_handedness, self.gesture_duration, vectors)
            
            
            # self.weighted_pointing_vector =  {
            # "eye_to_wrist": w_wrist_to_eye,
            # "shoulder_to_wrist": w_wrist_to_shoulder,
            # "elbow_to_wrist": w_wrist_to_elbow,
            # "nose_to_wrist": w_wrist_to_nose,
            # "wrist_to_index": w_wrist_to_index
            # }
        
        # if there is not pointing, clear all stored values
        else: 
            self.save_gesture_data(self.frame, None, None, None, None, landmarks_2d, landmarks)
            self.clear_values()
            
        self.frame += 1
        
        return image

    def find_vectors(self, pointing_hand, landmarks):
        """find eye-to-wrist, shoulder-to-wrist, elbow-to-wrist, and wrist-to-index vectors
        return a dictionary of all vectors and normalized wrist location
        """
        print(pointing_hand)
        try:
            wrist = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_WRIST if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

        except (IndexError, TypeError):
            wrist = None
       
            
        try:
            elbow = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            ]
        except (IndexError, AttributeError):
            elbow = None

        try:
            shoulder = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
            ]
        except (IndexError, AttributeError):
            shoulder = None

        try:
            eye = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_EYE if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_EYE
            ]
        except (IndexError, AttributeError):
            eye = None
        
        try:
            nose = landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        except (IndexError, AttributeError):
            nose = None

        eye_to_wrist_vector = calculate_vector(eye, wrist)
        shoulder_to_wrist_vector = calculate_vector(shoulder, wrist)
        elbow_to_wrist_vector = calculate_vector(elbow, wrist)
        nose_to_wrist_vector = calculate_vector(nose, wrist)
   

        return {
            "eye_to_wrist": eye_to_wrist_vector,
            "shoulder_to_wrist": shoulder_to_wrist_vector,
            "elbow_to_wrist": elbow_to_wrist_vector,
            "nose_to_wrist": nose_to_wrist_vector
        }

    def find_joint_locations(self, pointing_hand, landmarks):
        """find eye-to-wrist, shoulder-to-wrist, elbow-to-wrist, and wrist-to-index vectors
        return a dictionary of all vectors and normalized wrist location
        """
        try:
            wrist = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_WRIST if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        except (IndexError, TypeError):
            wrist = None
        
        try:
            index_finger_tip = pointing_hand[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        except (IndexError, TypeError):
            index_finger_tip = None
            
        try:
            elbow = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            ]
        except (IndexError, AttributeError):
            elbow = None

        try:
            shoulder = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
            ]
        except (IndexError, AttributeError):
            shoulder = None

        try:
            eye = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_EYE if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_EYE
            ]
            eye_middle = (mp.solutions.pose.PoseLandmark.LEFT_EYE + mp.solutions.pose.PoseLandmark.RIGHT_EYE)/2
        except (IndexError, AttributeError):
            eye = None
        
        try:
            nose = landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        except (IndexError, AttributeError):
            nose = None

        return {
            "eye": eye,
            "eye_middle": eye_middle,
            "shoulder": shoulder,
            "elbow": elbow,
            "nose": nose,
            "wrist": wrist,
            "index_finger": index_finger_tip
        }
        
        
    def display_visualization(self, image, joints, vectors):
        wrist = joints["wrist"]
        shoulder = joints["shoulder"]
        eye = joints["eye"]
        elbow = joints["elbow"]
        index_finger_tip = joints["index_finger"]
            
        visualize_vector(image, wrist, vectors["shoulder_to_wrist"], self.__SHOULDER_WRIST_COLOR)
        visualize_vector(image, wrist, self.weighted_pointing_vector["shoulder_to_wrist"], (0, 0, 0))
        visualize_vector(image, wrist, vectors["elbow_to_wrist"], self.__ELBOW_WRIST_COLOR)
        visualize_vector(image, wrist, vectors["eye_to_wrist"], self.__EYE_WRIST_COLOR)
        if index_finger_tip is not None:
            visualize_vector(image, wrist, self.weighted_pointing_vector, self.__GAZE_COLOR)

    def display_info(self, image, pointing_hand_handedness, gesture_duration, vectors):
        cv2.putText(image, f"Pointing Gesture Detected (Hand: {self.pointing_hand_handedness})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"Duration: {gesture_duration:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"Hand: {pointing_hand_handedness}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

        if vectors['eye_to_wrist'] is not None:
            cv2.putText(image, f"Eye-to-Wrist: {vectors['eye_to_wrist']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__EYE_WRIST_COLOR, 2)
        if vectors['shoulder_to_wrist'] is not None:
            cv2.putText(image, f"Shoulder-to-Wrist: {vectors['shoulder_to_wrist']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__SHOULDER_WRIST_COLOR, 2)
        if vectors['elbow_to_wrist'] is not None:
            cv2.putText(image, f"Elbow-to-Wrist: {vectors['elbow_to_wrist']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__ELBOW_WRIST_COLOR, 2)
        if vectors['nose_to_wrist'] is not None:
            cv2.putText(image, f"nose-to-Index: {vectors['nose_to_wrist']}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__NOSE_WRIST_COLOR, 2)
    
        
    def run_stream(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            # comment the line below if camera is not flipped.
            image = cv2.flip(image, 1)  # 1 flips horizontally

            processed_image = self.process_frame(image)
            cv2.imshow('Pointing Gesture Detection', processed_image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_video(self, video_path):
        """Runs the gesture detection on a local video file"""
        cap = cv2.VideoCapture(video_path)  # Open the video file
        frame_num = 0
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Flip image horizontally if necessary, else remove the flip
            # image = cv2.flip(image, 1)

            processed_image = self.process_frame(image)
            cv2.imshow('Pointing Gesture Detection', processed_image)
            import os
            output_path = video_path[0:-4]
            # Create the directory if it does not exist
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            cv2.imwrite(output_path+'/f%i.png'%(frame_num), processed_image)
            frame_num += 1
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
    def run_spot(self, robot_ip):
        # TODO: running gesture detection on spot
        return
# Usage
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Pointing Gesture Detection")
    parser.add_argument('--mode', type=str, required=True, choices=['live', 'video', 'spot'], help="Mode to run the gesture detection. Choices: 'live', 'video', 'spot'")
    parser.add_argument('--video_path', type=str, help="Path to the local video file. Required if mode is 'video'")
    parser.add_argument('--csv_path', type=str, help="Path to the saved csv file.'")

    args = parser.parse_args()

    detector = PointingGestureDetector()

    if args.mode == 'live':
        print("Running in live video mode (webcam)...")
        detector.run_stream()
        
    elif args.mode == 'video':
        if not args.video_path:
            print("Error: --video_path argument is required for 'video' mode")
        else:
            print(f"Running in local video mode with video: {args.video_path}")
            detector.run_video(args.video_path)
            
    elif args.mode == 'spot':
        print("Running in spot video mode...")
        detector.run_spot_video()
        
    detector.data.to_csv(args.csv_path, index=False)
    print(f"Data saved to {args.csv_path}")