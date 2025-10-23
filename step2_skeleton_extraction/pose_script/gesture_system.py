import os
import cv2
import json
import time
import numpy as np
from gesture_extractor import PointingGestureDetector
from gesture_util import read_json_locked, write_json_locked


def safe_read_image(image_path, max_retries=5, delay=0.1):
    """
    Safely read an image with multiple attempts.

    Args:
        image_path (str): Path to the image file.
        max_retries (int): Number of retries if loading fails.
        delay (float): Delay in seconds between retries.

    Returns:
        np.ndarray: Loaded image or None if loading fails.
    """
    for attempt in range(1, max_retries + 1):
        if not os.path.exists(image_path):
            print(f"⚠️ Attempt {attempt}/{max_retries}: Image {image_path} not found. Retrying...")
            time.sleep(delay)
            continue
        
        image = cv2.imread(image_path)
        if image is not None:
            return image  # Successfully loaded
        
        print(f"⚠️ Attempt {attempt}/{max_retries}: Failed to load image {image_path}. Retrying...")
        time.sleep(delay)

    print(f"❌ Failed to load image {image_path} after {max_retries} attempts.")
    return None  # Return None if still fails

class GestureProcessor:
    """
    Handles gesture detection and world mapping.
    """

    def __init__(self, detector, image_path, depth_path, transform_path):
        self.detector = detector
        self.image_path = image_path
        self.depth_path = depth_path
        self.transform_path = transform_path
        self.last_processed_time = 0

    def is_new_frame_available(self):
        """Check if a new frame is available by comparing file modification times."""
        if not os.path.exists(self.image_path):
            return False
        current_mod_time = os.path.getmtime(self.image_path)
        if current_mod_time > self.last_processed_time:
            self.last_processed_time = current_mod_time
            return True
        return False

    def process_gesture(self):
        """
        Process the latest gesture frame and update the world model.
        """
        # if not self.is_new_frame_available():
        #     print("no new image loaded")
        #     return False  # No new data to process
        
        image = safe_read_image(self.image_path)
        if image is None:
            return False
        
        # Run gesture detector
        curr_pointing = self.detector.run_spot()
        print(curr_pointing)
        return curr_pointing


class ObjectProcessor:
    """
    Handles object detection updates based on gesture inputs.
    """

    def __init__(self, object_list_path, gesture_vector_path):
        self.object_list_path = object_list_path
        self.gesture_vector_path = gesture_vector_path
        self.last_object_update_time = 0  # Track object list updates
        self.object_marks_record = self.load_object_marks()

    def load_object_marks(self):
        """Load existing object selection records from JSON file."""
        if os.path.exists(self.object_list_path):
            return read_json_locked(self.object_list_path)
        
        return {}  # Return empty record if file doesn't exist

    
    def is_point_in_cone(self, point, cone_origin, cone_direction, cone_angle, cone_conf, max_length=np.inf, sigma=0.2):
        """
        Compute confidence score based on whether a point is inside or outside the cone.
        
        Args:
            point (numpy.ndarray): The point to check [x, y, z].
            cone_origin (numpy.ndarray): The origin of the cone [x, y, z].
            cone_direction (numpy.ndarray): The direction vector of the cone.
            cone_angle (float): Half of the opening angle of the cone (in radians).
            max_length (float): Maximum length constraint (default: infinity).
            sigma (float): Standard deviation for the Gaussian falloff.

        Returns:
            float: 1 if inside the cone, otherwise a Gaussian-scaled value.
        """
        vector_to_point = np.array(point) - np.array(cone_origin)
        distance = np.linalg.norm(vector_to_point)

        # If point is beyond max length, return 0 (out of range)
        if distance > max_length:
            return 0

        # Normalize vectors
        vector_to_point_normalized = vector_to_point / distance
        cone_direction = cone_direction / np.linalg.norm(cone_direction)

        # Compute cosine of the angle between vectors
        cos_angle = np.dot(vector_to_point_normalized, cone_direction)
        angle_diff = np.arccos(np.clip(cos_angle, -1, 1))  # Ensure valid range

        # Check if inside cone
        # TODO: comment it out later
        if cos_angle >= np.cos(cone_angle):
            pointing_vector_conf = cone_conf
            return np.round(pointing_vector_conf, 6)  # Inside cone

        # Compute Gaussian-based confidence for outside points
        angle_diff_from_cone = angle_diff - cone_angle  # Distance from the cone boundary
        confidence = np.exp(-0.5 * (angle_diff_from_cone / sigma) ** 2)

        return np.round(confidence * cone_conf, 6)  # Lower confidence for points further from the cone


    def update_objects_in_cone(self):
        """
        Updates object probabilities based on gesture direction.
        """
        if not os.path.exists(self.gesture_vector_path):
            print("⚠️ No gesture data available.")
            return

        object_data = read_json_locked(self.object_list_path)
        gesture_data = read_json_locked(self.gesture_vector_path)

        cone_origin = gesture_data["origin_world"]
        cone_direction = gesture_data["direction_world"]
        cone_angle = np.deg2rad(gesture_data["pointing_vector_opening_angle"])
        cone_conf = gesture_data["pointing_vector_conf"]
        for item in object_data:

            object_world_point = item['center_3d']

            is_in_cone = self.is_point_in_cone(object_world_point, cone_origin, cone_direction, cone_angle, cone_conf)
            item["gest_prob"] = is_in_cone 

        write_json_locked(object_data, ".tmp/gesture_confidence.json")
        return object_data


    def is_new_object_list_available(self):
        """Check if the object list has been updated since last check."""
        if not os.path.exists(self.object_list_path):
            return False
        current_mod_time = os.path.getmtime(self.object_list_path)
        if current_mod_time > self.last_object_update_time:
            self.last_object_update_time = current_mod_time
            return True
        return False


class GestureSystem:
    def __init__(self, gesture_processor):
        self.gesture_processor = gesture_processor
        # self.object_processor = object_processor

    def run(self):
        """
        Runs the gesture detection only when a human is present.
        Stops when the human leaves.
        """
        print("🔍 Checking for human presence...")
        
        # Wait until a human is detected
        gesture_detected = self.gesture_processor.process_gesture()
        gesture_saved = os.path.exists("./.tmp/gesture_color.png")
        prev_gesture_detected = False
        while not self.gesture_processor.detector.human_presence:
            print("🚫 No human detected. Waiting...")
        #     time.sleep(0.1)

        print("👤 Human detected! Starting gesture detection...")
        
        while (not gesture_saved) or self.gesture_processor.detector.human_presence:
            new_image_available = self.gesture_processor.is_new_frame_available()
            # new_objects_available = self.object_processor.is_new_object_list_available()
            
            if new_image_available:
                print("🖼️ New image detected. Processing...")
                self.gesture_processor.process_gesture()
                # if not self.gesture_processor.detector.human_presence:
                #     continue
                gesture_saved = os.path.exists("./.tmp/gesture_color.png")
                time.sleep(0.1)
                # if human finished pointing
                finish_pointing = self.gesture_processor.detector.pointing != self.gesture_processor.detector.previous_pointing
                if (gesture_saved and finish_pointing):
                    print("📡 Updating objects in cone...")
                    self.object_processor.update_objects_in_cone()
                    print("loop1")
                    
                else:   
                    print(f"🖐 Gesture Detected: {gesture_detected} | Human Presence: {self.gesture_processor.detector.human_presence} {self.gesture_processor.detector.pointing_hand_handedness}")
    
        print("🚶‍♂️ Human left. Stopping gesture detection.")
        # TODO: refer to this code for check object update
        # while True:
        #     if new_objects_available:
        #         print("🆕 New objects detected. Checking if they are in the cone...")
        #         self.object_processor.update_objects_in_cone()
        #     self.object_processor.update_objects_in_cone()


        
        
if __name__ == "__main__":
    image_path = "./.tmp/gesture_color.png"
    depth_path = "./.tmp/gesture_depth.png"
    matrix_path = "./.tmp/gesture_transformation.json"
    vector_path = "./.tmp/gesture_vector.json"
    try:
        os.remove(image_path)
        os.remove(depth_path)
        os.remove(matrix_path)
        os.remove(vector_path)
    except FileNotFoundError:
        print("deleted")

    detector = PointingGestureDetector()

    gesture_processor = GestureProcessor(
        detector=detector,
        image_path="./spot_util/hand_color_image.png",
        depth_path="./spot_util/hand_depth_image.png",
        transform_path="./spot_util/hand_intrinsics_and_transform.json"
    )

    object_processor = ObjectProcessor(
        object_list_path="./.tmp/detection_confidence.json",
        gesture_vector_path="./.tmp/gesture_vector.json"
    )
    
    

    system = GestureSystem(gesture_processor)
    system.run()