import pandas as pd
import numpy as np
import yaml
import cv2
import ast
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gesture_util import *
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# -------------------------------
# Gesture Data Processor Class
# -------------------------------
class GestureDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.root_path = file_path.rsplit('/', 1)[0]  # Get the root path from the file path
        self.dog_id = file_path.split('/')[-3].split('.')[0].split('_')[0]
        self.dog_name = file_path.split('/')[-3].split('.')[0].split('_')[1]  # Extract trial name from the file path
        self.trial_number = file_path.split('/')[-2].split('.')[0]
        self.rgb_folder = os.path.join(self.root_path,'Color')
        # get the first index in the rgb folder
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_folder) if f.endswith('.jpg') or f.endswith('.png')])
        # start frame is the first frame in the rgb folder
        self.global_start_frame = int(self.rgb_files[0].split('_')[1].split('.')[0]) if self.rgb_files else 0
        intrinsics_path = "config/camera_config.yaml"
        targets_path = "config/targets.yaml"
        human_path = "config/human.yaml"
        if os.path.exists(os.path.join(self.root_path, 'fig')):
            # remove existing 'fig' directory if it exists
            import shutil
            shutil.rmtree(os.path.join(self.root_path, 'fig'))
        os.makedirs(os.path.join(self.root_path, 'fig'), exist_ok=True)  # Ensure root path exists


        # Load gesture data
        self.gesture_data = pd.read_csv(file_path)
        # Load camera intrinsics
        self.camera_intrinsics = yaml.safe_load(open(intrinsics_path, 'r'))
        # Load targets
        self.targets = yaml.safe_load(open(targets_path, 'r'))['targets']
        # Load human target
        self.human_target = yaml.safe_load(open(human_path, 'r'))['targets'][0]  # Assuming the first target is the human target

        # Print loaded data for verification
        print(f"Processing gesture data from: {file_path}")

    @staticmethod
    def transform_points(points, transformation_matrix):
        """
        Applies a 4x4 transformation matrix to a list of 3D points.
        Args:
            points (np.ndarray): Nx3 array of points.
            transformation_matrix (np.ndarray): 4x4 transformation matrix.
        Returns:
            np.ndarray: Transformed Nx3 points.
        """
        if points is None:
            return None
        N = points.shape[0]
        homog = np.hstack([points, np.ones((N, 1))])
        transformed = (transformation_matrix @ homog.T).T
        return transformed[:, :3]

    @staticmethod
    def parse_landmarks(landmark_str, mode="mediapipe", output_format="list"):
        """
        Parse a landmark string either in Mediapipe-style or nested list-style format.
        Args:
            landmark_str (str): Input string representing landmarks.
            mode (str): 'mediapipe' for parsing Mediapipe-style strings,
                        'list' for parsing Python-style nested list strings.
            output_format (str): 'dict' or 'list' for mediapipe parsing format.
        Returns:
            list: Parsed list of landmarks.
        """
        try:
            if mode == "mediapipe":
                pattern = r'landmark\s*{\s*x:\s*([-\d.eE]+)\s*y:\s*([-\d.eE]+)\s*z:\s*([-\d.eE]+)'
                matches = re.findall(pattern, landmark_str)
                if output_format == "dict":
                    return [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in matches]
                elif output_format == "list":
                    return [[float(x), float(y), float(z)] for x, y, z in matches]
                else:
                    raise ValueError("Invalid output_format. Use 'dict' or 'list'.")
            elif mode == "list":
                parsed = ast.literal_eval(landmark_str)
                if isinstance(parsed, list) and all(isinstance(p, list) for p in parsed):
                    if len(parsed) == 1 and isinstance(parsed[0], list):
                        return parsed[0]
                    return parsed
            else:
                raise ValueError("Invalid mode. Use 'mediapipe' or 'list'.")
        except Exception as e:
            print(f"Error parsing landmarks: {e}")
            return []

    @staticmethod
    def parse_vector_string(vector_str, to_numpy=False):
        try:
            # Replace 'array(' and ')' to turn it into a list-of-list string
            clean_str = vector_str.replace("array(", "").replace(")", "")
            parsed = ast.literal_eval(clean_str)
            return np.array(parsed) if to_numpy else parsed
        except Exception as e:
            print(f"Failed to parse vector string: {e}")
            return None

    @staticmethod
    def trim_data(self, start_frame, end_frame):
        
        """
        Trim the gesture data to the specified frame range.
        :param gesture_data: DataFrame containing gesture data
        :param start_frame: Starting frame number
        :param end_frame: Ending frame number
        :return: Trimmed DataFrame
        """
        gesture_data = self.gesture_data
        return gesture_data[(gesture_data['frame'] >= start_frame) & (gesture_data['frame'] <= end_frame)]

    @staticmethod
    def fit_ground_plane_w_targets(targets):
        """
        Fit a best-fit plane to 3D points using SVD (no assumption on axis orientation).
        Returns:
            - plane coefficients a, b, c, d such that ax + by + cz + d = 0
            - 4x4 transformation matrix that aligns this plane with the y=0 plane
        """
        points = [target['position_m'] for target in targets if 'position_m' in target]
        if len(points) < 3:
            raise ValueError("At least 3 points are required to fit a plane.")
        points = np.array(points)
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1, :]  # last row is normal to the best-fit plane
        a, b, c = normal
        d = -np.dot(normal, centroid)
        # Compute transformation matrix to align the plane to y=0
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        target_normal = np.array([0, 1, 0])
        target_normal = target_normal / np.linalg.norm(target_normal)
        axis = np.cross(plane_normal, target_normal)
        angle = np.arccos(np.clip(np.dot(plane_normal, target_normal), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-6:
            R = np.eye(3)
        else:
            from scipy.spatial.transform import Rotation as Rscipy
            R = Rscipy.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
        # The transformation matrix first translates centroid to origin, rotates, then translates back
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ centroid
        return (a, b, c, d), T

    def plot_2d_pointing_trace(self, processed_gesture_data, targets, human):

        color_map = {
            'eye_to_wrist': 'r',
            'shoulder_to_wrist': 'g',
            'elbow_to_wrist': 'b',
            'nose_to_wrist': 'm'
        }

        fig, ax = plt.subplots()

        # Plot targets
        for target in targets:
            x, _, z = target['position_m']
            ax.scatter(x, z, c='black', marker='x')
            ax.text(x, z, f"{target['id']}", fontsize=9, color='black')

        # Plot human center
        x, _, z = human
        ax.scatter(x, z, c='gray', marker='o')
        ax.text(x, z, "Human", fontsize=9, color='gray')

        # Plot ground intersections with alpha increasing by frame
        num_frame = len(processed_gesture_data['frame'])
        
        for idx, row in processed_gesture_data.iterrows():
            frame = row['frame']
            
            alpha =  0.1+0.9*(idx / num_frame)  # alpha from 0.1 to 1.0
            for vec_key, color in color_map.items():
                intersection_key = f'{vec_key}_ground_intersection'
                if (
                    intersection_key in row and
                    isinstance(row[intersection_key], list) and
                    len(row[intersection_key]) == 3
                ):
                    x, _, z = row[intersection_key]
                    ax.plot(x, z, marker='.', color=color, alpha=alpha)

        # Add legend
        legend_handles = [
            mpatches.Patch(color=color, label=vec_key.replace('_', ' '))
            for vec_key, color in color_map.items()
        ]
        legend_handles.append(mpatches.Patch(color='black', label='Targets'))
        legend_handles.append(mpatches.Patch(color='gray', label='Human'))
        ax.legend(handles=legend_handles)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_xlim([1.5, -1.5])
        ax.set_ylim([-2, 2.5])
        ax.set_title(f'{self.dog_id} 2D Ground Intersection Points of Pointing Vectors')
        ax.grid(True)

        plt.savefig(os.path.join(self.root_path, '2d_pointing_trace.png'), dpi=150)
        plt.close(fig)  # Close the figure to free memory
    def plot_3d_skeleton(self, landmarks_3d, targets=None,  vectors=None, wrist_location=None, head_orientation=None, frame_id = None):
        """
        Plot a 3D skeleton with optional targets, ground plane, and wrist-based pointing vectors.
        If head orientation vector is present in `vectors`, it is also visualized.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Set a fixed viewing angle
        ax.view_init(elev=40, azim=40, roll=115)
        xs = [pt[0] for pt in landmarks_3d]
        ys = [pt[1] for pt in landmarks_3d]
        zs = [pt[2] for pt in landmarks_3d]
        ax.scatter(xs, ys, zs, c='blue', marker='o')
        # Optionally connect points for visualization (basic limbs connections in Mediapipe pose)
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),  # shoulders
            (23, 24),  # hips
            (11, 23), (12, 24),  # torso
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28)   # right leg
        ]
        for start, end in connections:
            if start < len(landmarks_3d) and end < len(landmarks_3d):
                ax.plot([landmarks_3d[start][0], landmarks_3d[end][0]],
                        [landmarks_3d[start][1], landmarks_3d[end][1]],
                        [landmarks_3d[start][2], landmarks_3d[end][2]], c='blue')
        # Plot targets if provided
        if targets:
            for target in targets:
                pos = target.get('position_m', None)
                if pos and len(pos) == 3:
                    ax.scatter(pos[0], pos[1], pos[2], c='blue', marker='x')
                    ax.text(pos[0], pos[1], pos[2], f"{target.get('id', '')}", color='blue')
        # Add vector arrows from wrist to ground plane if vectors is provided
        if vectors is not None:
            vectors_to_plot = {
                'eye_to_wrist': 'r',
                'shoulder_to_wrist': 'g',
                'elbow_to_wrist': 'purple',
                'nose_to_wrist': 'm'
            }
            for vec_name, color in vectors_to_plot.items():
                vec = vectors.get(vec_name)
                wrist = wrist_location if wrist_location is not None else None
                vec = np.array(vec)
                wrist = np.array(wrist)
                if vec[1] != 0:
                    scale = wrist[1] / vec[1]
                    end_point = wrist - vec * scale
                    ax.plot([wrist[0], end_point[0]],
                            [wrist[1], end_point[1]],
                            [wrist[2], end_point[2]], color=color, label=vec_name)
            # Plot head orientation vector if available
            head_vec = head_orientation.get('head_orientation_vector')
            head_origin = head_orientation.get('head_orientation_origin')
            head_vec = np.array(head_vec) * 2 # Scale for visibility
            head_origin = np.array(head_origin)
            end_point = head_origin + head_vec
            ax.plot([head_origin[0], end_point[0]],
                        [head_origin[1], end_point[1]],
                        [head_origin[2], end_point[2]],
                        color='yellow', label='head_orientation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-0.2, 2])
        ax.set_zlim([-1, 1])
        ax.set_aspect('auto')
        ax.set_title(f'{self.dog_id}_{frame_id}_3D Skeleton Plot')
        ax.legend()
        plt.savefig(os.path.join(self.root_path,'fig', f'{frame_id}_skeleton_plot.png'), dpi=150)
        plt.close(fig)  # Close the figure to free memory
        # Do not call plt.show() here to allow for further drawing before showing.
        return ax, fig
    def process_data(self, trimmed_data, forced_pointing_arm=None):
        """
        Process gesture data with optional forced pointing arm selection.
        
        Args:
            trimmed_data: DataFrame containing trimmed gesture data
            forced_pointing_arm: Optional string ('Left', 'Right') to force pointing arm selection
                            If None, uses automatic detection based on frame counts
        """
        LEFT_WRIST_INDEX = 15
        RIGHT_WRIST_INDEX = 16
        
        # Fit ground plane and get transformation matrix
        ground_plane, T = self.fit_ground_plane_w_targets(self.targets)
        
        # Transform targets
        transformed_targets = []
        human_target_pos = np.array(self.human_target['position_m'])
        # Transform human target position to ground plane
        transformed_human_target = self.transform_points(np.array([human_target_pos]), T)[0]
        
        for target in self.targets:
            pos = np.array(target['position_m'])
            transformed_pos = self.transform_points(np.array([pos]), T)[0]
            transformed_targets.append({**target, 'position_m': transformed_pos.tolist()})
        
        ground_plane, _ = self.fit_ground_plane_w_targets(transformed_targets)
        
        # Process gesture data
        data = []
        for index, row in trimmed_data.iterrows():
            frame_id = row['frame']
            pointing_arm = row['pointing_arm']
            wrist_location = self.parse_vector_string(row['wrist_location'])
            
            # parse landmarks
            landmarks_2d = self.parse_landmarks(row['landmarks'], mode="mediapipe", output_format="list")
            landmarks_3d = self.parse_landmarks(row['landmarks_3d'], mode="list", output_format="list")
            
            # Transform the 3D landmarks before plotting them
            if landmarks_3d is None:
                continue
            transformed_landmarks_3d = self.transform_points(np.array(landmarks_3d), T)
            
            # --- Transform and plot gesture vectors ---
            vectors = {
                "eye_to_wrist": self.parse_vector_string(row['eye_to_wrist']),
                "shoulder_to_wrist": self.parse_vector_string(row['shoulder_to_wrist']),
                "elbow_to_wrist": self.parse_vector_string(row['elbow_to_wrist']),
                "nose_to_wrist": self.parse_vector_string(row['nose_to_wrist']),
            }
            
            # Transform vectors and wrist location to aligned ground frame
            vectors_to_plot = {}
            handedness = row['pointing_arm']
            wrist_index = LEFT_WRIST_INDEX if handedness == 'Left' else RIGHT_WRIST_INDEX
            wrist_transformed = transformed_landmarks_3d[wrist_index]
            
            distances = {}
            # Compute ground intersection for each vector
            ground_intersections = {}
            for name, vec in vectors.items():
                vec = np.array(vec)
                vec_rotated = T[:3, :3] @ vec.reshape(3, 1)  # Apply rotation
                vec_rot = vec_rotated.flatten()
                vectors_to_plot[name] = vec_rot.tolist()
                
                # Compute intersection with y=0 plane (ground)
                if vec_rot[1] != 0:
                    scale = wrist_transformed[1] / vec_rot[1]
                    intersection = wrist_transformed - vec_rot * scale
                    ground_intersections[f"{name}_ground_intersection"] = intersection.tolist()
                    
                    dists = []
                    for target in transformed_targets:
                        pos = np.array(target['position_m'])
                        dist = np.linalg.norm(intersection - pos)
                        dists.append(dist)
                    distances[name] = dists
                else:
                    ground_intersections[f"{name}_ground_intersection"] = [None, None, None]
                    distances[name] = [None] * len(transformed_targets)

            # --------- HEAD ORIENTATION COMPUTATION -----------
            # (keeping the existing head orientation code unchanged)
            head_orientation_vector = None
            head_orientation_origin = None
            head_ground_intersection = [None, None, None]
            head_orientation_dist_to_targets = [None] * 4
            
            try:
                # Use transformed landmarks for orientation in ground-aligned frame
                left_eye = transformed_landmarks_3d[2]
                right_eye = transformed_landmarks_3d[5]
                left_mouth = transformed_landmarks_3d[9]
                right_mouth = transformed_landmarks_3d[10]
                nose = transformed_landmarks_3d[0]
                
                left_eye_vec = np.array(nose) - np.array(left_eye)
                right_eye_vec = np.array(nose) - np.array(right_eye)
                eye_vec = (left_eye_vec + right_eye_vec) / 2
                
                left_mouth_vec = np.array(nose) - np.array(left_mouth)
                right_mouth_vec = np.array(nose) - np.array(right_mouth)
                mouth_vec = (left_mouth_vec- right_mouth_vec) / 2
                
                head_orientation_vector = (eye_vec + mouth_vec) / 2/np.linalg.norm((eye_vec + mouth_vec) / 2)
                head_orientation_origin = nose.tolist() if isinstance(nose, np.ndarray) else nose
                
                # --- Compute ground intersection for head orientation ---
                if head_orientation_vector[1] != 0:
                    scale = head_orientation_origin[1] / head_orientation_vector[1]
                    head_ground_intersection = (np.array(head_orientation_origin) - scale * head_orientation_vector).tolist()
                else:
                    head_ground_intersection = [None, None, None]
                
                # --- Compute distances to each target ---
                head_orientation_dist_to_targets = []
                for i, target in enumerate(transformed_targets, start=1):
                    target_pos = np.array(target["position_m"])
                    if head_ground_intersection[0] is not None:
                        dist = np.linalg.norm(target_pos - head_ground_intersection)
                    else:
                        dist = None
                    head_orientation_dist_to_targets.append(dist)
            except Exception as e:
                # If landmarks are not available, keep as None
                head_orientation_vector = None
                head_orientation_origin = None
                head_ground_intersection = [None, None, None]
                head_orientation_dist_to_targets = [None] * 4
            
            # Store row data for later DataFrame
            ground_intersection = ground_intersections.get('eye_to_wrist_ground_intersection')
            
            # Build row dictionary for saving, inserting ground intersections after wrist_location
            new_row = {
                'frame': frame_id,
                'pointing_hand_handedness': pointing_arm,
                'wrist_location': wrist_location,
            }
            
            # Insert ground intersection points in order
            for key in [
                'eye_to_wrist_ground_intersection',
                'shoulder_to_wrist_ground_intersection',
                'elbow_to_wrist_ground_intersection',
                'nose_to_wrist_ground_intersection'
            ]:
                new_row[key] = ground_intersections.get(key)
            
            new_row['ground_intersection'] = ground_intersection
            new_row['landmarks'] = landmarks_3d
            new_row['pointing_confidence'] = row.get('confidence', None)
            new_row['vectors'] = {
                'eye_to_wrist': vectors_to_plot.get('eye_to_wrist'),
                'shoulder_to_wrist': vectors_to_plot.get('shoulder_to_wrist'),
                'elbow_to_wrist': vectors_to_plot.get('elbow_to_wrist'),
                'nose_to_wrist': vectors_to_plot.get('nose_to_wrist'),
                'eye_to_wrist_distances': distances.get('eye_to_wrist'),
                'shoulder_to_wrist_distances': distances.get('shoulder_to_wrist'),
                'elbow_to_wrist_distances': distances.get('elbow_to_wrist'),
                'nose_to_wrist_distances': distances.get('nose_to_wrist'),
            }
            
            # Add head orientation to row
            new_row['head_orientation_vector'] = head_orientation_vector
            new_row['head_orientation_origin'] = head_orientation_origin
            # Add head orientation ground intersection and distances to targets
            new_row['head_orientation_ground_intersection'] = head_ground_intersection
            for i in range(4):
                new_row[f'head_orientation_dist_to_target_{i+1}'] = head_orientation_dist_to_targets[i]
            
            head_vector = {
                'head_orientation_vector': head_orientation_vector,
                'head_orientation_origin': head_orientation_origin
            }
            
            data.append(new_row)
            
            # Plot skeleton and ground plane, and pass row for vector arrows
            self.plot_3d_skeleton(transformed_landmarks_3d, transformed_targets, vectors_to_plot, 
                                wrist_location=wrist_transformed, head_orientation=head_vector, frame_id=frame_id)

        # Save trimmed results
        rows = []
        for row in data:
            # Prepare row_output with requested column order
            row_output = {
                'frame': row.get('frame'),
                'global_frame': row.get('frame') + self.global_start_frame,  # Adjust global frame
                'pointing_arm': row.get('pointing_hand_handedness'),
                'pointing_to': None,
                'wrist_location': row.get('wrist_location'),
                'eye_to_wrist_ground_intersection': row.get('eye_to_wrist_ground_intersection'),
                'shoulder_to_wrist_ground_intersection': row.get('shoulder_to_wrist_ground_intersection'),
                'elbow_to_wrist_ground_intersection': row.get('elbow_to_wrist_ground_intersection'),
                'nose_to_wrist_ground_intersection': row.get('nose_to_wrist_ground_intersection'),
                'ground_intersection': row.get('ground_intersection'),
                'head_orientation_ground_intersection': row.get('head_orientation_ground_intersection'),
            }
            
            for vec_name in ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist']:
                vec_key = f'{vec_name}_vec'
                row_output[vec_key] = row['vectors'].get(vec_name)
                
                if vec_name == 'shoulder_to_wrist':
                    min_dist_to_target = 100
                    min_dist_target_index = -1
                
                for i in range(4):
                    dist_key = f'{vec_name}_dist_to_target_{i+1}'
                    row_output[dist_key] = row['vectors'].get(f'{vec_name}_distances', [None]*4)[i]
                    
                    if vec_name == 'shoulder_to_wrist' and row_output[dist_key] is not None and row_output[dist_key] < min_dist_to_target:
                        min_dist_to_target = row_output[dist_key]
                        min_dist_target_index = i+1
            
            row_output['pointing_to'] = min_dist_target_index
            
            # Add head orientation distances to targets
            for i in range(4):
                row_output[f'head_orientation_dist_to_target_{i+1}'] = row.get(f'head_orientation_dist_to_target_{i+1}')
            
            row_output['confidence'] = row.get('pointing_confidence')
            row_output['landmarks'] = row.get('landmarks')
            # Add head orientation columns
            row_output['head_orientation_vector'] = row.get('head_orientation_vector')
            row_output['head_orientation_origin'] = row.get('head_orientation_origin')
            
            rows.append(row_output)

        # Define columns in the desired order
        columns = [
            'frame', 'global_frame','pointing_arm', 'pointing_to', 'wrist_location',
            'eye_to_wrist_ground_intersection',
            'shoulder_to_wrist_ground_intersection',
            'elbow_to_wrist_ground_intersection',
            'nose_to_wrist_ground_intersection',
            'ground_intersection',
            'head_orientation_ground_intersection',
            'eye_to_wrist_vec', 'eye_to_wrist_dist_to_target_1', 'eye_to_wrist_dist_to_target_2', 'eye_to_wrist_dist_to_target_3', 'eye_to_wrist_dist_to_target_4',
            'shoulder_to_wrist_vec', 'shoulder_to_wrist_dist_to_target_1', 'shoulder_to_wrist_dist_to_target_2', 'shoulder_to_wrist_dist_to_target_3', 'shoulder_to_wrist_dist_to_target_4',
            'elbow_to_wrist_vec', 'elbow_to_wrist_dist_to_target_1', 'elbow_to_wrist_dist_to_target_2', 'elbow_to_wrist_dist_to_target_3', 'elbow_to_wrist_dist_to_target_4',
            'nose_to_wrist_vec', 'nose_to_wrist_dist_to_target_1', 'nose_to_wrist_dist_to_target_2', 'nose_to_wrist_dist_to_target_3', 'nose_to_wrist_dist_to_target_4',
            'head_orientation_dist_to_target_1', 'head_orientation_dist_to_target_2', 'head_orientation_dist_to_target_3', 'head_orientation_dist_to_target_4',
            'head_orientation_vector', 'head_orientation_origin','landmarks', 'confidence'
        ]
        
        df = pd.DataFrame(rows, columns=columns)
        
        # NEW: Handle forced pointing arm selection
        if forced_pointing_arm is not None:
            print(f"🔒 Forcing pointing arm to: {forced_pointing_arm}")
            # Filter only frames with the forced pointing arm
            df = df[df['pointing_arm'] == forced_pointing_arm].reset_index(drop=True)
            print(f"📊 Filtered to {len(df)} frames with {forced_pointing_arm} arm")
        else:
            # Original logic: Count number of 'left' and 'right' in the pointing_arm column
            left_count = (df['pointing_arm'] == 'Left').sum()
            right_count = (df['pointing_arm'] == 'Right').sum()

            print(f"🫲 Left pointing frames: {left_count}")
            print(f"🫱 Right pointing frames: {right_count}")

            # Determine dominant hand (the one with more frames)
            dominant_hand = 'Left' if left_count > right_count else 'Right'

            # Filter only frames with the dominant pointing arm
            df = df[df['pointing_arm'] == dominant_hand].reset_index(drop=True)
            print(f"📊 Using dominant hand: {dominant_hand} ({len(df)} frames)")

        self.plot_2d_pointing_trace(df, transformed_targets, transformed_human_target)
        
        df.to_csv(os.path.join(self.root_path,"processed_gesture_data.csv"), index=False)
        return df


# --------- Script Entry Point ----------
if __name__ == "__main__":
    file_path = f"/Users/ivy/Library/CloudStorage/GoogleDrive-xiao_he@brown.edu/Shared drives/pointing_production_dog/BDL202_Dee-Dee_front/2/gesture_data.csv"  # Path to your gesture data file
    Gesture_data_processor = GestureDataProcessor(file_path)
    trimmed_data = Gesture_data_processor.trim_data(Gesture_data_processor, start_frame=129, end_frame=180)  # Example frame range
    Gesture_data_processor.process_data(trimmed_data)