def load_side_to_front_transform():
    transform_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "side_to_front_transform.json")
    if os.path.exists(transform_path):
        with open(transform_path, 'r') as f:
            data = json.load(f)
        return np.array(data['transform'])
    return np.eye(4)
import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import copy
import argparse
from matplotlib import cm
import pandas as pd
import pyrealsense2 as rs
import yaml

from types import SimpleNamespace
# matplotlib.use('TkAgg')
# === Paths ===

def interpolate_trace(trace_3d):
    trace_array = np.array(trace_3d, dtype=np.float32)
    for dim in range(3):
        col = trace_array[:, dim]
        nans = np.isnan(col)
        not_nans = ~nans
        if np.sum(not_nans) >= 2:
            col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
        trace_array[:, dim] = col
    return trace_array.tolist()
def get_valid_depth(depth_frame, u, v, patch_size=5):
    """
    Robustly extract depth at (u, v) by sampling a local patch.
    Returns median of valid (nonzero) depths, or None if no valid depth.
    """
    h, w = depth_frame.shape
    half = patch_size // 2
    u, v = int(u), int(v)
    patch = depth_frame[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)]
    valid = patch[patch > 0]
    return np.median(valid) if len(valid) > 0 else None

def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """
    Convert 2D pixel (u, v) and depth to 3D point using camera intrinsics.
    """
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])


def load_intrinsics_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    intr_data = data['intrinsics']

    # Create a simple object to mimic RealSense intrinsics
    intr = SimpleNamespace(
        width=intr_data['width'],
        height=intr_data['height'],
        ppx=intr_data['ppx'],
        ppy=intr_data['ppy'],
        fx=intr_data['fx'],
        fy=intr_data['fy'],
        model=intr_data['model'],
        coeffs=intr_data['coeffs']
    )

    print("Width:", intr.width)
    print("Height:", intr.height)
    print("PPX (cx):", intr.ppx)
    print("PPY (cy):", intr.ppy)
    print("FX:", intr.fx)
    print("FY:", intr.fy)
    print("Distortion Model:", intr.model)
    print("Coefficients:", intr.coeffs)

    return intr

def find_realsense_intrinsics(bag_path):
    # === Load the bag file ===
    cfg = rs.config()
    cfg.enable_device_from_file(bag_path)

    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)

    # === Wait for frames to arrive ===
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # === Get stream profile ===
    color_stream = profile.get_stream(rs.stream.color)
    video_profile = color_stream.as_video_stream_profile()

    # === Extract intrinsics ===
    intr = video_profile.get_intrinsics()

    print("Width:", intr.width)
    print("Height:", intr.height)
    print("PPX (cx):", intr.ppx)
    print("PPY (cy):", intr.ppy)
    print("FX:", intr.fx)
    print("FY:", intr.fy)
    print("Distortion Model:", intr.model)
    print("Coefficients:", intr.coeffs)

    # === Stop the pipeline ===
    pipeline.stop()

    return intr


def plot_cube(ax, center, size=0.1, color='red', alpha=0.6):
    """
    Plot a cube at the given center with given size.
    center: (x, y, z)
    size: length of cube edge
    """
    r = size / 2
    X = np.array([
        [r, -r, -r, r, r, -r, -r, r],
        [r, r, -r, -r, r, r, -r, -r],
        [-r, -r, -r, -r, r, r, r, r]
    ])
    X = X.T
    X = X + np.array(center)

    faces = [
        [X[0], X[1], X[2], X[3]],  # bottom
        [X[4], X[5], X[6], X[7]],  # top
        [X[0], X[1], X[5], X[4]],  # front
        [X[2], X[3], X[7], X[6]],  # back
        [X[1], X[2], X[6], X[5]],  # left
        [X[4], X[7], X[3], X[0]],  # right
    ]
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cube = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha)
    ax.add_collection3d(cube)
    
    
def smooth_bbox(prev_bbox, curr_bbox, alpha=0.5):
    return [alpha * p + (1 - alpha) * c for p, c in zip(prev_bbox, curr_bbox)]

def load_scale_info(output_dir):
    """Load scale information from segmentation processing."""
    scale_metadata_path = os.path.join(output_dir, "sam2_scale_metadata.json")
    if os.path.exists(scale_metadata_path):
        with open(scale_metadata_path, 'r') as f:
            return json.load(f)
    return None

def correct_keypoint_coordinates(keypoints, scale_info, video_source_path):
    """Correct keypoint coordinates based on scale factors."""
    if scale_info is None:
        return keypoints  # No correction needed
        
    # Check if keypoints come from masked video (resized) or original images
    if "masked_video.mp4" in video_source_path:
        # Keypoints are from masked video which was created from resized images
        # Need to scale back to original resolution
        corrected_keypoints = []
        for frame_kps in keypoints:
            if not frame_kps:  # Empty frame
                corrected_keypoints.append(frame_kps)
                continue
                
            frame_corrected = []
            for kp_set in frame_kps:
                if not kp_set:  # No keypoints in this set
                    frame_corrected.append(kp_set)
                    continue
                    
                # Apply scaling correction
                corrected_kp_set = []
                for kp in kp_set:
                    if len(kp) >= 2:
                        # Use average scale factor (could be made more precise per frame)
                        avg_scale = np.mean(list(scale_info['scale_factors'].values()), axis=0)
                        x_corrected = kp[0] * avg_scale[0]
                        y_corrected = kp[1] * avg_scale[1]
                        corrected_kp = [x_corrected, y_corrected] + list(kp[2:])  # Keep confidence etc.
                        corrected_kp_set.append(corrected_kp)
                    else:
                        corrected_kp_set.append(kp)
                frame_corrected.append(corrected_kp_set)
            corrected_keypoints.append(frame_corrected)
        return corrected_keypoints
    else:
        # Keypoints are already at original resolution
        return keypoints

def load_trial_targets(trial_dir):
    """Load trial-specific target coordinates."""
    target_file = os.path.join(trial_dir, "target_coordinates.json")
    print(f"🔍 Checking for target file: {target_file}")
    
    if os.path.exists(target_file):
        try:
            with open(target_file, 'r') as f:
                data = json.load(f)
            
            # Convert to format expected by existing code
            targets = []
            for target in data.get("targets", []):
                targets.append({
                    "label": target["label"],
                    "x": target["world_coords"][0],
                    "y": target["world_coords"][1], 
                    "z": target["world_coords"][2]
                })
            print(f"✅ Loaded {len(targets)} manual targets from {target_file}")
            return targets
        except Exception as e:
            print(f"❌ Error loading target file {target_file}: {e}")
            return None
    else:
        print(f"⚠️ Target file not found: {target_file}")
    return None

def validate_coordinate_alignment(keypoints, depth_frame, scale_info):
    """Validate that 2D coordinates properly map to depth data."""
    if not keypoints:
        return True
        
    h, w = depth_frame.shape
    valid_count = 0
    total_count = 0
    
    for kp in keypoints[:10]:  # Check first 10 keypoints
        if len(kp) >= 2:
            x, y = int(kp[0]), int(kp[1])
            total_count += 1
            if 0 <= x < w and 0 <= y < h:
                depth_val = depth_frame[y, x]
                if depth_val > 0:  # Valid depth
                    valid_count += 1
                    
    alignment_ratio = valid_count / total_count if total_count > 0 else 0
    print(f"🔍 Coordinate alignment: {valid_count}/{total_count} ({alignment_ratio:.1%}) valid mappings")
    
    if alignment_ratio < 0.3:
        print("⚠️ Warning: Low coordinate alignment - scale correction may be needed")
        return False
    return True

def load_intrinsics_and_targets(json_path, side_view = False):
    bag_path = ''
    
    output_dir = os.path.dirname(json_path)
    parent_dir = os.path.dirname(output_dir)
    # Load standard intrinsics first (consistent across all trials)
    standard_intrinsics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard_intrinsics.yaml")
    if os.path.exists(standard_intrinsics_path):
        print(f"📐 Using standard camera intrinsics from {standard_intrinsics_path}")
        intr = load_intrinsics_from_yaml(standard_intrinsics_path)
    else:
        # Fallback to existing methods
        intr = None
        
        # Try rosbag metadata YAML
        if os.path.exists(os.path.join(parent_dir,"rosbag_metadata.yaml")):
            yaml_path = os.path.join(parent_dir,"rosbag_metadata.yaml")
            print(f"📐 Using rosbag metadata from {yaml_path}")
            intr = load_intrinsics_from_yaml(yaml_path)
        else:
            # Try to extract from bag file
            for file in os.listdir(parent_dir):
                if file.endswith(".bag"):
                    bag_path = os.path.join(parent_dir, file)
                    print(f"📐 Extracting intrinsics from {bag_path}")
                    intr = find_realsense_intrinsics(bag_path)
                    break
        
        # Final fallback
        if intr is None:
            print("📐 Using default intrinsics file")
            intr = load_intrinsics_from_yaml('./intrinsics.yaml')
    
    # Try to load trial-specific targets first (check trial directory)
    print(f"🔍 Looking for targets in trial directory: {output_dir}")
    targets = load_trial_targets(output_dir)
    
    # If not found in trial directory, check parent directory
    if targets is None:
        print(f"🔍 Looking for targets in parent directory: {parent_dir}")
        targets = load_trial_targets(parent_dir)
    
    if targets is None:
        # Fall back to static target configuration
        current_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(current_path)
        if side_view:
            target_json_path = os.path.join(parent_path, "target_coords_side.json")
        else:
            target_json_path = os.path.join(parent_path, "target_coords_front.json")
        targets = []
        if os.path.exists(target_json_path):
            with open(target_json_path, 'r') as f:
                targets = json.load(f)
            print(f"📥 Using static targets from {target_json_path}")
        else:
            # Load from YAML config as last resort
            config_path = os.path.join(parent_path, "config", "targets.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                for target in yaml_data.get('targets', []):
                    targets.append({
                        "label": target['id'],
                        "x": target['position_m'][0],
                        "y": target['position_m'][1],
                        "z": target['position_m'][2]
                    })
                print(f"📥 Using static targets from {config_path}")
                
    return intr, targets, output_dir

def prepare_video_and_json(output_dir, json_path, side_view):
    output_json_path = os.path.join(output_dir, "processed_subject_result.json")
    output_video_path = os.path.join(output_dir, "subject_annotated_video.mp4")
    video_path = os.path.join(output_dir, "Color")
    depth_video_path = os.path.join(output_dir, "Depth")
    if video_path.endswith(".mp4"):
        use_image_folder = False
        cap = cv2.VideoCapture(video_path)
        depth_cap = cv2.VideoCapture(depth_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        color_files = depth_files = None
        start_frame_idx = 0
    else:
        use_image_folder = True
        color_files = sorted([f for f in os.listdir(video_path) if f.endswith(".png")])
        depth_files = sorted([f for f in os.listdir(depth_video_path) if f.endswith(".raw")])
        first_img = cv2.imread(os.path.join(video_path, color_files[0]))
        height, width = first_img.shape[:2]
        
        # Get framerate from the masked video which has correct timing
        masked_video_path = os.path.join(output_dir, "masked_video.mp4")
        if os.path.exists(masked_video_path):
            temp_cap = cv2.VideoCapture(masked_video_path)
            fps = temp_cap.get(cv2.CAP_PROP_FPS)
            temp_cap.release()
            print(f"🎬 Using masked video framerate: {fps} FPS from {masked_video_path}")
        else:
            # Fallback: try to calculate from original video and sampling
            original_video_path = video_path.replace('/Color', '/Color.mp4')
            if os.path.exists(original_video_path):
                temp_cap = cv2.VideoCapture(original_video_path)
                original_fps = temp_cap.get(cv2.CAP_PROP_FPS)
                temp_cap.release()
                
                # Check if SAM2 processing applied frame sampling
                scale_metadata_path = os.path.join(output_dir, "sam2_scale_metadata.json")
                sampling_rate = 1
                if os.path.exists(scale_metadata_path):
                    with open(scale_metadata_path, 'r') as f:
                        metadata = json.load(f)
                        sampling_rate = metadata.get('processing_info', {}).get('sampling_rate', 1)
                
                # Adjust framerate based on sampling
                fps = original_fps / sampling_rate
                print(f"🎬 Fallback - Original: {original_fps} FPS, Sampling: {sampling_rate}x → Output: {fps} FPS")
            else:
                fps = 6  # Final fallback
                print(f"⚠️ Could not find any video reference, using fallback framerate: {fps} FPS")
        
        cap = depth_cap = None
        # Extract starting frame index from first filename
        if color_files:
            start_frame_idx = int(color_files[0].split("_")[-1].split(".")[0])
        else:
            start_frame_idx = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return (cap, depth_cap, out, width, height, fps, use_image_folder, color_files, depth_files, video_path, depth_video_path, output_json_path, json_data, start_frame_idx)

def select_valid_candidates(bboxes, bodyparts, confidences, width, height, side_view = False):
    valid_candidates = []
    for i, (bbox, keypoints) in enumerate(zip(bboxes, bodyparts)):
        if bbox == [-1.0] * 4:
            continue
        x, y, w, h = map(int, bbox)
        x_center = x + w//2
        y_center = y + h//2
        confidence = confidences[i]
        aspect_ratio = h / w if w > 0 else 0

        if side_view:
            if  y_center < 0.3 * height:
                continue
        else:
            if aspect_ratio > 2.0 or y < height // 2 or confidence < 0.98 or (x_center > 0.5 * width and y_center > 0.8 * height):
                continue
        valid_candidates.append((x, bbox, keypoints, confidence))
    return valid_candidates

def predict_bbox_with_optical_flow(prev_gray, frame_gray, prev_center, valid_candidates):
    if prev_gray is not None and prev_center is not None:
        prev_pts = np.array([[prev_center]], dtype=np.float32)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None)
        predicted_center = next_pts[0][0] if status[0] else None
    else:
        predicted_center = None
    if predicted_center is not None:
        def distance_to_pred(candidate):
            cx = candidate[1][0] + candidate[1][2] / 2
            cy = candidate[1][1] + candidate[1][3] / 2
            return np.linalg.norm([cx - predicted_center[0], cy - predicted_center[1]])
        valid_candidates.sort(key=distance_to_pred)
    else:
        valid_candidates.sort(key=lambda item: item[1][2] * item[1][3])
    return valid_candidates

def extract_trace_point(kp_cleaned, depth_frame, fx, fy, cx, cy, bbox=None):
    valid_points = []
    for name, (x_px, y_px) in kp_cleaned.items():
        if 0 <= x_px < depth_frame.shape[1] and 0 <= y_px < depth_frame.shape[0]:
            depth = get_valid_depth(depth_frame, x_px, y_px)
            if depth is not None and depth > 0:
                x_m = (x_px - cx) * depth / fx
                y_m = (y_px - cy) * depth / fy
                z_m = depth
                valid_points.append((x_m, y_m, z_m))
    if valid_points:
        avg_point = np.mean(valid_points, axis=0)
        return tuple(avg_point)
    # Fallback to center of bbox if no valid keypoints
    if bbox is not None:
        x_center_px = bbox[0] + bbox[2] / 2
        y_center_px = bbox[1] + bbox[3] / 2
        depth = get_valid_depth(depth_frame, x_center_px, y_center_px)
        if depth is not None and depth > 0:
            x_center_m = (x_center_px - cx) * depth / fx
            y_center_m = (y_center_px - cy) * depth / fy
            z_center_m = depth
            return x_center_m, y_center_m, z_center_m
    return 0, 0, 0

def compute_target_distances(x_center_m, y_center_m, z_center_m, targets, include_human=False):
    frame_target_metrics = {}
    for item in targets:
        label = item.get("label", "")
        if not include_human and label == 'human':
            continue
        tx, ty, tz = item["x"], item["y"], item["z"]
        dx = tx - x_center_m
        dy = ty - y_center_m
        dz = tz - z_center_m
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        theta = np.arccos(dz / r) if r > 0 else 0.0
        phi = np.arctan2(dy, dx)
        frame_target_metrics[f"{label}_r"] = r
        frame_target_metrics[f"{label}_theta"] = theta
        frame_target_metrics[f"{label}_phi"] = phi
    return frame_target_metrics

def interpolate_missing_frames(processed_json):
    """Interpolate missing frames using previous valid detection."""
    last_valid = None
    for i in range(len(processed_json)):
        if not processed_json[i]["bboxes"]:
            if last_valid:
                interpolated = copy.deepcopy(last_valid)
                # Preserve and increment frame_index and local_frame_index properly
                interpolated["frame_index"] = i + last_valid.get("frame_index", 0)
                interpolated["local_frame_index"] = i
                processed_json[i] = interpolated
        else:
            last_valid = processed_json[i]
    return processed_json

def interpolate_distance_metrics(metrics_list):
    if not metrics_list:
        return metrics_list
    keys = set(k for d in metrics_list if d for k in d.keys())
    for key in keys:
        values = [d.get(key, np.nan) if d else np.nan for d in metrics_list]
        for i in range(len(values)):
            if np.isnan(values[i]):
                # Only interpolate if both prev and next exist
                prev_idx = next((j for j in range(i - 1, -1, -1) if not np.isnan(values[j])), None)
                next_idx = next((j for j in range(i + 1, len(values)) if not np.isnan(values[j])), None)
                if prev_idx is not None and next_idx is not None:
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    interpolated = (1 - alpha) * values[prev_idx] + alpha * values[next_idx]
                    if metrics_list[i] is None or not metrics_list[i]:
                        metrics_list[i] = {}
                    metrics_list[i][key] = interpolated
    return metrics_list

def save_processed_results(processed_json, fps, output_json_path, width, height, depth_frame, intrinsics, SELECTED_NAMES, per_frame_target_metrics, trace_3d, dog = True):
    """Save processed results: JSON with time and CSV table."""
    # Add timestamp and frame_index directly during CSV row creation
    processed_json_with_time = processed_json

    # Interpolate missing frames
    processed_json_with_time = interpolate_missing_frames(processed_json_with_time)

    # Interpolate missing distances in per_frame_target_metrics (new logic)
    per_frame_target_metrics = interpolate_distance_metrics(per_frame_target_metrics)

    # === Save flattened CSV ===
    # Define leg groups for averaging
    if dog:
        leg_groups = {
            "left_front": ["left_front_elbow", "left_front_knee", "left_front_paw"],
            "right_front": ["right_front_elbow", "right_front_knee", "right_front_paw"],
            "left_back": ["left_back_elbow", "left_back_knee", "left_back_paw"],
            "right_back": ["right_back_elbow", "right_back_knee", "right_back_paw"],
        }
    else:
        leg_groups = {
            "left_front": ["left_shoulder", "left_elbow", "left_wrist"],
            "right_front": ["right_shoulder", "right_elbow", "right_wrist"],
            "left_back": ["left_hip", "left_knee", "left_ankle"],
            "right_back": ["right_hip", "right_knee", "right_ankle"],
        }
    # --- Orientation storage for interpolation ---
    head_orients = []
    torso_orients = []
    # --- Save all rows, but defer orientation assignment ---
    rows_tmp = []
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    for i, entry in enumerate(processed_json_with_time):
        if not entry["bboxes"] or not entry["bodyparts"]:
            continue
        bbox = entry["bboxes"][0]
        keypoints = entry["bodyparts"][0]
        direction = entry["subject_directions"][0] if entry["subject_directions"] else ['', '', '']
        conf = entry["bbox_scores"][0] if entry["bbox_scores"] else None
        frame_idx = entry.get("frame_index")
        # Include local_frame_index in the row
        local_frame_idx = entry.get("local_frame_index")
        time = round(local_frame_idx / fps , 3)
        # Start with time and frame index
        row = {
            "frame_index": frame_idx,
            "time_sec": time,
            "local_frame_index": local_frame_idx,
        }

        # # Add interpolation flags for bbox, keypoints, and distances
        # row["interpolated_bbox"] = entry.get("interpolated_bbox", 0)
        # row["interpolated_keypoints"] = entry.get("interpolated_keypoints", 0)
        # row["interpolated_distances"] = entry.get("interpolated_distances", 0)

        # Add per-frame distances immediately after frame index and time
        target_metrics = entry.get("target_metrics", per_frame_target_metrics[i] if i < len(per_frame_target_metrics) else {})
        if target_metrics:
            ordered_keys = sorted([k for k in target_metrics if k.endswith('_r')]) + \
                           sorted([k for k in target_metrics if k.endswith('_theta')]) + \
                           sorted([k for k in target_metrics if k.endswith('_phi')])
            for key in ordered_keys:
                row[key] = target_metrics[key]
            # Mark interpolated if any metric was not originally in entry["target_metrics"]
            if "target_metrics" not in entry or not any(k in entry["target_metrics"] for k in ordered_keys):
                row["interpolated_distances"] = 1

        # Continue building the row as before (add bbox, confidence, dog_dir, etc.)
        row.update({
            "bbox_x": int(bbox[0]), "bbox_y": int(bbox[1]), "bbox_w": (bbox[2]), "bbox_h": int(bbox[3]),
            "confidence": conf,
            "dog_dir": direction if direction is not None else ''
        })

        # Compute head_orientation and torso_orientation vectors in pixel coordinates
        kp_cleaned_for_orient = {}
        for joint, idx in SELECTED_NAMES.items():
            if idx < len(keypoints):
                x_px, y_px, conf_kp = keypoints[idx][:3]
                kp_cleaned_for_orient[joint] = np.array([x_px, y_px])
            else:
                kp_cleaned_for_orient[joint] = None

        # --- 3D keypoints in meters for all named joints (for orientation logic) ---
        joint_coords = {}
        for joint, idx in SELECTED_NAMES.items():
            if idx < len(keypoints):
                x_px, y_px, conf_kp = keypoints[idx][:3]
                if 0 <= int(y_px) < height and 0 <= int(x_px) < width:
                    z = depth_frame[int(y_px), int(x_px)]
                    x_m = (x_px - cx) * z / fx
                    y_m = (y_px - cy) * z / fy
                    z_m = z
                else:
                    x_m, y_m, z_m = None, None, None
                joint_coords[joint] = {
                    "x": x_px,
                    "y": y_px,
                    "conf": conf_kp,
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": z_m
                }
            else:
                joint_coords[joint] = {
                    "x": None,
                    "y": None,
                    "conf": None,
                    "x_m": None,
                    "y_m": None,
                    "z_m": None
                }

        # Store individual joint values for all non-leg keypoints
        for joint, jc in joint_coords.items():
            if joint in leg_groups["left_front"] + leg_groups["right_front"] + leg_groups["left_back"] + leg_groups["right_back"]:
                continue  # skip leg joints individually
            row[f"{joint}_x"] = jc["x"]
            row[f"{joint}_y"] = jc["y"]
            row[f"{joint}_conf"] = jc["conf"]
            row[f"{joint}_x_m"] = jc["x_m"]
            row[f"{joint}_y_m"] = jc["y_m"]
            row[f"{joint}_z_m"] = jc["z_m"]

        # Save only average keypoints for each leg group
        for leg, joints in leg_groups.items():
            xs, ys, confs, xs_m, ys_m, zs_m = [], [], [], [], [], []
            for j in joints:
                jc = joint_coords[j]
                if jc["conf"] is not None and jc["conf"] > 0.1 and jc["x"] is not None and jc["y"] is not None:
                    xs.append(jc["x"])
                    ys.append(jc["y"])
                    confs.append(jc["conf"])
                    if jc["x_m"] is not None and jc["y_m"] is not None and jc["z_m"] is not None:
                        xs_m.append(jc["x_m"])
                        ys_m.append(jc["y_m"])
                        zs_m.append(jc["z_m"])
            if xs:
                row[f"{leg}_x"] = float(np.mean(xs))
                row[f"{leg}_y"] = float(np.mean(ys))
                row[f"{leg}_conf"] = float(np.mean(confs))
            else:
                row[f"{leg}_x"] = None
                row[f"{leg}_y"] = None
                row[f"{leg}_conf"] = None
            if xs_m:
                row[f"{leg}_x_m"] = float(np.mean(xs_m))
                row[f"{leg}_y_m"] = float(np.mean(ys_m))
                row[f"{leg}_z_m"] = float(np.mean(zs_m))
            else:
                row[f"{leg}_x_m"] = None
                row[f"{leg}_y_m"] = None
                row[f"{leg}_z_m"] = None

        # --- Robust 3D orientation computation using local patch depth sampling ---
        head_orientation_3d = [None, None, None]
        torso_orientation_3d = [None, None, None]
        nose_px = joint_coords["nose"]
        if dog:

            tailbase_px = joint_coords["tail_base"]
        else:
            if joint_coords["left_hip"]['x'] is not None:
                tailbase_px = joint_coords["left_hip"]
            else:
                tailbase_px = joint_coords["right_hip"]

        # Choose human or dog relevant joints
        if dog:
            alt_withers_keys = ["withers", "throat", "neck"]
        else:
            alt_withers_keys = ["left_shoulder", "right_shoulder"]  # for human torso direction

        for name in alt_withers_keys:
            if name in joint_coords and joint_coords[name]["x"] is not None:
                withers_px = joint_coords[name]
                break
        else:
            withers_px = None

        if nose_px["x"] is not None and withers_px:
            d1 = get_valid_depth(depth_frame, nose_px["x"], nose_px["y"])
            d2 = get_valid_depth(depth_frame, withers_px["x"], withers_px["y"])
            if d1 and d2:
                nose_3d = pixel_to_3d(nose_px["x"], nose_px["y"], d1, fx, fy, cx, cy)
                withers_3d = pixel_to_3d(withers_px["x"], withers_px["y"], d2, fx, fy, cx, cy)
                head_orientation_3d = (nose_3d - withers_3d).tolist()

        if withers_px and tailbase_px["x"] is not None:
            d1 = get_valid_depth(depth_frame, withers_px["x"], withers_px["y"])
            d2 = get_valid_depth(depth_frame, tailbase_px["x"], tailbase_px["y"])
            if d1 and d2:
                withers_3d = pixel_to_3d(withers_px["x"], withers_px["y"], d1, fx, fy, cx, cy)
                tail_3d = pixel_to_3d(tailbase_px["x"], tailbase_px["y"], d2, fx, fy, cx, cy)
                torso_orientation_3d = (withers_3d - tail_3d).tolist()

        head_orients.append(head_orientation_3d)
        torso_orients.append(torso_orientation_3d)

        # Orientation in pixel coordinates
        head_orientation = [None, None]
        torso_orientation = [None, None]
        if dog:
            # Head orientation: nose to withers/throat/neck
            if kp_cleaned_for_orient.get("nose") is not None:
                for alt_torso in ["withers", "throat", "neck"]:
                    if kp_cleaned_for_orient.get(alt_torso) is not None:
                        head_orientation = (kp_cleaned_for_orient["nose"] - kp_cleaned_for_orient[alt_torso]).tolist()
                        break
            # Torso orientation: withers/throat/neck to tail_base
            if (kp_cleaned_for_orient.get("withers") is not None or kp_cleaned_for_orient.get("throat") is not None or kp_cleaned_for_orient.get("neck") is not None) and kp_cleaned_for_orient.get("tail_base") is not None:
                for alt_torso in ["withers", "throat", "neck"]:
                    if kp_cleaned_for_orient.get(alt_torso) is not None:
                        torso_orientation = (kp_cleaned_for_orient[alt_torso] - kp_cleaned_for_orient["tail_base"]).tolist()
                        break
        else:
            # Human head orientation: left eye to right eye
            if kp_cleaned_for_orient.get("left_eye") is not None and kp_cleaned_for_orient.get("right_eye") is not None:
                head_orientation = (kp_cleaned_for_orient["left_eye"] - kp_cleaned_for_orient["right_eye"]).tolist()
            # Human torso orientation: left shoulder to right shoulder
            if kp_cleaned_for_orient.get("left_shoulder") is not None and kp_cleaned_for_orient.get("right_shoulder") is not None:
                torso_orientation = (kp_cleaned_for_orient["left_shoulder"] - kp_cleaned_for_orient["right_shoulder"]).tolist()

        row["head_orientation_x"], row["head_orientation_y"] = head_orientation
        row["torso_orientation_x"], row["torso_orientation_y"] = torso_orientation

        # Remove keys where all values are None or empty lists, safely handling types
        row = {k: v for k, v in row.items()
               if not (
                   v is None or
                   (isinstance(v, list) and all(item is None for item in v)) or
                   (isinstance(v, np.ndarray) and np.all(v == None))
               )}
        rows_tmp.append(row)

    # --- Interpolate missing orientations ---
    def interpolate_vectors(vectors):
        vectors = [np.array(v) if v is not None else None for v in vectors]
        for i in range(len(vectors)):
            v = vectors[i]
            if v is None or any(val is None for val in v) or np.allclose(v, [0, 0, 0]):
                # Find previous and next valid
                prev_idx = next((j for j in range(i-1, -1, -1)
                                 if vectors[j] is not None and all(val is not None for val in vectors[j]) and not np.allclose(vectors[j], [0, 0, 0])), None)
                next_idx = next((j for j in range(i+1, len(vectors))
                                 if vectors[j] is not None and all(val is not None for val in vectors[j]) and not np.allclose(vectors[j], [0, 0, 0])), None)
                if prev_idx is not None and next_idx is not None:
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    vectors[i] = (1 - alpha) * np.array(vectors[prev_idx]) + alpha * np.array(vectors[next_idx])
                elif prev_idx is not None:
                    vectors[i] = np.array(vectors[prev_idx])
                elif next_idx is not None:
                    vectors[i] = np.array(vectors[next_idx])
                else:
                    vectors[i] = np.array([None, None, None])
        # Convert to lists for DataFrame assignment
        return [v.tolist() if v is not None else [None, None, None] for v in vectors]

    head_orients = interpolate_vectors(head_orients)
    torso_orients = interpolate_vectors(torso_orients)

    # --- Assign interpolated orientations to rows ---
    records = []
    for i, row in enumerate(rows_tmp):
        row["head_orientation_x_m"], row["head_orientation_y_m"], row["head_orientation_z_m"] = head_orients[i]
        row["torso_orientation_x_m"], row["torso_orientation_y_m"], row["torso_orientation_z_m"] = torso_orients[i]
        # Save the trace_3d coordinates if available
        if i < len(trace_3d):
            row["trace3d_x"], row["trace3d_y"], row["trace3d_z"] = trace_3d[i]
        
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = output_json_path.replace(".json", "_table.csv")
    df.to_csv(csv_path, index=False)

def plot_trace_and_targets(trace_3d, targets, output_json_path, fps, side_view ):
    """Plot 3D trace and targets."""
    from matplotlib import cm
    import matplotlib.pyplot as plt
    trace_3d = np.array(trace_3d)
    t = np.linspace(0, 1, len(trace_3d))
    colors = cm.rainbow(t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Compute a smoothed gray line using a simple moving average over every 3 points
    if len(trace_3d) >= 3:
        smoothed_trace = []
        for j in range(1, len(trace_3d)-1):
            prev_pt = trace_3d[j-1]
            curr_pt = trace_3d[j]
            next_pt = trace_3d[j+1]
            weighted_avg = (0.25 * prev_pt + 0.5 * curr_pt + 0.25 * next_pt)
            smoothed_trace.append(weighted_avg)
        smoothed_trace = np.array(smoothed_trace)
        for idx in range(1, len(smoothed_trace)):
            ax.plot(
                smoothed_trace[idx-1:idx+1, 0],
                smoothed_trace[idx-1:idx+1, 2],
                smoothed_trace[idx-1:idx+1, 1],
                color=colors[idx], alpha=0.6
            )
    for i in range(1, len(trace_3d)):
        ax.plot(trace_3d[i-1:i+1, 0], trace_3d[i-1:i+1, 2], trace_3d[i-1:i+1, 1], '.', color=colors[i])
    for item in targets:
        from itertools import combinations
        for a, b in combinations(targets, 2):
            dx = a['x'] - b['x']
            dy = a['y'] - b['y']
            dz = a['z'] - b['z']
            d = np.sqrt(dx**2 + dy**2 + dz**2)
        x, y, z = item["x"], item["y"], item["z"]
        label = item.get("label", "")
        if label == "human":
            ax.scatter(x, z, y, c='blue', s=100, marker='x', label='Human')
        else:
            plot_cube(ax, center=(x, z, y), size=0.1, color='red', alpha=0.6)
        ax.text(x, z, y + 0.05, label, color='black', fontsize=8)
    ax.set_title("Subject Trace in 3D (X: width, Y: height, Z: depth in meters)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Depth (meters)")
    ax.set_zlabel("Y (meters)")
    if not side_view:
        ax.set_xlim([-1, 2])
        ax.set_ylim([2.0, 4])
        ax.set_zlim([0, 2])
        ax.view_init(elev=45, azim=135)
    else:

        ax.view_init(elev=-135, azim=20)  # Side-on view
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=cm.rainbow(0.0), lw=2, label='Start (Red)'),
        Line2D([0], [0], color=cm.rainbow(1.0), lw=2, label='End (Violet)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    def set_axes_equal(ax):
        """Set equal scaling for all axes in a 3D plot."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max(x_range, y_range, z_range)

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    set_axes_equal(ax)
    trace_path = output_json_path.replace(".json", "_trace3d.png")
    plt.savefig(trace_path)
    plt.close()
    print(f"3D trace saved to: {trace_path}")
    

    # === Add a 2D X-Z plot ===
    fig2, ax2 = plt.subplots()
    for i in range(1, len(trace_3d)):
        ax2.plot(trace_3d[i-1:i+1, 0], trace_3d[i-1:i+1, 2], '.', color=colors[i], alpha=0.8)
    for item in targets:
        x, z = item["x"], item["z"]
        label = item.get("label", "")
        if label == "human":
            ax2.scatter(x, z, c='blue', s=80, marker='x', label='Human')
        else:
            ax2.scatter(x, z, c='gray', s=80, label=label)
        ax2.text(x, z + 0.05, label, color='black', fontsize=8)
    ax2.set_title("Subject Trace in 2D (X-Z Plane)")
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Depth Z (meters)")
    ax2.grid(True)

    # Save figure
    trace_plot_path = output_json_path.replace(".json", "_trace.png")
    fig2.savefig(trace_plot_path)
    plt.close()
    print(f"✅ Saved 2D trace plot to: {trace_plot_path}")

def plot_distance_comparison(per_frame_target_metrics, targets, output_json_path, fps):
    """Plot distance to each target over time."""
    if per_frame_target_metrics:
        times = np.linspace(0, len(per_frame_target_metrics) / fps , len(per_frame_target_metrics))
        fig, ax = plt.subplots()
        for item in targets:
            label = item.get("label", "")
            dists = [frame.get(f"{label}_r", np.nan) if frame else np.nan for frame in per_frame_target_metrics]
            if label == 'human':
                ax.plot(times, dists, label='human', color='gray', linestyle='--')
            else:
                ax.plot(times, dists, label=label)
        ax.set_xlabel('Frames (idx)')
        ax.set_ylabel('Distance to Target (meters)')
        ax.set_title('subject Distance to Each Target Over Time')
        ax.legend()
        plt.grid(True)
        dist_plot_path = output_json_path.replace(".json", "_distance_comparison.png")
        plt.savefig(dist_plot_path)
        plt.close()
        print(f"Saved distance comparison plot to: {dist_plot_path}")


def pose_visualize(json_path, side_view = False, dog= True):
    prev_gray = None
    prev_center = None
    smoothed_bbox = None

    intr, targets, output_dir = load_intrinsics_and_targets(json_path, side_view)
    
    # Load scale information from segmentation step
    scale_info = load_scale_info(output_dir)
    if scale_info:
        print(f"📏 Loaded scale metadata: {scale_info['processing_info']}")
    else:
        print("⚠️ No scale metadata found - assuming no resize correction needed")
    (cap, depth_cap, out, width, height, fps, use_image_folder, color_files, depth_files, video_path, depth_video_path, output_json_path, json_data, start_frame_idx) = prepare_video_and_json(output_dir, json_path, side_view)
    
    # Validate image dimensions match intrinsics
    expected_width, expected_height = 1280, 720
    if width != expected_width or height != expected_height:
        print(f"⚠️ WARNING: Image dimensions {width}x{height} don't match expected {expected_width}x{expected_height}")
        print("📐 Consider running image standardization first")
    else:
        print(f"✅ Image dimensions {width}x{height} match camera intrinsics")

    # --- Resample JSON to match number of image frames if needed ---
    if use_image_folder:
        num_images = len(color_files)
        num_json = len(json_data)
        print(f"🔍 JSON Debug: type={type(json_data)}, len={num_json}, images={num_images}")
        
        if num_json != num_images:
            print(f"Resampling JSON data: {num_json} entries → {num_images} frames")
            if num_json > 0 and isinstance(json_data, list):
                # Create safe indices that don't exceed bounds
                indices = np.linspace(0, num_json - 1, num=min(num_images, num_json)).astype(int)
                # Ensure indices are within bounds
                indices = np.clip(indices, 0, num_json - 1)
                print(f"🔍 Index range: {indices.min()} to {indices.max()}")
                
                if len(indices) < num_images:
                    # If we need more frames than available, repeat the last frame
                    try:
                        last_frame = json_data[-1]
                        print(f"🔍 Got last frame: {type(last_frame)}")
                    except Exception as e:
                        print(f"❌ Error getting last frame: {e}")
                        last_frame = {"bodyparts": [], "bboxes": [], "bbox_scores": []}
                    
                    resampled_data = [json_data[i] for i in indices]
                    # Pad with copies of the last frame
                    while len(resampled_data) < num_images:
                        import copy
                        resampled_data.append(copy.deepcopy(last_frame))
                    json_data = resampled_data
                else:
                    json_data = [json_data[i] for i in indices[:num_images]]
                print(f"✅ Resampled to {len(json_data)} frames")
            else:
                print("⚠️ No JSON data to resample or wrong type")
                # Create empty frames
                json_data = [{"bodyparts": [], "bboxes": [], "bbox_scores": []} for _ in range(num_images)]

    transform_matrix = load_side_to_front_transform() if side_view else np.eye(4)
    with open("./config/skeleton_config.json", "r") as f:
        skeleton_data = json.load(f)
    if dog:
        SELECTED_NAMES = skeleton_data["dog"]["SELECTED_NAMES"]
        SKELETON = skeleton_data["dog"]["SKELETON"] 
    else:
        SELECTED_NAMES = skeleton_data["human"]["SELECTED_NAMES"]
        SKELETON = skeleton_data["human"]["SKELETON"]

    processed_json = []
    trace_3d = []
    last_valid = None
    frame_idx = 0  # Start at 0 instead of -1
    max_deviation = 200
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    per_frame_target_metrics = []
    missing_frame_count = 0
    last_color_frame = None
    last_depth_frame = None
    
    print(f"🎬 Starting frame processing: {len(json_data)} JSON entries, {len(color_files) if use_image_folder else 'video'} frames")
    
    while True:
        if use_image_folder:
            if frame_idx >= len(color_files) or frame_idx >= len(json_data):
                break
            frame = cv2.imread(os.path.join(video_path, color_files[frame_idx]))
            raw_path = os.path.join(depth_video_path, depth_files[frame_idx])
            with open(raw_path, 'rb') as f:
                raw = np.frombuffer(f.read(), dtype=np.uint16).reshape((height, width))
                depth_frame = raw / 1000
            ret = True
            ret_d = True
            
            # Debug frame synchronization
            if frame_idx < 5:  # Only for first few frames
                print(f"📹 Frame {frame_idx}: Loading {color_files[frame_idx]} with JSON entry {frame_idx}")
        else:
            ret, frame = cap.read()
            ret_d, depth_frame = depth_cap.read()

            if not ret:
                frame = last_color_frame.copy() if last_color_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
            if not ret_d:
                depth_frame = last_depth_frame.copy() if last_depth_frame is not None else np.zeros((height, width), dtype=np.uint8)

            if frame_idx >= len(json_data):
                break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        entry = json_data[frame_idx]
        bboxes = entry.get("bboxes", [])
        bodyparts = entry.get("bodyparts", [])
        confidences = entry.get("bbox_scores", [1.0] * len(bboxes))
        
        # Debug bbox alignment for first few frames
        if frame_idx < 5 and bboxes:
            print(f"🎯 Frame {frame_idx}: JSON bbox {bboxes[0][:2]} (center: {bboxes[0][0] + bboxes[0][2]/2:.1f}, {bboxes[0][1] + bboxes[0][3]/2:.1f})")

        # else:
        valid_candidates = select_valid_candidates(bboxes, bodyparts, confidences, width, height, side_view)
        if not valid_candidates:
            # Always append a NaN trace point and empty metrics, even if detection fails
            trace_3d.append([np.nan, np.nan, np.nan])
            per_frame_target_metrics.append({})
            # Detection started, but now missing detection → use last_valid
            if last_valid and last_valid["bboxes"]:
                interpolated_bbox = last_valid["bboxes"][0]
                interpolated_keypoints = last_valid["bodyparts"][0]
                interpolated_conf = last_valid["bbox_scores"][0]
                processed_json.append({
                    "frame_index": frame_idx + start_frame_idx,
                    "local_frame_index": frame_idx,
                    "bboxes": [interpolated_bbox],
                    "bodyparts": [interpolated_keypoints],
                    "subject_directions": last_valid.get("dog_directions", []),
                    "bbox_scores": [interpolated_conf]
                })
                # Draw interpolated box
                x, y, w, h = map(int, interpolated_bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, 'Interpolated', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                processed_json.append({
                    "frame_index": frame_idx + start_frame_idx,
                    "local_frame_index": frame_idx,
                    "bboxes": [],
                    "bodyparts": [],
                    "subject_directions": [],
                    "bbox_scores": []
                })
            # Fallback: use last_color_frame if available for output
            frame = last_color_frame.copy() if 'last_color_frame' in locals() and last_color_frame is not None else frame
            out.write(frame)
            # Increment missing frame counter
            missing_frame_count += 1
            if side_view and missing_frame_count >= 5:
                print(f"Side view: detection stopped after {missing_frame_count} consecutive missing frames at frame {frame_idx}")
            frame_idx += 1
            continue
        
        # Found a valid candidate, reset missing frame counter
        missing_frame_count = 0
        valid_candidates = predict_bbox_with_optical_flow(prev_gray, frame_gray, prev_center, valid_candidates)
        _, bbox, keypoints, conf = valid_candidates[0]

        candidate_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        if smoothed_bbox is not None and not side_view:
            smoothed_center = (smoothed_bbox[0] + smoothed_bbox[2] / 2, smoothed_bbox[1] + smoothed_bbox[3] / 2)
            deviation = np.linalg.norm(np.array(candidate_center) - np.array(smoothed_center))
            if deviation > max_deviation:
                print(f"Skipping frame {frame_idx} due to large bbox deviation: {deviation:.2f}")
                # Always append a NaN trace point and empty metrics, even if skipping due to deviation
                trace_3d.append([np.nan, np.nan, np.nan])
                per_frame_target_metrics.append({})
                frame_idx += 1
                continue
        if smoothed_bbox is None:
            smoothed_bbox = bbox
        else:
            smoothed_bbox = smooth_bbox(smoothed_bbox, bbox)

        x, y, w, h = map(int, smoothed_bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        kp = np.array(keypoints)
        kp_cleaned = {}
        for name, idx in SELECTED_NAMES.items():
            
            if idx < len(kp) and kp[idx][2] > 0.1:
                pt = tuple(np.round(kp[idx][:2]).astype(int))
                kp_cleaned[name] = pt
                cv2.circle(frame, pt, 4, (0, 255, 0), -1)
        for a, b in SKELETON:
            if a in kp_cleaned and b in kp_cleaned:
                cv2.line(frame, kp_cleaned[a], kp_cleaned[b], (0, 255, 255), 2)
        if "tail_base" in kp_cleaned and "neck" in kp_cleaned:
            vec = np.array(kp_cleaned["neck"]) - np.array(kp_cleaned["tail_base"])
            norm = np.linalg.norm(vec)
            if norm > 0:
                dog_dir = (vec / norm).tolist()
                cv2.arrowedLine(frame, kp_cleaned["tail_base"], kp_cleaned["neck"], (255, 0, 255), 2)
            else:
                dog_dir = None
        else:
            dog_dir = None

        x_center_m, y_center_m, z_center_m = extract_trace_point(kp_cleaned, depth_frame, fx, fy, cx, cy, smoothed_bbox)
        if (x_center_m, y_center_m, z_center_m) != (0, 0, 0):
            if not side_view:
                # uncomment for transformation to align with front camera frame
                point_cam = np.array([x_center_m, y_center_m, z_center_m, 1.0])
                point_transformed = transform_matrix @ point_cam
                trace_3d.append(point_transformed[:3])
                frame_target_metrics = compute_target_distances(*point_transformed[:3], targets)
            trace_3d.append([x_center_m, y_center_m, z_center_m])
            frame_target_metrics = compute_target_distances(x_center_m, y_center_m, z_center_m, targets, include_human=True)
            distances = [v for k, v in frame_target_metrics.items() if k.endswith('_r')]
            if distances:
                min_dist = min(distances)
                # print(f"Frame {frame_idx}: {' | '.join(f'{d:.2f}' for d in distances)}, Minimum Euclidean distance to a target = {min_dist:.3f} meters")
            per_frame_target_metrics.append(frame_target_metrics)
        else:
            # Append NaNs to keep frame alignment
            trace_3d.append([np.nan, np.nan, np.nan])
            per_frame_target_metrics.append({})

        # Update last_valid before appending
        last_valid = {
            "frame_index": frame_idx + start_frame_idx,
            "local_frame_index": frame_idx,
            "bboxes": [smoothed_bbox],
            "bodyparts": [[kp[idx].tolist() if isinstance(kp[idx], np.ndarray) else kp[idx] for name, idx in SELECTED_NAMES.items()]],
            "subject_directions": [dog_dir],
            "bbox_scores": [conf],
            "target_metrics": frame_target_metrics if (x_center_m, y_center_m, z_center_m) != (0, 0, 0) else {},
            "interpolated_bbox": 0,
            "interpolated_keypoints": 0,
            "interpolated_distances": 0,
        }
        processed_json.append(last_valid)
        out.write(frame)
        prev_gray = frame_gray
        prev_center = (smoothed_bbox[0] + smoothed_bbox[2] / 2, smoothed_bbox[1] + smoothed_bbox[3] / 2)
        frame_idx += 1

    cap.release() if not use_image_folder else None
    depth_cap.release() if not use_image_folder else None
    out.release()
    cv2.destroyAllWindows()
    print(f"FRAME INDEX TOTAL LENGTH ------------> {frame_idx}")
    
    processed_json = interpolate_missing_frames(processed_json)
    trace_3d = interpolate_trace(trace_3d)
    # Reassign interpolated target_metrics back to processed_json
    for i in range(len(processed_json)):
        if i < len(per_frame_target_metrics):
            processed_json[i]["target_metrics"] = per_frame_target_metrics[i]
    save_processed_results(processed_json, fps, output_json_path, width, height, depth_frame, intr, SELECTED_NAMES, per_frame_target_metrics, trace_3d, dog=dog)
    plot_trace_and_targets(trace_3d, targets, output_json_path, fps, side_view)
    plot_distance_comparison(per_frame_target_metrics, targets, output_json_path, fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--side_view', action='store_true', help='Disable filtering for side view camera input')
    # parser.add_argument('--color_video', type=str, required=True)
    # parser.add_argument('--depth_video', type=str, required=True)
    try:
        args = parser.parse_args()
        json_path = args.json_path 
    except:
        json_path = "/home/xhe71/Desktop/dog_data/baby/CCD0442_side/1/masked_video_skeleton.json"
    
    # video_path = args.video_path
    # depth_video_path = args.depth_video
    json_path = "/home/xhe71/Desktop/dog_data/baby/CCD0442_side/1/masked_video_skeleton.json"
    # video_path = "/home/xhe71/Desktop/dog_data/BDL204_Waffle/2/Color.mp4"
    # depth_video_path = "/home/xhe71/Desktop/dog_data/BDL204_Waffle/2/Depth.mp4"
    pose_visualize(json_path, side_view = True, dog = False)


if __name__ == "__main__":
    main()