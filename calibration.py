import json
import numpy as np
import cv2
import pyrealsense2 as rs

world_coords_dict = {
    "target_1": [-1.05, 0.0, 1.52],
    "target_2": [-0.39, 0.0, 1.79],
    "target_3": [0.39, 0.0, 1.82],
    "target_4": [1.05, 0.0, 1.52]
}

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


def calibrate_camera_to_world(detected_json_path, world_coords_dict, save_path):
    with open(detected_json_path, 'r') as f:
        targets = json.load(f)

    cam_pts = []
    world_pts = []
    for item in targets:
        label = item.get("label")
        if label in world_coords_dict:
            cam_pts.append([item["x"], item["y"], item["z"]])
            world_pts.append(world_coords_dict[label])

    if len(cam_pts) < 3:
        print("Need at least 3 matching points for calibration.")
        return

    cam_pts = np.array(cam_pts, dtype=np.float64).T  # shape (3, N)
    world_pts = np.array(world_pts, dtype=np.float64).T

    # Compute rigid transformation using Umeyama method
    mu_cam = np.mean(cam_pts, axis=1, keepdims=True)
    mu_world = np.mean(world_pts, axis=1, keepdims=True)

    X = cam_pts - mu_cam
    Y = world_pts - mu_world

    U, _, Vt = np.linalg.svd(Y @ X.T)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = mu_world - R @ mu_cam

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()

    with open(save_path, 'w') as f:
        json.dump({"transform": T.tolist()}, f, indent=2)
    print(f"Saved camera-to-world transform to {save_path}")

def calibrate_side_to_front(from_json_path, to_json_path, save_path):
    """Compute transform that maps side camera coordinates into front camera coordinate frame."""
    with open(from_json_path, 'r') as f:
        from_targets = json.load(f)
    with open(to_json_path, 'r') as f:
        to_targets = json.load(f)

    from_pts = []
    to_pts = []
    for f_item in from_targets:
        label = f_item.get("label")
        for t_item in to_targets:
            if t_item.get("label") == label:
                from_pts.append([f_item["x"], f_item["y"], f_item["z"]])
                to_pts.append([t_item["x"], t_item["y"], t_item["z"]])
                break

    if len(from_pts) < 3:
        print("Need at least 3 matching points for calibration.")
        return

    from_pts = np.array(from_pts, dtype=np.float64).T
    to_pts = np.array(to_pts, dtype=np.float64).T

    mu_from = np.mean(from_pts, axis=1, keepdims=True)
    mu_to = np.mean(to_pts, axis=1, keepdims=True)

    X = from_pts - mu_from
    Y = to_pts - mu_to

    U, _, Vt = np.linalg.svd(Y @ X.T)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = mu_to - R @ mu_from

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()

    with open(save_path, 'w') as f:
        json.dump({"transform": T.tolist()}, f, indent=2)
    print(f"Saved side-to-front transform to {save_path}")

print("--front camera--") 
rosbag_front_path = "dog_data/BDL226_Taro_front/BDL232_Taro_PVP_14.bag"

front_intrinsics = find_realsense_intrinsics(rosbag_front_path)
print("--side camera--")
rosbag_side_path = "dog_data/BDL225_Taro_side/BDL232_Taro_PVP_14_Cam2.bag"
side_intrinsics = find_realsense_intrinsics(rosbag_side_path)

front_targets_json_path = "target_coords_front.json"
side_targets_json_path = "target_coords_side.json"

# Run calibration for front and side
calibrate_camera_to_world(front_targets_json_path, world_coords_dict, "front_cam_to_world.json")
calibrate_side_to_front(side_targets_json_path, front_targets_json_path, "side_to_front_transform.json")