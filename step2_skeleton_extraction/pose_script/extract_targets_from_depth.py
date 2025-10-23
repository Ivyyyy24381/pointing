import pyrealsense2 as rs
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import mediapipe as mp

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

import cv2
import numpy as np
import json

def load_raw_depth(path, shape=(720, 1280)):
    with open(path, 'rb') as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint16).reshape(shape)

def project_to_3d(x, y, depth, intrinsics):
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    z = depth
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    return X, Y, z

def analyze_bboxes(color_path, depth_path, bboxes, intrinsics, output_json):
    color_img = cv2.imread(color_path)
    h, w = color_img.shape[:2]
    depth_img = load_raw_depth(depth_path, shape=(h, w))

    results = []
    id = 0
    for (x1, y1, x2, y2) in bboxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        roi = depth_img[y1:y2, x1:x2]
        valid = roi[roi > 0]
        id += 1
        if valid.size == 0:
            continue
        avg_depth = np.mean(valid) / 1000.0  # convert mm to meters

        cx_px = (x1 + x2) // 2
        cy_px = (y1 + y2) // 2
        X, Y, Z = project_to_3d(cx_px, cy_px, avg_depth, intrinsics)

        results.append({
            "bbox": [x1, y1, x2, y2],
            "center_px": [cx_px, cy_px],
            "avg_depth_m": round(avg_depth, 3),
            "x": round(X, 3),
            "y": round(Y, 3),
            "z": round(Z, 3),
            "label": f'target_{id}'
        })

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} entries to {output_json}")

def select_bounding_boxes(image_path, depth_png_path=None):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if depth_png_path:
        depth_img = cv2.imread(depth_png_path, cv2.IMREAD_UNCHANGED)
    else:
        depth_img = None
    bboxes = []

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        bboxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        if depth_img is not None:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(image_rgb)
            axs[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none'))
            axs[0].set_title("Color Image")
            axs[1].imshow(depth_img, cmap='gray')
            axs[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none'))
            axs[1].set_title("Depth Image")
            plt.show()

    fig, ax = plt.subplots(1)
    ax.imshow(image_rgb)
    toggle_selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5, spancoords='pixels',
        interactive=True
    )

    plt.title("Draw bounding boxes and close window when done")
    plt.show()

    return bboxes


# ---- Visualization of 3D targets ----
def visualize_3d_targets(results_json, save_plane_info_path="plane_info.json"):
    import json
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    with open(results_json, 'r') as f:
        data = json.load(f)

    targets = [item for item in data if item.get("label") != "human"]
    humans = [item for item in data if item.get("label") == "human"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for item in humans + targets:
        x, y, z = item["x"], item["y"], item["z"]
        color = 'b' if item.get("label") == "human" else 'r'
        ax.scatter(x, y, z, c=color, marker='o')
        ax.text(x, y, z, f'{item["center_px"]}', fontsize=8)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Target Visualization with Plane")

    # Fit and plot plane using first 4 non-human targets
    plane_targets = targets[:4]
    if len(plane_targets) >= 3:
        Xs = np.array([[t["x"], t["y"]] for t in plane_targets])
        Zs = np.array([t["z"] for t in plane_targets])
        model = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
        model.fit(Xs, Zs)

        x_vals = np.linspace(min(Xs[:, 0]), max(Xs[:, 0]), 10)
        y_vals = np.linspace(min(Xs[:, 1]), max(Xs[:, 1]), 10)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z_grid = model.predict(np.c_[X_grid.ravel(), Y_grid.ravel()]).reshape(X_grid.shape)

        ax.plot_surface(X_grid, Y_grid, Z_grid, color='cyan', alpha=0.5)

        # Save plane coefficients
        plane_info = {
            "model": "z = a*x + b*y + c",
            "coefficients": model.named_steps["ransacregressor"].estimator_.coef_.tolist(),
            "intercept": model.named_steps["ransacregressor"].estimator_.intercept_
        }
        with open(save_plane_info_path, 'w') as pf:
            json.dump(plane_info, pf, indent=2)
        print(f"Saved plane coefficients to {save_plane_info_path}")

    plt.show()


def estimate_human_center_mediapipe(color_path, depth_img, intrinsics, output_json, label="human"):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    image = cv2.imread(color_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("No human detected.")
        return

    landmarks = results.pose_landmarks.landmark
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    cx_px = int((left_hip.x + right_hip.x) * 0.5 * image.shape[1])
    cy_px = int((left_hip.y + right_hip.y) * 0.5 * image.shape[0])

    patch_size = 10
    x1 = max(cx_px - patch_size, 0)
    x2 = min(cx_px + patch_size, depth_img.shape[1])
    y1 = max(cy_px - patch_size, 0)
    y2 = min(cy_px + patch_size, depth_img.shape[0])

    roi = depth_img[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size == 0:
        print("No valid depth at hip center.")
        return
    avg_depth = np.mean(valid) / 1000.0

    X, Y, Z = project_to_3d(cx_px, cy_px, avg_depth, intrinsics)

    with open(output_json, 'r') as f:
        data = json.load(f)

    data.append({
        "bbox": [x1, y1, x2, y2],
        "center_px": [cx_px, cy_px],
        "avg_depth_m": round(avg_depth, 3),
        "x": round(X, 3),
        "y": round(Y, 3),
        "z": round(Z, 3),
        "label": label
    })

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)

def fit_and_plot_plane_from_targets(results_json, save_path="plane_visualization.png"):
    import json
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    with open(results_json, 'r') as f:
        data = json.load(f)

    # Use only the first 4 non-human targets
    targets = [item for item in data if item.get("label") != "human"][:4]
    if len(targets) < 3:
        print("Need at least 3 non-human points to fit a plane.")
        return

    Xs = np.array([[t["x"], t["y"]] for t in targets])
    Zs = np.array([t["z"] for t in targets])

    model = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
    model.fit(Xs, Zs)

    # Create meshgrid
    x_vals = np.linspace(min(Xs[:, 0]), max(Xs[:, 0]), 10)
    y_vals = np.linspace(min(Xs[:, 1]), max(Xs[:, 1]), 10)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_grid = model.predict(np.c_[X_grid.ravel(), Y_grid.ravel()]).reshape(X_grid.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t in targets:
        ax.scatter(t["x"], t["y"], t["z"], c='r', marker='o')
    ax.plot_surface(X_grid, Y_grid, Z_grid, color='cyan', alpha=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Fitted Ground Plane from Targets")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plane visualization to {save_path}")

if __name__ == "__main__":
    # color_image_path = "dog_data/BDL205_Indie_front/1/Color/_Color_1807.png"
    # raw_depth_path = "dog_data/BDL205_Indie_front/1/Depth_Color/_Depth_Color_1807.raw"
    # color_depth_path = "dog_data/BDL205_Indie_front/1/_Depth_1807.png"
    # output_json = "target_coords_front.json"
    # ros_bag_path = "dog_data/BDL205_Indie_front/BDL205_Indie_PVP_02.bag"
    # intr = find_realsense_intrinsics(ros_bag_path)
    # # RealSense intrinsics
    # intrinsics = {
    #     "fx": intr.fx,
    #     "fy": intr.fy,
    #     "cx": intr.ppx,
    #     "cy": intr.ppy
    # }

    # bboxes = select_bounding_boxes(color_image_path, color_depth_path)
    # print(f"Selected {len(bboxes)} bounding boxes.")

    # analyze_bboxes(color_image_path, raw_depth_path, bboxes, intrinsics, output_json)
    # depth_img = load_raw_depth(raw_depth_path, shape=(720, 1280))
    # estimate_human_center_mediapipe(color_image_path, depth_img, intrinsics, output_json)
    # visualize_3d_targets(output_json)

    # === User Input ===
    color_image_path = "dog_data/BDL212_Milton_side/2/Color/_Color_0641.png"
    raw_depth_path = "dog_data/BDL212_Milton_side/2/Depth_Color/_Depth_Color_0641.raw"
    color_depth_path = "dog_data/BDL212_Milton_side/2/Depth/_Depth_0641.png"
    output_json = "target_coords_side.json"
    ros_bag_path = "dog_data/BDL212_Milton_side/BDL212_Milton_PVP_021_Cam2.bag"
    intr = find_realsense_intrinsics(ros_bag_path)
    # RealSense intrinsics
    intrinsics = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy
    }

    bboxes = select_bounding_boxes(color_image_path, color_depth_path)
    print(f"Selected {len(bboxes)} bounding boxes.")

    analyze_bboxes(color_image_path, raw_depth_path, bboxes, intrinsics, output_json)
    depth_img = load_raw_depth(raw_depth_path, shape=(720, 1280))
    estimate_human_center_mediapipe(color_image_path, depth_img, intrinsics, output_json)
    visualize_3d_targets(output_json)
    
