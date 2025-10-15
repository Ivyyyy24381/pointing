import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import yaml

# ideal world relative coordinate
world_coords_dict = {
    "target_1": [-1.05, 0.0, 1.52],
    "target_2": [-0.39, 0.0, 1.79],
    "target_3": [0.39, 0.0, 1.82],
    "target_4": [1.05, 0.0, 1.52]
}

def calibrate_targets(color_path, depth_path, metadata_yaml, output_json):
    """select 4 targets, calculate 3d space"""

    # === Load camera intrinsics ===
    with open(metadata_yaml, 'r') as f:
        meta = yaml.safe_load(f)
    fx, fy = meta["intrinsics"]["fx"], meta["intrinsics"]["fy"]
    cx, cy = meta["intrinsics"]["ppx"], meta["intrinsics"]["ppy"]

    # === Load images ===
    color_img = cv2.imread(color_path)
    if color_img is None:
        raise FileNotFoundError(f"❌ Cannot read color image: {color_path}")
    height, width = color_img.shape[:2]

    with open(depth_path, 'rb') as f:
        depth_img = np.frombuffer(f.read(), dtype=np.uint16).reshape((height, width)).astype(np.float32) / 1000.0

    # === Select ROIs ===
    rois = []
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Drag 4 ROIs (close when done)")
    ax.set_navigate(False)
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x1 - x2), abs(y1 - y2)
        rois.append((x, y, w, h))
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        fig.canvas.draw()

    selector = RectangleSelector(
        ax, 
        onselect,
        useblit=True,               
        button=[1],                 
        minspanx=5, minspany=5,     
        spancoords='pixels',
        interactive=False           
    )
    plt.show()

    # === find coordinations from the bounding box ===
    measured_positions = {}
    targets = []

    for i, (x, y, w, h) in enumerate(rois):
        label = f"target_{i+1}"
        center_x, center_y = x + w // 2, y + h // 2
        roi = depth_img[y:y+h, x:x+w]
        nonzero_vals = roi[roi > 0]
        if nonzero_vals.size > 0:
            d = float(nonzero_vals.mean())
            X = (center_x - cx) * d / fx
            Y = (center_y - cy) * d / fy
            Z = d
        else:
            X, Y, Z = 0.0, 0.0, 0.0  # Missing

        measured_positions[label] = np.array([X, Y, Z])
        targets.append({
            "bbox": [x, y, x + w, y + h],
            "center_px": [center_x, center_y],
            "avg_depth_m": Z,
            "x": X,
            "y": Y,
            "z": Z,
            "label": label
        })

    # === calculate offset to correct zero targets ===
    measured_valid = []
    world_valid = []
    for label, pos in measured_positions.items():
        if not np.allclose(pos, [0, 0, 0]):
            measured_valid.append(pos)
            world_valid.append(world_coords_dict[label])
    measured_valid = np.array(measured_valid)
    world_valid = np.array(world_valid)

    if len(measured_valid) > 0:
        # offset = world - measured（use the first effective target）
        offset = np.mean(world_valid - measured_valid, axis=0)
    else:
        offset = np.zeros(3)

    # === fix missing target target ===
    for t in targets:
        if np.allclose([t["x"], t["y"], t["z"]], [0, 0, 0]):
            # world_coords_dict + offset 
            world_base = np.array(world_coords_dict[t["label"]])
            corrected = world_base - offset  # keep relative relation
            t["x"], t["y"], t["z"] = corrected.tolist()
            t["avg_depth_m"] = corrected[2]

    # === save ===
    with open(output_json, 'w') as f:
        json.dump(targets, f, indent=2)

    print(f"✅ Saved {len(targets)} targets to {output_json}")

def find_targets(base_dir):
    import os
    base_folder = base_dir
    trial = input("Enter trial number (e.g. 8): ").strip()
    frame_num = input("Enter frame number (e.g. 2317): ").strip()
    frame = f"{int(frame_num):06d}" 
    trial_path = os.path.join(base_folder, trial)
    color_path = os.path.join(trial_path, "Color", f"Color_{frame}.png")
    depth_path = os.path.join(trial_path, "Depth", f"Depth_{frame}.raw")
    metadata_yaml = os.path.join(base_folder, "rosbag_metadata.yaml")
    output_json = os.path.join(base_folder, "targets.json")

    calibrate_targets(color_path, depth_path, metadata_yaml, output_json)

if __name__ == "__main__":
    base_dir = input("Enter the base directory containing trials: ").strip()
    find_targets(base_dir)