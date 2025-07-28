import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

MAX_SIZE = 640  # Resize long edge to this size for labeling

def prepare_temp_images(color_dir):
    """Resize images to temp folder, keep scale factors for back conversion."""
    tmp_dir = os.path.join(color_dir, "_tmp_resized")
    os.makedirs(tmp_dir, exist_ok=True)
    scale_factors = {}

    for fname in sorted(os.listdir(color_dir)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        src_path = os.path.join(color_dir, fname)
        img = Image.open(src_path).convert("RGB")
        orig_w, orig_h = img.size
        img.thumbnail((MAX_SIZE, MAX_SIZE))  # resize
        new_w, new_h = img.size
        scale_factors[fname] = (orig_w / new_w, orig_h / new_h)
        img.save(os.path.join(tmp_dir, fname))
    return tmp_dir, scale_factors

def select_points_on_image(frame_path, scale_factor):
    """Open resized image, allow clicking, and map points back to original size."""
    img = Image.open(frame_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    coords, labels = [], []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            label = 1 if event.button == 1 else 0  # Left=foreground, Right=background
            # Convert back to original resolution
            sx, sy = scale_factor
            coords.append((float(event.xdata) * sx, float(event.ydata) * sy))
            labels.append(label)
            color = 'go' if label == 1 else 'ro'
            ax.plot(event.xdata, event.ydata, color)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title(f"Click points on {os.path.basename(frame_path)} (Close when done)")
    plt.show()
    return coords, labels

def label_trial(trial_color_dir):
    """Label 3 sampled frames from a Color folder (using resized images)."""
    tmp_dir, scale_factors = prepare_temp_images(trial_color_dir)
    frame_names = sorted([f for f in os.listdir(tmp_dir) if f.lower().endswith((".jpg", ".png"))])
    if not frame_names:
        print(f"⚠️ No frames in {trial_color_dir}")
        return None

    selected_idxs = np.linspace(0, len(frame_names)-1, num=min(3, len(frame_names)), dtype=int)
    all_points, all_labels, all_frames = [], [], []

    for idx in selected_idxs:
        fname = frame_names[idx]
        frame_path = os.path.join(tmp_dir, fname)
        print(f"🖱️ Labeling frame: {fname}")
        pts, lbs = select_points_on_image(frame_path, scale_factors[fname])
        all_points.extend(pts)
        all_labels.extend(lbs)
        all_frames.extend([fname] * len(pts))

    return {"frames": all_frames, "points": all_points, "labels": all_labels}

def process_base_folder(base_dir):
    """Loop through trials and run labeling."""
    for subject in sorted(os.listdir(base_dir)):
        subject_path = os.path.join(base_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        for trial in sorted(os.listdir(subject_path)):
            trial_color = os.path.join(subject_path, trial, "Color")
            if not os.path.isdir(trial_color):
                continue

            points_file = os.path.join(subject_path, trial, "interactive_points.json")
            if os.path.exists(points_file):
                print(f"✅ {points_file} exists, skipping.")
                continue

            data = label_trial(trial_color)
            if data:
                with open(points_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"💾 Saved {points_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base folder with all subject folders")
    args = parser.parse_args()
    process_base_folder(os.path.expanduser(args.base_dir))