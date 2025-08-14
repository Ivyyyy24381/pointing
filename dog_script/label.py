import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

MAX_SIZE = 640  # Resize long edge to this size for labeling

def select_subject_entry_frame(color_dir):
    """Select the first frame where the subject clearly enters the scene."""
    frame_names = sorted([f for f in os.listdir(color_dir) if f.lower().endswith((".jpg", ".png"))])
    if not frame_names:
        print(f"⚠️ No frames found in {color_dir}")
        return None, None
    
    print("🎯 Select the first frame where the subject enters the scene...")
    
    # Show a grid of sample frames across the timeline
    sample_indices = np.linspace(0, len(frame_names) - 1, num=min(12, len(frame_names))).astype(int)
    
    # Create subplot grid
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    frame_info = []
    for i, idx in enumerate(sample_indices):
        if i >= len(axes):
            break
            
        frame_path = os.path.join(color_dir, frame_names[idx])
        img = Image.open(frame_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Frame {idx}: {frame_names[idx]}")
        axes[i].axis('off')
        frame_info.append((idx, frame_names[idx]))
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Choose the frame number where subject FIRST enters clearly (close window to continue)", fontsize=14, y=0.98)
    plt.show()
    
    while True:
        try:
            user_input = input(f"Enter frame number (0-{len(frame_names)-1}) or press Enter for middle frame: ").strip()
            if not user_input:
                selected_idx = len(frame_names) // 2
                return selected_idx, frame_names[selected_idx]
                
            frame_idx = int(user_input)
            if 0 <= frame_idx < len(frame_names):
                return frame_idx, frame_names[frame_idx]
            else:
                print(f"Please enter a number between 0 and {len(frame_names)-1}")
        except ValueError:
            print("Please enter a valid number")

def prepare_temp_images(color_dir):
    """Resize images to temp folder, keep scale factors and original sizes for back conversion."""
    tmp_dir = os.path.join(color_dir, "_tmp_resized")
    os.makedirs(tmp_dir, exist_ok=True)
    scale_factors = {}
    orig_sizes = {}

    for fname in sorted(os.listdir(color_dir)):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        src_path = os.path.join(color_dir, fname)
        img = Image.open(src_path).convert("RGB")
        orig_w, orig_h = img.size
        orig_sizes[fname] = (orig_w, orig_h)
        img.thumbnail((MAX_SIZE, MAX_SIZE))  # resize
        new_w, new_h = img.size
        scale_factors[fname] = (orig_w / new_w, orig_h / new_h)
        img.save(os.path.join(tmp_dir, fname))
    return tmp_dir, scale_factors, orig_sizes

def select_points_on_image(frame_path, scale_factor, orig_size):
    """Open resized image, allow clicking, and map points back to original size."""
    img = Image.open(frame_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    coords, labels, normalized_coords = [], [], []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            label = 1 if event.button == 1 else 0  # Left=foreground, Right=background
            # Convert back to original resolution
            sx, sy = scale_factor
            orig_x = float(event.xdata) * sx
            orig_y = float(event.ydata) * sy
            
            # Normalize to [0,1] range
            orig_w, orig_h = orig_size
            normalized_x = orig_x / orig_w
            normalized_y = orig_y / orig_h
            
            coords.append((orig_x, orig_y))
            normalized_coords.append((normalized_x, normalized_y))
            labels.append(label)
            color = 'go' if label == 1 else 'ro'
            ax.plot(event.xdata, event.ydata, color)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title(f"Click points on {os.path.basename(frame_path)} Right click to exclude objects and left click to select objects(Close when done)")
    plt.show()
    return coords, labels, normalized_coords

def label_trial_for_segmentation(trial_color_dir):
    """Label the selected frame where subject enters for SAM2 segmentation."""
    # First, select the frame where subject enters
    entry_frame_idx, entry_frame_name = select_subject_entry_frame(trial_color_dir)
    if entry_frame_idx is None:
        return None
    
    print(f"🎯 Selected frame {entry_frame_idx}: {entry_frame_name} for segmentation labeling")
    
    # Prepare temp images for display
    tmp_dir, scale_factors, orig_sizes = prepare_temp_images(trial_color_dir)
    
    # Get the specific frame for labeling
    entry_frame_tmp = entry_frame_name.replace('.png', '.jpg')  # Convert to jpg for tmp
    if entry_frame_tmp not in scale_factors:
        entry_frame_tmp = entry_frame_name  # Use original if no conversion
    
    frame_path = os.path.join(tmp_dir, entry_frame_tmp)
    if not os.path.exists(frame_path):
        print(f"⚠️ Frame not found in temp dir: {frame_path}")
        return None
    
    print(f"🖱️ Labeling subject points on: {entry_frame_name}")
    pts, lbs, norm_pts = select_points_on_image(frame_path, scale_factors[entry_frame_tmp], orig_sizes[entry_frame_tmp])
    
    if not pts:
        print("⚠️ No points selected for segmentation")
        return None
    
    # Clean up temp directory
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    
    return {
        "starting_frame_idx": entry_frame_idx,
        "starting_frame_name": entry_frame_name,
        "frames": [entry_frame_name],
        "points": [list(p) for p in pts],
        "labels": lbs,
        "normalized_points": [list(p) for p in norm_pts]
    }

def label_trial(trial_color_dir):
    """Label multiple frames from a Color folder (for backward compatibility)."""
    tmp_dir, scale_factors, orig_sizes = prepare_temp_images(trial_color_dir)
    frame_names = sorted([f for f in os.listdir(tmp_dir) if f.lower().endswith((".jpg", ".png"))])
    if not frame_names:
        print(f"⚠️ No frames in {trial_color_dir}")
        return None

    selected_idxs = np.linspace(0, len(frame_names)-1, num=min(5, len(frame_names)), dtype=int)
    all_points, all_labels, all_frames, all_normalized = [], [], [], []

    for idx in selected_idxs:
        fname = frame_names[idx]
        frame_path = os.path.join(tmp_dir, fname)
        print(f"🖱️ Labeling frame: {fname}")
        pts, lbs, norm_pts = select_points_on_image(frame_path, scale_factors[fname], orig_sizes[fname])
        all_points.extend(pts)
        all_labels.extend(lbs)
        all_normalized.extend(norm_pts)
        all_frames.extend([fname] * len(pts))

    return {"frames": all_frames, "points": all_points, "labels": all_labels, "normalized_points": all_normalized}

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