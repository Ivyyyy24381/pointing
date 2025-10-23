# 🐶 Dog Behavior Tracking Pipeline

This repository processes dog behavior data from RealSense cameras to analyze how dogs select targets in response to human gestures. The pipeline includes detection, pose estimation, 3D localization, behavior visualization, and trial-level/global analyses.

---

## 📌 1️⃣ Detection with DeepLabCut

**Camera**: RealSense D455 (side view)

**Steps**:
- Side-view video (`Color.mp4`) processed using DeepLabCut (quadruped model).
- DeepLabCut detects:
  - 2D keypoints for the dog (nose, ears, eyes, neck, tail base, limbs, etc.).
  - Bounding boxes.
  - Confidence scores.
- Results saved as JSON (per frame keypoints, bounding boxes, scores).

---

## 📌 2️⃣ Pose Processing & CSV Data

**Script**: `dog_pose_visualize.py`

Processes the DLC JSON and depth frames to extract:

### 🐕 Dog Direction (`dog_dir`)
- Defined as the unit vector from the tail base to the neck.
- Saved as `[x, y, z]` per frame.

### 🐾 Dog Trace (`trace3d_x`, `trace3d_y`, `trace3d_z`)
- 3D position of the nose (or bounding box center as fallback).
- Smoothed and interpolated over frames.

### 🎯 Distances to Targets & Human
For each frame:
- Distance (`*_r`): Euclidean distance (meters).
- Azimuth (`*_phi`): Angle around vertical axis.
- Elevation (`*_theta`): Angle from horizontal plane.
- Distances interpolated when missing.

### 🔎 Metadata Columns
- `frame_index`: Global frame index (matches image names).
- `time_sec`: Timestamp.
- `bbox_*`: Bounding box position & size.
- 2D & 3D joint keypoints.
- Head and torso orientation vectors.
- Flags noting interpolated data.

---

## 📊 3️⃣ Visualizations (Per Trial)

### 🔹 3D Dog Trace (`*_trace3d.png`)
- Dog’s trajectory in 3D.
- Rainbow color shows time progression.
- Targets shown as cubes (gray).
- Human shown as a red X.
- Smoothed spline overlay for trajectory.

### 🔹 2D Top-Down Trace (`*_trace2d.png`)
- Same as 3D but projected onto the ground (X-Z plane).
- Labels for each target.
- Rainbow color for time progression.
- Spline fit for smoothed trajectory.

### 🔹 Distance to Targets (`*_distance_plot.png`)
- Distance to each target over time.
- Automatically detects when the dog approaches a target (distance < threshold).
- Labels these events on the plot.
- Rolling weighted average applied for smoothing.

---

## 🗂 4️⃣ Trial Metadata Integration

**Script**: `dog_data_eval_plot.py`

- Each CSV is enriched with:
  - Dog ID and name.
  - Trial number.
  - Assigned target (from `PVP_Comprehension_Data.csv`).
- Automatically trims data to valid search period:
  - Starts at user-defined time.
  - Ends when dog first approaches a target and then backs away.
- Flags which target was touched per frame (`touched_target`).

---

## 🌎 5️⃣ Global Analysis Across Trials

**Script**: `dog_global_eval.py`

For all trials in a folder:
- Groups trials by **assigned location** (Target 1 to 4).
- Concatenates 3D and 2D traces per target.
- Rainbow colors reset per trial.
- Smoothed spline added for each trial (distinct color per trial, with legend).

**Output**:
- `global_trace_target_X_3d.png`
- `global_trace_target_X_2d.png`

✅ Targets are plotted and labeled.  
✅ Human location plotted.  
✅ Legends identify spline lines by trial number.

---

## ✅ Interpolation & Cleaning Notes

- Bounding boxes, keypoints, dog direction, trace, and distances are interpolated.
- Time and frame indices remain continuous even when frames are skipped.
- Interpolated frames flagged in the CSV.
- All 3D traces aligned to a common ground plane where applicable.

---

## 🐕 Summary of Data Columns (Per Frame)

| Column                 | Description                                  |
|------------------------|----------------------------------------------|
| dog_id                 | Unique dog identifier                        |
| dog_name               | Dog’s name                                   |
| trial_number           | Trial number                                 |
| touched_target         | 0 (no target), or 1-4                       |
| trace3d_x/y/z          | Smoothed 3D nose positions                  |
| dog_dir                | Dog direction vector                         |
| *_r                    | Distance to each target/human                |
| *_theta                | Elevation angle                              |
| *_phi                  | Azimuth angle                                |
| interpolated_bbox      | 0/1 flag                                    |
| interpolated_keypoints | 0/1 flag                                    |
| interpolated_dir       | 0/1 flag                                    |

---

## 🛠 Usage Example

**For a single trial**:

```bash
python dog_data_eval_plot.py --csv path/to/your_csv.csv --start START_TIME --side_view
