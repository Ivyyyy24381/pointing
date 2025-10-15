# 🎯 Pointing Gesture Analysis - Project Structure

Complete codebase organization for dog/infant pointing behavior research.

---

## 📁 Directory Structure

```
pointing/
│
├── 📹 DATA COLLECTION & PREPROCESSING
│   ├── record_bag.py                    # Multi-camera sync recording (3 RealSense cameras)
│   └── rosbag_script/
│       ├── rosbag_main.py               # ROS bag → frames + trial splitting
│       ├── bag_to_video.py              # Bag → MP4 conversion
│       └── batch_split_bag.py           # Batch processing
│
├── 🔧 CAMERA CALIBRATION
│   ├── calibration.py                   # Extract intrinsics + extrinsics calibration
│   ├── calibrate_target.py              # Interactive target labeling with GUI
│   └── config/
│       ├── camera_config.yaml           # Camera intrinsics (fx,fy,cx,cy) + extrinsics (R,t)
│       ├── targets.yaml                 # Static target positions (fallback)
│       └── human.yaml                   # Human origin position
│
├── 👋 GESTURE ANALYSIS (Front Camera)
│   └── gesture/
│       ├── gesture_detection.py         # ⭐ MediaPipe: Detect pointing gestures
│       ├── gesture_util.py              # Vector math, pointing validation
│       ├── gesture_data_process.py      # ⭐ Ray casting, target inference
│       └── eval/
│           ├── data_cleanup.py          # Filter short gestures (<0.4s)
│           ├── pointing_eval.py         # Accuracy evaluation
│           ├── plot.py                  # Statistical plots (violin, time-series)
│           ├── run_eval.py              # Full pipeline orchestrator
│           └── video_output.py          # Video stitching
│
├── 🐕 BEHAVIOR ANALYSIS (Side Camera)
│   ├── dog_script/
│   │   ├── dog_main.py                  # ⭐ Original pipeline (single trial)
│   │   ├── segmentation.py              # ⭐ SAM2 foreground segmentation
│   │   ├── detect_dog_skeleton.py       # ⭐ DeepLabCut (dogs) / MediaPipe (babies)
│   │   ├── label_targets.py             # ⭐ Interactive target labeling GUI
│   │   ├── dog_pose_visualize.py        # ⭐ 3D reconstruction + distance calc
│   │   ├── dog_data_eval_plot.py        # Trial metadata integration
│   │   └── dog_global_eval.py           # Cross-trial aggregation
│   │
│   └── integrated_pipeline.py           # ⭐ Complete pipeline with validation
│
├── 🌐 WEB INTERFACE
│   └── web_pipeline/
│       ├── app.py                       # ⭐ Flask server (simple version)
│       ├── templates/
│       │   └── index.html               # Interactive web UI
│       ├── requirements.txt             # Python dependencies
│       └── README.md                    # Web interface docs
│
├── 🔍 VALIDATION & UTILITIES
│   ├── validate_pipeline.py             # Quality checks
│   └── data_structure.md                # Data format documentation
│
└── 📄 DOCUMENTATION
    ├── README.md                        # Main project README
    ├── PROJECT_STRUCTURE.md             # This file
    ├── PIPELINE_README.md               # Detailed pipeline guide
    └── data_structure.md                # Data format specs
```

---

## 🎯 Main Entry Points (⭐ = Most Important)

### For Data Collection
```bash
# Record multi-camera session
python record_bag.py

# Extract frames from ROS bags
python rosbag_script/rosbag_main.py
```

### For Gesture Analysis (Front Camera)
```bash
# 1. Detect gestures
python gesture/gesture_detection.py \
  --mode video \
  --video_path {SubjectID}_front/1/Color.mp4 \
  --csv_path gesture_data.csv

# 2. Clean data
python gesture/eval/data_cleanup.py \
  --input gesture_data.csv \
  --output gesture_data_cleanup.csv

# 3. Process gestures (compute pointing)
python gesture/gesture_data_process.py
```

### For Behavior Analysis (Side Camera) - **RECOMMENDED**
```bash
# Complete pipeline (all steps)
python integrated_pipeline.py \
  --base_dir {SubjectID}_side \
  --side_view
```

### For Behavior Analysis (Side Camera) - Original
```bash
# Original pipeline (single trial)
python dog_script/dog_main.py \
  --root_path {SubjectID}_side \
  --side_view
```

### For Web Interface
```bash
cd web_pipeline
python app.py
# Open http://localhost:5000
```

---

## 📊 Pipeline Workflows

### Pipeline A: Gesture Analysis (Front Camera)

```
Input: {SubjectID}_front/trial_X/Color/, Depth/

Step 1: gesture_detection.py
        ↓ gesture_data.csv (raw detections)

Step 2: data_cleanup.py
        ↓ gesture_data_cleanup.csv (filtered)

Step 3: gesture_data_process.py
        ↓ processed_gesture_data.csv

Output:
  - processed_gesture_data.csv
    • pointing_to (which target)
    • ground_intersection (ray cast to ground)
    • distances to all targets
  - 2d_pointing_trace.png (top-down view)
  - fig/{frame}_skeleton_plot.png (3D per frame)
```

### Pipeline B: Behavior Analysis (Side Camera)

```
Input: {SubjectID}_side/trial_X/Color/, Depth/

Step 1: dog_main.py or integrated_pipeline.py
        ↓ Standardize images (640×480)

Step 2: segmentation.py (SAM2)
        ↓ masked_video.mp4 + sam2_scale_metadata.json

Step 3: detect_dog_skeleton.py (DLC/MediaPipe)
        ↓ masked_video_skeleton.json

Step 4: label_targets.py (GUI or load existing)
        ↓ target_coordinates.json

Step 5: dog_pose_visualize.py
        ↓ 3D reconstruction + analysis

Output:
  - processed_subject_result_table.csv
    • trace3d_x/y/z (3D trajectory)
    • dog_dir (orientation vector)
    • target_1_r/theta/phi (distance/angles to each target)
    • closest_target_label
  - *_trace3d.png (3D scatter)
  - *_trace2d.png (top-down)
  - *_distance_plot.png (distance vs time)
  - subject_annotated_video.mp4
```

---

## 🔑 Key Configuration Files

### 1. Camera Intrinsics
`config/camera_config.yaml`
```yaml
cam1:  # Front camera option 1
  fx: 385.716
  fy: 385.254
  cx: 320.872
  cy: 239.636
  width: 640
  height: 480
  extrinsics_to_cam1: [4x4 identity]

cam2:  # Front camera option 2
  fx: 614.521
  fy: 614.460
  cx: 322.136
  cy: 248.588
  extrinsics_to_cam1: [4x4 transform]

cam3:  # Side camera
  fx: 388.569
  fy: 388.116
  cx: 319.456
  cy: 248.937
  extrinsics_to_cam1: [4x4 transform]
```

### 2. Target Positions (Static Fallback)
`config/targets.yaml`
```yaml
targets:
- id: target_1
  center: [913, 552]      # pixel coordinates
  depth_m: 2.618
  position_m: [1.163, 0.818, 2.618]  # 3D coordinates
- id: target_2
  ...
```

### 3. Human Position
`config/human.yaml`
```yaml
targets:
- id: human
  center: [647, 318]
  depth_m: 3.145
  position_m: [0.036, -0.215, 3.145]
```

### 4. Per-Trial Targets (Preferred)
`{trial_dir}/target_coordinates.json`
```json
{
  "targets": [
    {
      "id": 0,
      "label": "target_1",
      "pixel_coords": [x, y],
      "world_coords": [X, Y, Z],
      "depth_m": d
    },
    ...
  ]
}
```

---

## 📦 Data Flow

### Input Data Structure
```
{SubjectID}_front/
├── 1/                     # Trial 1
│   ├── Color/
│   │   └── Color_XXXXXX.png
│   ├── Depth/
│   │   └── Depth_XXXXXX.raw (or .npy)
│   └── Depth_Color/
│       └── Depth_Color_XXXXXX.png
├── 2/                     # Trial 2
├── auto_split.csv         # Trial boundaries
└── rosbag_metadata.yaml   # Camera intrinsics

{SubjectID}_side/
├── (same structure)
```

### Output Data Structure
```
{SubjectID}_front/1/
├── gesture_data.csv
├── gesture_data_cleanup.csv
├── processed_gesture_data.csv
└── 2d_pointing_trace.png

{SubjectID}_side/1/
├── masked_video.mp4
├── masked_video_skeleton.json
├── target_coordinates.json
├── sam2_scale_metadata.json
├── processed_subject_result_table.csv
├── *_trace3d.png
├── *_trace2d.png
├── *_distance_plot.png
└── subject_annotated_video.mp4
```

---

## 🛠️ Common Tasks

### 1. Process a New Subject
```bash
# 1. Extract from ROS bags
python rosbag_script/rosbag_main.py
# Input bag directory, mark trial boundaries

# 2. Process front camera (gestures)
python gesture/gesture_detection.py --mode video \
  --video_path {SubjectID}_front/1/Color.mp4 \
  --csv_path {SubjectID}_front/1/gesture_data.csv

python gesture/eval/data_cleanup.py \
  --input {SubjectID}_front/1/gesture_data.csv \
  --output {SubjectID}_front/1/gesture_data_cleanup.csv

python gesture/gesture_data_process.py
# Edit paths in script first

# 3. Process side camera (behavior)
python integrated_pipeline.py \
  --base_dir {SubjectID}_side \
  --side_view
```

### 2. Calibrate New Cameras
```bash
# Extract intrinsics from bag
python -c "
from calibration import find_realsense_intrinsics
find_realsense_intrinsics('path/to/file.bag')
"

# Label targets for calibration
python calibrate_target.py
# Input: base_dir, trial, frame number

# Compute camera-to-world transform
python calibration.py
```

### 3. Reprocess Specific Steps
```bash
# Re-segment only
python integrated_pipeline.py \
  --base_dir {SubjectID}_side \
  --trial_dir {SubjectID}_side/1 \
  --steps segmentation \
  --force_reprocess

# Re-detect skeleton only
python integrated_pipeline.py \
  --base_dir {SubjectID}_side \
  --trial_dir {SubjectID}_side/1 \
  --steps skeleton \
  --force_reprocess

# Re-label targets
python dog_script/label_targets.py \
  --trial_folder {SubjectID}_side/1
```

---

## 🐛 Troubleshooting

### "Image size mismatch"
- Problem: Images not 640×480
- Solution: `integrated_pipeline.py` auto-standardizes, or run manually:
  ```python
  from dog_script.dog_main import standardize_images
  standardize_images('Color/', 'Depth/')
  ```

### "Skeleton coordinates off"
- Problem: SAM2 scale factor not applied
- Solution: Check `sam2_scale_metadata.json` exists, or rerun segmentation

### "No targets found"
- Problem: Missing `target_coordinates.json`
- Solution: Run `label_targets.py` or ensure `config/targets.yaml` exists

### "Depth file format error"
- Problem: Unsupported depth format
- Solution: Convert to `.raw` (uint16) or `.npy`:
  ```python
  import numpy as np
  depth = np.load('depth.npy')
  depth_uint16 = (depth * 1000).astype(np.uint16)
  depth_uint16.tofile('depth.raw')
  ```

---

## 📚 Key Dependencies

```bash
# Core
opencv-python>=4.8.0
numpy>=1.24.0
mediapipe>=0.10.0
matplotlib>=3.8.0

# Segmentation
torch>=2.0.0
segment-anything-2  # SAM2

# Skeleton detection
deeplabcut  # For dogs
# mediapipe already included

# Web interface
flask>=3.0.0
PyYAML>=6.0.0

# Optional
apriltag>=0.3.1  # For AprilTag calibration
```

---

## 🎓 Research Context

This pipeline analyzes:
1. **Human pointing gestures** → Which target is human indicating?
2. **Dog/infant responses** → Which target does subject choose?
3. **Correlation** → Does subject follow human gesture?

### Key Metrics
- **Pointing accuracy**: Distance from ray to target
- **Selection accuracy**: Which target subject approached
- **Response time**: Delay between gesture and selection
- **Trajectory analysis**: Path subject took to target

---

## 📧 Need Help?

1. Check `PIPELINE_README.md` for detailed workflow
2. Check `web_pipeline/README.md` for web interface
3. Check individual script docstrings
4. Review validation logs from `validate_pipeline.py`

---

**Last Updated**: 2025-01-09
**Version**: 2.0
