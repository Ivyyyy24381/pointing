# 🚀 Quick Start Guide

Get up and running with the pointing gesture analysis pipeline in 5 minutes!

---

## 📋 Prerequisites

```bash
# Python 3.8+
python --version

# Install core dependencies
pip install opencv-python numpy mediapipe matplotlib PyYAML flask

# Optional: For SAM2 segmentation
pip install torch torchvision
# Follow SAM2 installation: https://github.com/facebookresearch/segment-anything-2

# Optional: For dog skeleton detection
pip install deeplabcut
```

---

## 🎯 Choose Your Path

### Option A: Web Interface (Easiest) ✨

**Best for**: Quick testing, single images, no batch processing

```bash
cd web_pipeline
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` and follow the 7-step wizard:
1. Upload color + depth images
2. (Optional) Calibrate camera
3. Detect AprilTag (auto)
4. Detect human
5. Click on 4 cups
6. View results
7. Download JSON

**Output**: `results/analysis_summary.json` + 3D visualization

---

### Option B: Command Line (Full Pipeline) 🔧

**Best for**: Batch processing, research workflows, automation

#### Step 1: Organize Your Data

```bash
# Expected structure:
{SubjectID}_side/
├── 1/
│   ├── Color/Color_000000.png, Color_000001.png, ...
│   └── Depth/Depth_000000.raw (or .npy), Depth_000001.raw, ...
├── 2/
├── 3/
...
```

#### Step 2: Run Integrated Pipeline

```bash
python integrated_pipeline.py \
  --base_dir data/{SubjectID}_side \
  --side_view
```

This will:
- ✅ Standardize images to 640×480
- ✅ Run SAM2 segmentation (interactive GUI)
- ✅ Detect skeleton (DeepLabCut or MediaPipe)
- ✅ Label targets (GUI if needed)
- ✅ Compute 3D positions and distances
- ✅ Generate visualizations

**Output**: `processed_subject_result_table.csv` + plots

---

## 🎬 Example Workflow: New Subject

### 1️⃣ If you have ROS bags

```bash
# Extract frames from bags
python rosbag_script/rosbag_main.py
# Enter: /path/to/bag_folder
# Mark trial boundaries by pressing 's' (start) and 'e' (end)
```

### 2️⃣ Process front camera (human gestures)

```bash
# Detect gestures
python gesture/gesture_detection.py \
  --mode video \
  --video_path BDL001_front/1/Color.mp4 \
  --csv_path BDL001_front/1/gesture_data.csv

# Clean up short gestures
python gesture/eval/data_cleanup.py \
  --input BDL001_front/1/gesture_data.csv \
  --output BDL001_front/1/gesture_data_cleanup.csv \
  --threshold 0.4

# Process gestures (compute pointing direction)
python gesture/gesture_data_process.py
# NOTE: Edit script to set correct paths first!
```

### 3️⃣ Process side camera (dog/infant behavior)

```bash
python integrated_pipeline.py \
  --base_dir BDL001_side \
  --side_view
```

### 4️⃣ Analyze results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load gesture data
gestures = pd.read_csv('BDL001_front/1/processed_gesture_data.csv')
print("Human pointed to:", gestures['pointing_to'].value_counts())

# Load behavior data
behavior = pd.read_csv('BDL001_side/1/processed_subject_result_table.csv')
print("Dog approached:", behavior['closest_target_label'].value_counts())

# Plot trajectory
plt.figure(figsize=(10, 8))
plt.scatter(behavior['trace3d_x'], behavior['trace3d_z'], c=behavior.index, cmap='viridis')
plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.title('Dog Trajectory (Top View)')
plt.colorbar(label='Frame')
plt.show()
```

---

## 📸 Example: Process Single Image (Web Interface)

```bash
cd web_pipeline
python app.py
```

1. **Upload**:
   - Color: `BDL001_side/1/Color/Color_001000.png`
   - Depth: `BDL001_side/1/Depth/Depth_001000.raw`

2. **Calibration** (optional):
   - Uncheck "Use custom calibration" → uses defaults
   - Or manually enter: fx=388.57, fy=388.12, cx=319.46, cy=248.94

3. **Detect Human**: Click button → sees skeleton overlay

4. **Mark Cups**: Click on 4 cup locations

5. **Results**:
   - See 3D plot showing human + cups + pointing ray
   - Download `results/analysis_summary.json`

---

## 🎥 Example: Process Single Trial (Command Line)

```bash
# Just one trial
python integrated_pipeline.py \
  --base_dir BDL001_side \
  --trial_dir BDL001_side/3 \
  --side_view
```

**Interactive prompts**:
1. **Segmentation**: Click green dots (foreground) and red dots (background), press 's' when done
2. **Targets**: If no `target_coordinates.json`, GUI opens → click 4 cup centers

**Outputs in `BDL001_side/3/`**:
- `masked_video.mp4`
- `masked_video_skeleton.json`
- `target_coordinates.json`
- `processed_subject_result_table.csv`
- `BDL001_side_3_trace3d.png`
- `BDL001_side_3_trace2d.png`
- `BDL001_side_3_distance_plot.png`

---

## 🔍 Verify Installation

```bash
# Test imports
python -c "import cv2, numpy, mediapipe, matplotlib, yaml, flask; print('✅ Core dependencies OK')"

# Test camera config
python -c "import yaml; print(yaml.safe_load(open('config/camera_config.yaml')))"

# Test MediaPipe
python -c "import mediapipe as mp; pose = mp.solutions.pose.Pose(); print('✅ MediaPipe OK')"
```

---

## 🐛 Common Issues

### "ModuleNotFoundError: No module named 'X'"
```bash
pip install X
```

### "No such file or directory: config/camera_config.yaml"
```bash
# You're in the wrong directory
cd /path/to/pointing/
ls config/  # Should see camera_config.yaml
```

### "SAM2 not found"
```bash
# SAM2 requires manual installation
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
# Download checkpoint: https://github.com/facebookresearch/segment-anything-2#download-checkpoints
```

### Images are wrong size
```bash
# Integrated pipeline auto-fixes this
python integrated_pipeline.py --base_dir YOUR_DIR --side_view
```

### Web interface won't start
```bash
cd web_pipeline
pip install -r requirements.txt
python app.py
# Check output for error messages
```

---

## 💡 Tips

1. **Start with web interface** for quick testing
2. **Use integrated_pipeline.py** for production workflows
3. **Always check image dimensions** (should be 640×480)
4. **Label targets once per trial**, then reuse `target_coordinates.json`
5. **Use `--force_reprocess`** to overwrite existing outputs
6. **Check validation logs** in `validate_pipeline.py` output

---

## 📚 Next Steps

- Read [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for full codebase overview
- Read [`PIPELINE_README.md`](PIPELINE_README.md) for detailed pipeline docs
- Check [`web_pipeline/README.md`](web_pipeline/README.md) for web interface docs
- Review example notebooks (if available) in `notebooks/`

---

## 🆘 Still Stuck?

1. Check logs in terminal output
2. Read error messages carefully
3. Verify file paths are correct
4. Check that images exist and are valid
5. Try running on a single trial first
6. Use `--help` flag for command options:
   ```bash
   python integrated_pipeline.py --help
   python gesture/gesture_detection.py --help
   ```

---

**Happy analyzing! 🎯**
