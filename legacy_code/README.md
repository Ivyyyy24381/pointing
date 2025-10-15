# 🎯 Pointing Gesture Analysis for Dog/Infant Cognition Research

A complete pipeline for analyzing human pointing gestures and subject (dog/infant) responses using multi-camera RealSense depth data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

This project provides tools to:
1. 📹 **Record** synchronized multi-camera RGB-D data (RealSense)
2. 🔧 **Calibrate** cameras using AprilTags or manual methods
3. 👋 **Detect** human pointing gestures (MediaPipe)
4. 🐕 **Track** dog/infant behavior (DeepLabCut/MediaPipe + SAM2)
5. 📊 **Analyze** correlation between gestures and responses
6. 🌐 **Visualize** results via web interface or command line

### Research Question
**Do dogs/infants follow human pointing gestures to select targets?**

---

## 🚀 Quick Start

### Option 1: Web Interface (Easiest)
```bash
cd web_pipeline
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### Option 2: Command Line (Full Pipeline)
```bash
# Process entire subject
python integrated_pipeline.py --base_dir {SubjectID}_side --side_view

# Process gestures
python gesture/gesture_detection.py --mode video --video_path video.mp4 --csv_path output.csv
```

📖 **See [QUICK_START.md](QUICK_START.md) for detailed instructions**

---

## 📁 Project Structure

```
pointing/
├── 🌐 web_pipeline/              # Flask web interface (simple, interactive)
├── 🐕 dog_script/                # Behavior analysis (side camera)
├── 👋 gesture/                   # Gesture detection (front camera)
├── 🔧 config/                    # Camera calibration files
├── 📹 rosbag_script/             # Data extraction from ROS bags
└── 📚 docs/                      # Documentation
```

📖 **See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for full details**

---

## 🎯 Key Features

### 🌐 Web Interface
- ✅ Step-by-step wizard (7 steps)
- ✅ Upload color + depth images
- ✅ Optional camera calibration
- ✅ AprilTag detection (automatic)
- ✅ Human pose detection (MediaPipe)
- ✅ Interactive cup labeling (click on image)
- ✅ 3D visualization + JSON export

### 📹 Front Camera Pipeline (Gestures)
- ✅ MediaPipe Pose + Hands detection
- ✅ Pointing gesture recognition (hand/arm)
- ✅ Ray casting to ground plane
- ✅ Target inference (which cup?)
- ✅ 2D/3D trajectory visualization

### 🐕 Side Camera Pipeline (Behavior)
- ✅ SAM2 foreground segmentation
- ✅ DeepLabCut (dogs) / MediaPipe (babies)
- ✅ 3D trajectory reconstruction
- ✅ Distance/angle to targets
- ✅ Closest target detection
- ✅ Automated validation

---

## 📊 Data Flow

```
🎥 Recording (3 RealSense cameras)
    ↓
💾 ROS Bags (.bag files)
    ↓
📂 Frame Extraction (rosbag_main.py)
    ↓
    ├─ Front Camera → Gesture Detection → Pointing Analysis
    └─ Side Camera  → Segmentation → Skeleton → 3D Tracking
         ↓
    📊 Combined Analysis (correlation)
```

---

## 🔧 Installation

### Basic Requirements
```bash
pip install opencv-python numpy mediapipe matplotlib PyYAML flask
```

### Full Installation (with SAM2 + DeepLabCut)
```bash
# Core dependencies
pip install -r requirements.txt

# SAM2 (for segmentation)
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
# Download checkpoint: sam2_hiera_large.pt

# DeepLabCut (for dog skeleton)
pip install deeplabcut
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [QUICK_START.md](QUICK_START.md) | Get started in 5 minutes |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Complete codebase overview |
| [PIPELINE_README.md](PIPELINE_README.md) | Detailed pipeline documentation |
| [web_pipeline/README.md](web_pipeline/README.md) | Web interface guide |
| [data_structure.md](data_structure.md) | Data format specifications |

---

## 🎓 Example Use Cases

### 1. Analyze Single Image
```bash
cd web_pipeline
python app.py
# Upload image → Click cups → Get results
```

### 2. Process One Trial
```bash
python integrated_pipeline.py \
  --base_dir BDL001_side \
  --trial_dir BDL001_side/3 \
  --side_view
```

### 3. Batch Process Subject
```bash
python integrated_pipeline.py \
  --base_dir BDL001_side \
  --side_view
```

### 4. Compare Gestures vs Behavior
```python
import pandas as pd

# Load gesture data
gestures = pd.read_csv('BDL001_front/1/processed_gesture_data.csv')
pointing_to = gestures['pointing_to'].mode()[0]

# Load behavior data
behavior = pd.read_csv('BDL001_side/1/processed_subject_result_table.csv')
approached = behavior['closest_target_label'].mode()[0]

# Did dog follow gesture?
success = (pointing_to == approached)
print(f"Gesture: {pointing_to}, Response: {approached}, Success: {success}")
```

---

## 📊 Output Files

### Gesture Analysis
- `gesture_data.csv` - Raw detections
- `processed_gesture_data.csv` - Ray casting results
  - Columns: `pointing_to`, `eye_to_wrist_ground_intersection`, distances to targets
- `2d_pointing_trace.png` - Top-down trajectory

### Behavior Analysis
- `processed_subject_result_table.csv` - Main output
  - Columns: `trace3d_x/y/z`, `dog_dir`, `target_1_r/theta/phi`, `closest_target_label`
- `*_trace3d.png` - 3D scatter plot
- `*_trace2d.png` - Top-down view
- `*_distance_plot.png` - Distance over time
- `subject_annotated_video.mp4` - Visualization

---

## 🔬 Technical Details

### Camera Setup
- 3× RealSense D435/D455 cameras
- 640×480 @ 15 FPS
- RGB + Depth synchronized
- Calibrated using AprilTags or manual methods

### Algorithms
- **Human detection**: MediaPipe Pose + Hands
- **Segmentation**: SAM2 (Segment Anything 2)
- **Dog skeleton**: DeepLabCut (quadruped model)
- **Baby skeleton**: MediaPipe Pose
- **3D reconstruction**: Pinhole camera model with depth

### Coordinate Systems
- **Camera frame**: X (right), Y (down), Z (forward)
- **World frame**: Established via AprilTag or manual calibration
- **Ground plane**: Y=0 after alignment

---

## 🛠️ Configuration

### Camera Intrinsics
Edit `config/camera_config.yaml`:
```yaml
cam3:  # Side camera
  fx: 388.569  # Focal length X
  fy: 388.116  # Focal length Y
  cx: 319.456  # Principal point X
  cy: 248.937  # Principal point Y
  width: 640
  height: 480
```

### Target Positions
Static fallback in `config/targets.yaml`:
```yaml
targets:
- id: target_1
  position_m: [1.163, 0.818, 2.618]
```

Or per-trial in `{trial_dir}/target_coordinates.json` (preferred)

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Image size mismatch | Use `integrated_pipeline.py` (auto-fixes) |
| Skeleton coords off | Check `sam2_scale_metadata.json` exists |
| No targets found | Run `label_targets.py` or check `config/targets.yaml` |
| SAM2 import error | Install SAM2 from GitHub |
| Depth file error | Convert to `.raw` (uint16) or `.npy` |

See [QUICK_START.md](QUICK_START.md#-common-issues) for more

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{pointing_gesture_analysis,
  title = {Pointing Gesture Analysis for Dog/Infant Cognition Research},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/pointing}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 🔗 Links

- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [SAM2 Repository](https://github.com/facebookresearch/segment-anything-2)
- [DeepLabCut Documentation](http://www.mackenziemathislab.org/deeplabcut)
- [RealSense SDK](https://github.com/IntelRealSense/librealsense)

---

## 📧 Contact

For questions or issues:
- Open a GitHub issue
- Email: your.email@example.com

---

**Made with ❤️ for cognitive science research**

---

## 📊 Pipeline Diagram

```
┌─────────────────────────────────────────────────────────┐
│            MULTI-CAMERA DATA COLLECTION                 │
│  👤 Human → 📷 Front Cam (RGB-D)                        │
│  🎯 Targets ← 🐕 Dog → 📷 Side Cam (RGB-D)             │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────┴─────────┐
    │                  │
    ▼                  ▼
┌─────────┐      ┌──────────┐
│ FRONT   │      │ SIDE     │
│ CAMERA  │      │ CAMERA   │
└────┬────┘      └────┬─────┘
     │                │
     │ MediaPipe      │ SAM2 + DLC
     │ Pose+Hands     │ Segmentation
     ↓                ↓
┌─────────┐      ┌──────────┐
│ Gesture │      │ Behavior │
│ Data    │      │ Data     │
└────┬────┘      └────┬─────┘
     │                │
     │ Ray Casting    │ 3D Tracking
     │ Target Infer   │ Distance Calc
     ↓                ↓
┌─────────┐      ┌──────────┐
│Pointing │      │ Approach │
│to Cup X │      │ to Cup Y │
└────┬────┘      └────┬─────┘
     │                │
     └────────┬───────┘
              ▼
       ┌──────────────┐
       │  CORRELATION │
       │   ANALYSIS   │
       │  X == Y ?    │
       └──────────────┘
```

---

**Version**: 2.0
**Last Updated**: 2025-01-09
