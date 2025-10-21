# Pointing Detection System

**Comprehensive RGB-D analysis system for detecting human gestures and tracking subjects**

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8--3.12-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Overview

The Pointing Detection System analyzes RGB-D camera data to detect and track human pointing gestures and subjects in the scene. The system provides a unified graphical interface (`main_ui.py`) for the complete analysis pipeline.

### Key Capabilities

- **Human Skeleton Detection**: 33-keypoint pose detection using MediaPipe
- **Pointing Direction Analysis**: Automatic detection of pointing arm and direction
- **Target Detection**: YOLO-based detection of objects (cups) being pointed at
- **Subject Tracking**: Detect and track dogs or babies in the scene
- **3D Reconstruction**: Depth-based 3D coordinate extraction
- **Visualization**: Real-time 2D and 3D skeleton visualization

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     main_ui.py                          │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ Page 1:          │  │ Page 2:                  │   │
│  │ Target Detection │→ │ Skeleton Processing      │   │
│  │                  │  │                          │   │
│  │ • Load data      │  │ • Extract skeleton       │   │
│  │ • Detect targets │  │ • Detect subjects        │   │
│  │ • Save locations │  │ • Visualize 3D           │   │
│  └──────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
           ↓                          ↓
    trial_output/            Original folder (synced)
```

## Quick Start

**Get started in under 30 minutes!**

### 1. Install Python 3.8-3.12

```bash
# Check Python version
python3 --version

# If needed, install Python 3.12
# macOS:
brew install python@3.12

# Linux:
sudo apt-get install python3.12 python3.12-venv
```

### 2. Set Up Environment

```bash
# Clone/download repository
cd /path/to/pointing

# Create virtual environment
python3.12 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Required Model

Download `best.pt` (21 MB YOLOv8 model) and place in:
```
step0_data_loading/best.pt
```

### 4. Launch the UI

```bash
python main_ui.py
```

### 5. Process Your First Trial

1. **Page 1**: Load trial → Detect targets → Save
2. **Page 2**: Auto-loads trial → Process frames → View results

**That's it!** See [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions.

## Documentation

### User Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[USER_GUIDE.md](USER_GUIDE.md)** | Installation and quick start | All users |
| **[MAIN_UI_GUIDE.md](MAIN_UI_GUIDE.md)** | Complete UI workflow guide | All users |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common problems and solutions | All users |

### Technical Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[DATA_FORMAT.md](DATA_FORMAT.md)** | Input/output data formats | Developers |
| **[CONFIGURATION.md](CONFIGURATION.md)** | Camera settings and tuning | Advanced users |
| **[OUTPUT_REFERENCE.md](OUTPUT_REFERENCE.md)** | Output file specifications | Developers |

### Legacy Documentation

Older documentation is available in subdirectories:
- `step0_data_loading/README.md` - Data loading details
- `step2_skeleton_extraction/README.md` - Skeleton extraction
- `step3_subject_extraction/README.md` - Subject detection
- `MACOS_SETUP.md` - macOS-specific setup
- `DOG_BABY_DETECTION.md` - Subject detection details

## System Requirements

### Software

- **Python**: 3.8 - 3.12 (REQUIRED - MediaPipe does NOT support 3.13+)
- **Operating System**: macOS 10.15+, Ubuntu 18.04+, or Windows 10+
- **Disk Space**: ~2 GB for dependencies and models
- **Memory**: 4 GB RAM minimum, 8 GB recommended

### Hardware

- **Processor**: Multi-core CPU (Intel i5/AMD Ryzen 5 or better)
- **Graphics**: Optional GPU for faster processing (CUDA/Metal supported)
- **Storage**: SSD recommended for large datasets

### Dependencies

Core packages (automatically installed):
- **opencv-python** - Image processing
- **mediapipe** - Pose detection
- **ultralytics** - YOLO object detection
- **torch/torchvision** - Deep learning backend
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **matplotlib** - Visualization
- **pandas** - Data analysis

See `requirements.txt` for complete list.

## Features

### Page 1: Data Loading & Target Detection

- Auto-detects multi-camera and single-camera folder structures
- Standardizes all data to unified format (`trial_input/`)
- YOLO-based automatic cup/target detection
- 3D localization with depth data
- Interactive frame browser with depth visualization
- Saves detections to JSON

### Page 2: Skeleton Processing & Visualization

- MediaPipe Pose for 33-keypoint human skeleton
- Automatic pointing arm detection (left/right)
- 3D skeleton reconstruction from depth
- Dog and baby detection (optional)
- Real-time 2D and 3D visualization
- Batch processing with progress tracking
- Automatic result syncing to original folder

### Output Files

Generated after processing:
- `skeleton_2d.json` - Frame-by-frame 2D keypoints
- `skeleton_3d.json` - 3D skeleton coordinates
- `processed_gesture.csv` - Pointing analysis data
- `2d_pointing_trace.png` - Trajectory visualization
- `dog_detection_results.json` - Dog tracking (if enabled)
- `baby_detection_results.json` - Baby tracking (if enabled)
- `fig/` - 3D skeleton visualizations per frame

See [OUTPUT_REFERENCE.md](OUTPUT_REFERENCE.md) for details.

## Supported Data Formats

### Multi-Camera Structure
```
trial_1/
├── cam1/
│   ├── color/
│   │   └── frame_000001.png
│   └── depth/
│       └── frame_000001.npy
├── cam2/
│   ├── color/
│   └── depth/
└── cam3/
    ├── color/
    └── depth/
```

### Single-Camera Structure
```
1/
├── Color/
│   └── _Color_0001.png
├── Depth/
│   └── _Depth_0001.raw
└── Depth_Color/
    └── _Depth_Color_0001.png
```

**The system auto-detects and standardizes all formats!**

See [DATA_FORMAT.md](DATA_FORMAT.md) for complete specifications.

## Usage Examples

### Basic Workflow

```bash
# 1. Launch UI
python main_ui.py

# 2. In Page 1:
#    - Browse to data folder
#    - Select trial and camera
#    - Navigate to frame with targets
#    - Click "Detect Targets"
#    - Click "Save Detections"

# 3. In Page 2:
#    - Trial auto-loads
#    - (Optional) Enable dog/baby detection
#    - Click "Process All Frames"
#    - Review results

# 4. Find outputs in:
#    - trial_output/<trial>/<camera>/
#    - Original data folder (synced)
```

### Command-Line Processing

For automation or batch processing:

```bash
# Process specific trial
python step2_skeleton_extraction/batch_processor.py \
  --trial trial_input/trial_1/cam1

# Process all trials
python step2_skeleton_extraction/batch_processor.py
```

### Python API

```python
from step2_skeleton_extraction import MediaPipeHumanDetector
import cv2

# Initialize detector
detector = MediaPipeHumanDetector(
    model_complexity=1,
    min_detection_confidence=0.5
)

# Process image
image = cv2.imread('frame_000001.png')
result = detector.detect_single(image, frame_number=1)

if result:
    print(f"Detected {len(result.landmarks_2d)} keypoints")
    print(f"Pointing arm: {result.metadata['pointing_arm']}")

# Process folder
results = detector.detect_image_folder('trial_input/trial_1/cam1/color')
detector.save_results(results, 'output/skeleton_2d.json')
```

See individual module documentation for more examples.

## Project Structure

```
pointing/
├── main_ui.py                      # Main entry point (2-page UI)
│
├── step0_data_loading/             # Data loading & target detection
│   ├── ui_data_loader.py          # Data loading UI (embedded in Page 1)
│   ├── target_detector.py         # YOLO target detection
│   ├── best.pt                    # YOLOv8 model (REQUIRED - download separately)
│   └── ...
│
├── step2_skeleton_extraction/      # Skeleton extraction & analysis
│   ├── ui_skeleton_extractor.py   # Skeleton UI (embedded in Page 2)
│   ├── mediapipe_human.py         # MediaPipe pose detector
│   ├── visualize_skeleton_3d.py   # 3D visualization
│   └── ...
│
├── step3_subject_extraction/       # Dog/baby detection
│   ├── subject_detector.py        # Subject detection (lower-half crop)
│   └── ...
│
├── trial_input/                    # Standardized data (auto-created)
├── trial_output/                   # Processing results (auto-created)
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── USER_GUIDE.md                  # User guide
├── MAIN_UI_GUIDE.md               # UI workflow guide
├── DATA_FORMAT.md                 # Data format specifications
├── CONFIGURATION.md               # Configuration guide
├── OUTPUT_REFERENCE.md            # Output file reference
└── TROUBLESHOOTING.md             # Troubleshooting guide
```

## Configuration

### Camera Intrinsics

Auto-detected based on image resolution. For custom calibration:

**Edit:** `main_ui.py`, line ~203
```python
self.page2_ui.fx = 618.5  # Your calibrated fx
self.page2_ui.fy = 618.3  # Your calibrated fy
self.page2_ui.cx = 321.2  # Your calibrated cx
self.page2_ui.cy = 239.8  # Your calibrated cy
```

See [CONFIGURATION.md](CONFIGURATION.md) for details.

### Detection Settings

Adjustable in Page 2 UI:
- **Model Complexity**: 0 (fast), 1 (balanced), 2 (accurate)
- **Detection Confidence**: 0.0-1.0 threshold
- **Subject Detection**: Enable/disable dog and baby tracking

### Performance Tuning

For faster processing:
- Use Model Complexity 0
- Disable dog/baby detection if not needed
- Process every Nth frame instead of all

See [CONFIGURATION.md](CONFIGURATION.md) for optimization tips.

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "MediaPipe requires Python <3.13" | Use Python 3.8-3.12 |
| "No module named 'tkinter'" | Install Python from python.org |
| "Model not found" | Download best.pt to step0_data_loading/ |
| "No frames found" | Check folder structure matches supported formats |
| Processing is slow | Reduce model complexity, disable unused features |

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for comprehensive solutions.

## Development

### Running Tests

```bash
# Test imports
python -c "import cv2, mediapipe, torch; print('OK')"

# Test YOLO model
python -c "from ultralytics import YOLO; YOLO('step0_data_loading/best.pt')"

# Test MediaPipe
python -c "import mediapipe as mp; mp.solutions.pose.Pose()"
```

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions/classes
- Comment complex logic
- Keep functions focused and small

## Citation

If you use this system in your research, please cite:

```bibtex
@software{pointing_detection_system,
  title = {Pointing Detection System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/pointing}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

This system uses the following open-source projects:
- [MediaPipe](https://github.com/google/mediapipe) - Pose detection
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [OpenCV](https://opencv.org/) - Computer vision
- [PyTorch](https://pytorch.org/) - Deep learning

## Support

For help and questions:

1. **Check documentation**:
   - [USER_GUIDE.md](USER_GUIDE.md) - Getting started
   - [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common problems

2. **Run diagnostics**:
   ```bash
   python --version
   pip list
   ls -lh step0_data_loading/best.pt
   ```

3. **Check console output**:
   ```bash
   python main_ui.py
   # Read all error messages
   ```

4. **Search existing issues** (if repository is public)

5. **Create a new issue** with:
   - Python version
   - OS version
   - Complete error message
   - Steps to reproduce

## Changelog

### Version 2.0 (Current)
- Unified main_ui.py with 2-page workflow
- Auto-detection of data formats
- Automatic result syncing
- Dog and baby detection
- Improved 3D visualization
- Comprehensive documentation

### Version 1.0
- Initial release
- Separate UIs for each step
- Manual data standardization
- Basic skeleton detection

---

**Ready to get started?** See [USER_GUIDE.md](USER_GUIDE.md) for installation and your first trial!
