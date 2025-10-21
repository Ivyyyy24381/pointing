# Pointing Detection System - Complete User Guide

**A Comprehensive Guide to Processing RGB-D Pointing Gesture Data**

Version 2.0 | Updated October 2025

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Page 1: Data Loading & Target Detection](#3-page-1-data-loading--target-detection)
4. [Page 2: Skeleton Processing & 3D Visualization](#4-page-2-skeleton-processing--3d-visualization)
5. [Complete Workflows](#5-complete-workflows)
6. [Input Data Format](#6-input-data-format)
7. [Output Files](#7-output-files)
8. [Tips & Best Practices](#8-tips--best-practices)
9. [Troubleshooting](#9-troubleshooting)
10. [Appendix](#10-appendix)

---

## 1. Introduction

### 1.1 What This Tool Does

The Pointing Detection System is a comprehensive RGB-D analysis platform that:

- **Detects human skeletons** using MediaPipe (33 keypoints)
- **Identifies pointing gestures** with automatic arm detection (left/right)
- **Locates targets** (cups/objects) using YOLO object detection
- **Tracks subjects** (dogs or babies) in the scene
- **Reconstructs 3D poses** from depth camera data
- **Visualizes results** in real-time 2D and 3D views
- **Exports analysis data** in JSON and CSV formats

### 1.2 Who Should Use This Tool

This system is designed for:

- **Researchers** studying human-object interaction and pointing gestures
- **Computer Vision Engineers** processing RGB-D sensor data
- **Data Scientists** analyzing gesture and tracking datasets
- **Students** learning about pose estimation and 3D reconstruction

### 1.3 System Requirements

#### Hardware
- **Processor**: Multi-core CPU (Intel i5/AMD Ryzen 5 or better)
- **Memory**: 4 GB RAM minimum, 8 GB recommended
- **Storage**: 2 GB for dependencies, plus space for your data
- **Display**: 1024x768 minimum, 1920x1080 recommended

#### Software
- **Operating System**: macOS 10.15+, Ubuntu 18.04+, or Windows 10+
- **Python**: 3.8 - 3.12 (CRITICAL: MediaPipe does NOT support Python 3.13+)
- **Storage Type**: SSD recommended for large datasets

#### Input Data
- RGB-D camera data (color images + depth maps)
- Supported formats: PNG, JPG (color); NPY, RAW (depth)
- Typical resolution: 640x480, 1280x720, or 1920x1080

---

## 2. Getting Started

### 2.1 Installation

#### Step 1: Verify Python Version

```bash
python3 --version
```

You MUST use Python 3.8-3.12. If you have Python 3.13+, install Python 3.12:

**macOS (Homebrew):**
```bash
brew install python@3.12
```

**macOS (Official Installer):**
Download from https://www.python.org/downloads/macos/

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3-tk
```

**Windows:**
Download from https://www.python.org/downloads/windows/
- Check "Add Python to PATH"
- Check "tcl/tk and IDLE"

#### Step 2: Create Virtual Environment

Navigate to the project directory:

```bash
cd /path/to/pointing
```

Create and activate virtual environment:

**macOS/Linux:**
```bash
python3.12 -m venv venv
source venv/bin/activate
python --version  # Verify: should show 3.12.x
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
python --version
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all packages (5-15 minutes)
pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
python -c "import cv2, mediapipe, torch, numpy, ultralytics; print('All packages installed!')"
```

If you see "All packages installed!", you're ready!

### 2.2 Download Required Model

The system requires a YOLOv8 model for target detection:

**Required File:**
- Filename: `best.pt`
- Size: ~21 MB
- Location: `/Users/ivy/Documents/GitHub/pointing/step0_data_loading/best.pt`

**To obtain:**
1. Check if it exists: `ls -lh step0_data_loading/best.pt`
2. If missing, copy from backup or contact your data provider
3. Alternative path: `step1_calibration_process/target_detection/automatic_mode/best.pt`

### 2.3 Launching the Application

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Launch main UI
python main_ui.py
```

A window will open with two tabs:
- **Page 1: Target Detection** (blue header)
- **Page 2: Skeleton Processing** (green header)

---

## 3. Page 1: Data Loading & Target Detection

### 3.1 Overview

**Purpose**: Load trial data, browse frames, detect and save target locations

**Workflow**: Load Data → Browse Frames → (Optional) Trim → Detect Targets → Save

### 3.2 Loading Trial Data

#### Method 1: Browse for Folder

1. Click **"Browse..."** button (or **File > Open Folder**)
2. Navigate to your data folder (e.g., `/Users/ivy/Downloads/dog_data/`)
3. Select the parent folder containing your trials
4. Click **"Select Folder"** or **"Open"**

**What Happens:**
- System scans folder for all trials
- Auto-detects folder structure (multi-camera or single-camera)
- Populates the **"Trial"** dropdown with discovered trials

#### Method 2: Command-Line Argument

```bash
python main_ui.py /path/to/your/data
```

### 3.3 Selecting Trial and Camera

1. **Select Trial**: Choose from the **"Trial"** dropdown
   - Example: `trial_1`, `BDL049_Star_side_cam`, `1`

2. **Select Camera**: Choose from the **"Camera"** dropdown
   - Multi-camera: `cam1`, `cam2`, `cam3`
   - Single-camera: `single_camera` (auto-selected)

**What Happens:**
- System processes trial to standardized format
- Creates `trial_input/<trial_name>/<camera>/` folder
- Converts all frames to:
  - Color: PNG format in `color/` subfolder
  - Depth: NPY format in `depth/` subfolder
- Saves metadata to `metadata.json`
- Loads first frame for viewing
- Status shows: "Ready: X frames in trial_input/"

**Note**: Original data is NEVER modified. All processing uses copies in `trial_input/`.

### 3.4 Browsing Frames

#### Navigation Controls

**Slider**: Drag to jump to any frame

**Navigation Buttons**:
- **◀◀ -10**: Jump back 10 frames
- **◀ Prev**: Previous frame
- **Next ▶**: Next frame
- **+10 ▶▶**: Jump forward 10 frames
- **🔄 Reload Frame**: Refresh current frame

**Frame Information Display**:
```
📂 Source: trial_input/ (standardized)
Trial: 1
Camera: single_camera
Frame: 215
Color: (720, 1280, 3), PNG
Depth: (720, 1280), .npy, [0.456-3.892]m
```

#### Display Windows

**Left Panel: Color Image**
- RGB image as captured by camera
- Shows detection overlays (green boxes) when targets detected

**Right Panel: Depth Image**
- Colorized depth map (JET colormap)
- Closer objects = red/yellow, farther = blue/purple
- Shows detection overlays with depth values

### 3.5 Frame Trimming (Optional)

Use trimming to extract a specific range of frames from your trial.

#### Trimming Workflow

1. **Navigate** to the frame where you want to start
2. Click **"📍 Set Start"**
   - Status updates: "Start: Frame 215 (set end frame)"

3. **Navigate** to the frame where you want to end
4. Click **"📍 Set End"**
   - Status updates: "Trim: Frame 215 to 275 (61 frames)"

5. Click **"✂️ Apply Trim & Save"**
   - Confirmation dialog appears
   - Click **"Yes"** to proceed

**What Happens:**
- System copies only the selected frame range to `trial_input/`
- Renumbers frames starting from the original frame numbers
- Updates slider to show trimmed range
- Resets trim markers
- Status shows: "✅ Trimmed: 61 frames saved to trial_input/"

**To Reset**: Click **"🔄 Reset"** to clear trim markers

**Important**: Trimming overwrites the existing `trial_input/` data for this trial. Original data is unchanged.

### 3.6 Target Detection

Targets are objects (typically cups) that subjects point at.

#### Detection Workflow

1. **Navigate** to a frame with clear view of targets
   - Targets should be unobstructed
   - Good lighting preferred
   - Typically use an early frame before interaction

2. Click **"🎯 Detect Targets"**

**What Happens (First Time):**
- Loads YOLO model (~5 seconds, one-time)
- Model file: `step0_data_loading/best.pt`
- Status shows: "Loading YOLO model..."

**What Happens (Every Time):**
- Detects cups/objects in current frame
- Computes 3D positions using depth data
- Auto-detects camera intrinsics based on image resolution
- Overlays green bounding boxes on both views
- Labels targets as `target_1`, `target_2`, etc.
- Sorted from right to left (by x-coordinate)
- Status shows: "✅ Detected 4 target(s)"

**Detection Information**:
```
Found 4 target(s)
target_1: avg:1.686m, med:1.689m
target_2: avg:3.564m, med:3.580m
target_3: avg:2.918m, med:2.922m
target_4: avg:3.132m, med:3.135m
```

3. **Verify** detections look correct
   - Check bounding boxes align with cups
   - Check depth values are reasonable
   - Re-run detection on different frame if needed

4. Click **"💾 Save Detections"**

**What Happens:**
- Saves to `trial_output/<trial>/<camera>/target_detections_cam_frame.json`
- Computes ground plane transformation (if 3+ targets detected)
- Saves ground plane to `ground_plane_transform.json`
- Status shows: "💾 Saved detections with ground plane correction (5.3° tilt)"

**To Clear**: Click **"🗑️ Clear"** to remove detection overlays

### 3.7 Understanding Target Detection Output

**File**: `trial_output/<trial>/<camera>/target_detections_cam_frame.json`

**Format**:
```json
[
  {
    "bbox": [810, 583, 864, 645],
    "center_px": [837, 614],
    "avg_depth_m": 1.6864252090454102,
    "x": 0.3601363319045483,
    "y": 0.46433821473987447,
    "z": 1.6864252090454102,
    "label": "target_1"
  },
  ...
]
```

**Fields**:
- `bbox`: [x1, y1, x2, y2] bounding box in pixels
- `center_px`: [x, y] center point in pixels
- `avg_depth_m`: Average depth within bbox (meters)
- `x, y, z`: 3D position in camera coordinates (meters)
- `label`: Target identifier (sorted right-to-left)

**Ground Plane File**: `ground_plane_transform.json`
```json
{
  "rotation_matrix": [[...], [...], [...]],
  "info": {
    "angle_deg": 5.3,
    "axis": [0.998, 0.062, 0.003]
  },
  "description": "Rotation matrix to align ground plane to horizontal"
}
```

---

## 4. Page 2: Skeleton Processing & 3D Visualization

### 4.1 Overview

**Purpose**: Extract skeleton from all frames, detect subjects, visualize 3D poses

**Workflow**: Auto-Load Trial → Configure Settings → Process Frames → View Results

### 4.2 Auto-Loading from Page 1

When you switch from Page 1 to Page 2:

**Automatic Actions:**
1. Trial is auto-loaded from Page 1
2. Loads all frames from `trial_input/<trial>/<camera>/`
3. Loads ground plane transform (if available)
4. Loads target locations (if saved)
5. Displays first frame
6. Status shows: "✓ Loaded <trial>: 61 frames, Ground plane: Yes, Targets: 4"

**Manual Loading:**
If you start on Page 2 directly, you can:
1. Navigate to `trial_input/<trial>/<camera>/` folder
2. System will load all standardized data

### 4.3 Detector Settings

#### Model Complexity

Controls accuracy vs. speed trade-off:

- **0 (Fast)**: Fastest processing, lower accuracy
- **1 (Balanced)**: Default, good balance (recommended)
- **2 (Accurate)**: Highest accuracy, slower processing

**When to Use:**
- Use **0** for quick previews or large datasets
- Use **1** for most production work
- Use **2** for final analysis or challenging poses

#### Detection Confidence

Range: 0.0 - 1.0 (default: 0.5)

- **Lower (0.3-0.4)**: More detections, more false positives
- **Higher (0.6-0.8)**: Fewer detections, fewer false positives

**When to Adjust:**
- Lower if skeletons not detected in good frames
- Raise if getting spurious detections in background

#### Subject Detection

**👤 Human**: Always enabled (primary subject)

**🐶 Dog (lower half)**:
- Enable if dog is present in scene
- Uses lower 60% of frame for detection
- Runs batch processing when enabled
- Creates cropped video and skeleton data

**👶 Baby (lower half)**:
- Enable if baby/infant is present in scene
- Uses lower 60% of frame for detection
- Similar processing to dog detection

**Important**: Enabling dog/baby triggers batch processing of ALL frames. This may take several minutes.

#### Pointing Arm Selection

- **Auto Detect**: Automatically determines left/right arm (recommended)
- **Left Arm**: Force use of left arm for pointing vector
- **Right Arm**: Force use of right arm for pointing vector

**When to Override:**
- Auto-detection fails
- Subject switches arms mid-trial
- Specific research requirements

### 4.4 Frame Navigation

Same controls as Page 1:

**Playback Controls**:
- **⏮**: Jump to first frame
- **◀**: Previous frame
- **▶**: Play/pause video playback
- **▶▶**: Next frame
- **⏭**: Jump to last frame

**Slider**: Drag to any frame

**Frame Counter**: Shows "Frame: 215 / 275"

### 4.5 Processing Frames

#### Single Frame Processing

**Use Case**: Test settings, preview results

1. Navigate to desired frame
2. Click **"🎯 Process Current Frame"**

**What Happens:**
- Processes current frame only
- Updates 2D display with skeleton overlay
- Updates 3D visualization
- Shows detection info in bottom panel
- Results NOT saved

#### Batch Processing (All Frames)

**Use Case**: Complete trial analysis

1. Configure settings (model complexity, confidence, subjects)
2. Click **"🔄 Process All Frames (Auto-saves)"**

**What Happens:**
1. **Initialization**:
   - Initializes MediaPipe detector(s)
   - Loads all frames
   - Creates output directories

2. **Processing Loop** (for each frame):
   - Loads color and depth images
   - Runs pose detection
   - Extracts 3D skeleton from depth
   - Computes arm vectors
   - Determines pointing arm
   - Updates progress bar

3. **Subject Processing** (if enabled):
   - Creates cropped video of lower half
   - Runs subject-specific pose estimation
   - Saves subject skeleton data

4. **Analysis**:
   - Determines consistent pointing arm across trial
   - Computes pointing trajectories
   - Generates visualizations

5. **Saving**:
   - Saves to `trial_output/<trial>/<camera>/`
   - Syncs to original data folder
   - Generates summary files

**Progress Indication:**
- Progress bar: 0-100%
- Status updates: "Processing frame 45/61..."
- Time estimate shown

**Duration**: Approximately 1-2 minutes for 100 frames (Complexity 1)

### 4.6 Visualization Features

#### 2D Display (Middle Panel)

**Display Options** (checkboxes above canvas):

- **Show Skeleton**: Green overlay of detected pose
  - Keypoints shown as colored circles
  - Bones shown as green lines
  - Line thickness indicates confidence

- **Show Arm Vectors**: Yellow arrow from shoulder→wrist→infinity
  - Shows pointing direction
  - Labeled "Left Arm" or "Right Arm"

- **Show Depth**: Toggle between color and depth view
  - Useful for verifying depth quality
  - Shows depth-based 3D reconstruction

**Skeleton Color Coding:**
- Green circles: Detected keypoints
- Green lines: Skeleton connections
- Yellow arrow: Pointing vector
- Thickness: Indicates detection confidence

#### 3D Visualization (Right Panel)

**3D Plot Features:**
- **Skeleton**: 3D pose in camera coordinates
- **Targets**: Red circles at target locations (if detected)
- **Ground Plane**: Grid showing ground reference
- **Arm Vector**: Extended line showing pointing direction
- **Coordinate Axes**: X (red), Y (green), Z (blue)

**Coordinate System:**
- X: Right (meters)
- Y: Down (meters)
- Z: Forward/depth (meters)

**Interactive Controls:**
- Rotate: Click and drag
- Zoom: Scroll wheel
- Pan: Right-click and drag (or Shift+Click)

**View Automatically Updates** as you browse frames

### 4.7 Detection Information Panel

**Displays** (left panel, bottom):

**Human Detection:**
```
Human Detected: Yes
Keypoints: 33/33
Pointing Arm: right
Confidence: 0.87

Arm Vector (3D):
  Direction: [0.234, -0.123, 0.964]
  Shoulder: [0.123, 0.456, 1.234]
  Wrist: [0.345, 0.234, 1.567]

Landmarks (2D): 33 keypoints
Landmarks (3D): 33 keypoints with depth
```

**Subject Detection** (if enabled):
```
Dog Detected: Yes
Keypoints: 24/24
Confidence: 0.91

Subject Bounding Box:
  [145, 320, 678, 720]
```

---

## 5. Complete Workflows

### 5.1 Workflow 1: Full Pipeline (New Trial)

**Goal**: Process a new trial from scratch

**Steps:**

1. **Launch Application**
   ```bash
   source venv/bin/activate
   python main_ui.py
   ```

2. **Page 1: Load and Detect Targets**
   - Browse to data folder
   - Select trial and camera
   - Navigate to frame with visible targets
   - Click "🎯 Detect Targets"
   - Verify detections
   - Click "💾 Save Detections"

3. **Page 2: Extract Skeleton**
   - Switch to Page 2 tab (auto-loads trial)
   - (Optional) Enable dog/baby detection
   - Verify settings (Model Complexity: 1, Confidence: 0.5)
   - Click "🔄 Process All Frames (Auto-saves)"
   - Wait for completion (~1-2 min per 100 frames)

4. **Review Results**
   - Browse through frames
   - Check skeleton accuracy
   - View 3D visualizations
   - Verify pointing directions

5. **Locate Output Files**
   - Check `trial_output/<trial>/<camera>/`
   - Check original data folder (synced)

**Total Time**: 5-10 minutes for typical trial

### 5.2 Workflow 2: Frame Trimming

**Goal**: Extract specific time segment from trial

**Use Case**: Long recording, only need specific interaction period

**Steps:**

1. **Page 1: Load Trial**
   - Browse and load full trial
   - Status shows: "Ready: 500 frames"

2. **Identify Start Frame**
   - Use slider to find interaction start
   - Example: Frame 215 (subject enters scene)
   - Click "📍 Set Start"

3. **Identify End Frame**
   - Navigate to interaction end
   - Example: Frame 275 (subject exits)
   - Click "📍 Set End"
   - Status: "Trim: Frame 215 to 275 (61 frames)"

4. **Apply Trim**
   - Click "✂️ Apply Trim & Save"
   - Confirm: "Yes"
   - Wait for processing
   - Status: "✅ Trimmed: 61 frames saved"

5. **Proceed with Page 2**
   - Switch to Page 2
   - Process trimmed frames (much faster!)

**Time Saved**: Processing 61 frames vs 500 frames = ~7x faster

### 5.3 Workflow 3: Re-processing Existing Data

**Goal**: Re-run analysis with different settings

**Use Case**: Improve accuracy, try different subject detection, adjust arm selection

**Steps:**

1. **Navigate to trial_input**
   - Trial already exists in `trial_input/<trial>/<camera>/`
   - No need to reload from original source

2. **Page 1: Load Existing**
   - Browse to `trial_input/` folder
   - Select trial
   - Frames load instantly (already standardized)

3. **Page 1: Verify/Update Targets** (optional)
   - If targets already saved, skip
   - If need to redetect, navigate to frame and detect again

4. **Page 2: Adjust Settings**
   - Change Model Complexity (try 2 for better accuracy)
   - Adjust Confidence threshold
   - Enable/disable subject detection
   - Change arm selection if needed

5. **Page 2: Reprocess**
   - Click "🔄 Process All Frames (Auto-saves)"
   - New results overwrite old results
   - Compare outputs

**Benefits**: Faster iteration, no need to re-standardize data

### 5.4 Workflow 4: Multi-Trial Batch Processing

**Goal**: Process many trials with same settings

**Approach**: Manual batch (future versions may automate)

**Steps:**

For each trial:

1. **Page 1**: Load trial → Detect targets → Save
2. **Page 2**: Process frames (use same settings)
3. Move to next trial

**Tips for Efficiency:**
- Use Model Complexity 0 for speed
- Skip target detection if not needed
- Disable subject detection if not applicable
- Process during off-hours for long runs

**Approximate Time**:
- 2-3 minutes per trial (100 frames, no subjects)
- 5-10 minutes per trial (100 frames, with dog/baby)

---

## 6. Input Data Format

### 6.1 Supported Folder Structures

The system auto-detects and supports two structures:

#### Structure 1: Multi-Camera

```
your_data/
└── trial_1/
    ├── cam1/
    │   ├── color/
    │   │   ├── frame_000001.png
    │   │   ├── frame_000002.png
    │   │   └── ...
    │   └── depth/
    │       ├── frame_000001.npy
    │       ├── frame_000002.npy
    │       └── ...
    ├── cam2/
    │   ├── color/
    │   └── depth/
    └── cam3/
        ├── color/
        └── depth/
```

**Characteristics:**
- Multiple camera subfolders
- Each camera has `color/` and `depth/` subfolders
- Frame numbering: `frame_NNNNNN.png` (6 digits)

#### Structure 2: Single-Camera

```
your_data/
└── 1/
    ├── Color/  (or color/)
    │   ├── _Color_0001.png
    │   ├── _Color_0002.png
    │   └── ...
    ├── Depth/  (or Depth_Color/)
    │   ├── _Depth_0001.raw
    │   ├── _Depth_0002.raw
    │   └── ...
    └── Depth_Color/
        ├── _Depth_Color_0001.png
        └── ...
```

**Characteristics:**
- No camera subfolders
- Has `Color/` and `Depth/` (or `Depth_Color/`) at trial root
- Frame numbering: `_Color_NNNN.png` or similar patterns
- Depth may be `.raw` or `.png` (auto-detected)

### 6.2 Supported File Formats

#### Color Images
- **PNG**: Recommended, lossless
- **JPG/JPEG**: Supported, compressed
- **Resolution**: Any (640x480, 1280x720, 1920x1080 typical)
- **Color Space**: RGB or BGR (auto-handled by OpenCV)

#### Depth Images

**NPY (NumPy Array)**:
- Format: `.npy` binary
- Data type: `float32` or `uint16`
- Units: Meters (float32) or millimeters (uint16)
- Shape: (height, width)

**RAW (Binary)**:
- Format: `.raw` binary file
- Data type: `uint16` (big-endian or little-endian)
- Units: Millimeters
- Shape: Determined from corresponding color image

**PNG (Depth Colorized)**:
- Format: `.png` (in Depth_Color/ folder)
- Visual representation only
- Not used for 3D reconstruction

### 6.3 Frame Naming Conventions

The system recognizes these patterns:

**Multi-Camera Style:**
- `frame_000001.png`, `frame_000002.png`, ...
- `frame_1.png`, `frame_2.png`, ...

**Single-Camera Style:**
- `_Color_0001.png`, `_Color_0002.png`, ...
- `Color_0001.png`, `Color_0002.png`, ...

**Frame Number Extraction:**
- System extracts numerical portion
- Matches corresponding color/depth pairs
- Sorts chronologically

### 6.4 Camera Intrinsics (Auto-Detection)

The system auto-detects intrinsics based on image resolution:

**640x480:**
```python
fx = fy = 615.0
cx = 320.0
cy = 240.0
```

**1280x720:**
```python
fx = fy = 922.5
cx = 640.0
cy = 360.0
```

**1920x1080:**
```python
fx = fy = 1383.75
cx = 960.0
cy = 540.0
```

**Other Resolutions:**
```python
fx = fy = width * 0.9
cx = width / 2.0
cy = height / 2.0
```

**To Override**: Edit `main_ui.py` lines 203-219 (see Appendix)

---

## 7. Output Files

### 7.1 Output Directory Structure

```
trial_output/
└── <trial_name>/
    └── <camera_id>/
        ├── target_detections_cam_frame.json
        ├── ground_plane_transform.json
        ├── skeleton_2d.json
        ├── skeleton_3d.json
        ├── processed_gesture.csv
        ├── pointing_hand.json
        ├── 2d_pointing_trace.png
        ├── detection_summary.txt
        ├── dog_detection_results.json  (if enabled)
        ├── baby_detection_results.json  (if enabled)
        ├── dog_detection_cropped_video.mp4  (if enabled)
        └── fig/
            ├── frame_000001.png
            ├── frame_000002.png
            └── ...
```

**AND** synced to your original data folder!

### 7.2 Target Detection Files

#### target_detections_cam_frame.json

**Format**: JSON array of detections

```json
[
  {
    "bbox": [810, 583, 864, 645],
    "center_px": [837, 614],
    "avg_depth_m": 1.686,
    "x": 0.360,
    "y": 0.464,
    "z": 1.686,
    "label": "target_1"
  }
]
```

**Fields:**
- `bbox`: Bounding box [x1, y1, x2, y2] in pixels
- `center_px`: Center point [x, y] in pixels
- `avg_depth_m`: Average depth in meters
- `x, y, z`: 3D position in camera coordinates (meters)
- `label`: Identifier (sorted right-to-left)

#### ground_plane_transform.json

**Format**: Rotation matrix and metadata

```json
{
  "rotation_matrix": [
    [0.996, -0.002, 0.092],
    [0.001, 1.000, 0.011],
    [-0.092, -0.011, 0.996]
  ],
  "info": {
    "angle_deg": 5.3,
    "axis": [0.998, 0.062, 0.003]
  },
  "description": "Rotation matrix to align ground plane to horizontal"
}
```

**Usage**: Transforms 3D skeleton to ground-plane-aligned coordinates

### 7.3 Skeleton Files

#### skeleton_2d.json

**Format**: Frame-by-frame 2D keypoints

```json
{
  "frame_000215": {
    "landmarks_2d": [
      {"x": 640.5, "y": 123.4, "visibility": 0.95},
      ...  // 33 keypoints total
    ],
    "frame_number": 215,
    "timestamp": null
  },
  "frame_000216": { ... }
}
```

**Keypoint Order** (MediaPipe Pose):
```
0: nose, 1: left_eye_inner, 2: left_eye, 3: left_eye_outer,
4: right_eye_inner, 5: right_eye, 6: right_eye_outer,
7: left_ear, 8: right_ear, 9: mouth_left, 10: mouth_right,
11: left_shoulder, 12: right_shoulder,
13: left_elbow, 14: right_elbow,
15: left_wrist, 16: right_wrist,
17-22: left/right hand landmarks,
23: left_hip, 24: right_hip,
25: left_knee, 26: right_knee,
27: left_ankle, 28: right_ankle,
29-32: left/right foot landmarks
```

#### skeleton_3d.json

**Format**: Frame-by-frame 3D keypoints with depth

```json
{
  "frame_000215": {
    "landmarks_3d": [
      {"x": 0.123, "y": 0.456, "z": 1.234, "visibility": 0.95},
      ...  // 33 keypoints
    ],
    "frame_number": 215
  }
}
```

**Units**: Meters in camera coordinate system

### 7.4 Analysis Files

#### processed_gesture.csv

**Format**: CSV with frame-by-frame gesture analysis

**Columns:**
```csv
frame_number,timestamp,
pointing_arm,
shoulder_x,shoulder_y,shoulder_z,
elbow_x,elbow_y,elbow_z,
wrist_x,wrist_y,wrist_z,
arm_vector_x,arm_vector_y,arm_vector_z,
arm_length,
confidence,
...
```

**Use Cases:**
- Time-series analysis
- Statistical analysis in Python/R
- Plotting trajectories
- Machine learning features

#### pointing_hand.json

**Format**: Determined pointing arm for trial

```json
{
  "pointing_hand": "right",
  "confidence": 0.85,
  "method": "majority_vote"
}
```

#### 2d_pointing_trace.png

**Visual**: 2D trajectory of wrist position over time
- Overlaid on first frame
- Shows movement path
- Useful for qualitative inspection

#### detection_summary.txt

**Format**: Text summary of processing

```
Processing Summary
==================
Trial: 1
Camera: single_camera
Total Frames: 61
Frames Processed: 61
Frames with Detection: 58
Detection Rate: 95.1%

Pointing Arm: right (85% confidence)

Processing Time: 1.8 minutes
Avg FPS: 0.56
```

### 7.5 Subject Detection Files (Optional)

#### dog_detection_results.json / baby_detection_results.json

**Format**: Frame-by-frame subject skeleton

```json
{
  "frame_000215": {
    "keypoints": [
      {"x": 0.234, "y": 0.567, "z": 1.234, "confidence": 0.89},
      ...  // Subject-specific keypoint count
    ],
    "bbox": [145, 320, 678, 720],
    "confidence": 0.91
  }
}
```

#### dog_detection_cropped_video.mp4

**Format**: MP4 video of cropped region (lower 60% of frame)
- Shows subject detection area
- Useful for verification
- Can be used for further analysis

### 7.6 Visualization Files

#### fig/ folder

Contains 3D skeleton visualization for each frame:

```
fig/
├── frame_000215.png
├── frame_000216.png
└── ...
```

**Content:**
- 3D skeleton plot
- Target locations (if detected)
- Ground plane grid
- Arm pointing vector
- Coordinate axes

**Use Cases:**
- Quality inspection
- Creating animations (stitch frames together)
- Presentations and reports

---

## 8. Tips & Best Practices

### 8.1 Recommended Workflow Order

**Always follow this order for best results:**

1. **Load and inspect data** (Page 1)
   - Check frame quality
   - Verify depth data is valid
   - Identify good frames for target detection

2. **Trim if needed** (Page 1)
   - Extract relevant time segment
   - Reduces processing time
   - Focuses on interaction period

3. **Detect targets** (Page 1)
   - Use early frame with clear view
   - Before subject occludes targets
   - Save detections before switching pages

4. **Configure settings** (Page 2)
   - Choose model complexity based on speed/accuracy needs
   - Enable subject detection if applicable
   - Set appropriate confidence threshold

5. **Process all frames** (Page 2)
   - Batch process for consistency
   - Don't process frame-by-frame manually

6. **Review results** (Page 2)
   - Browse through frames
   - Check detection quality
   - Re-run with adjusted settings if needed

### 8.2 Performance Optimization

#### Speed Up Processing

**Use Model Complexity 0:**
- 2-3x faster than Complexity 1
- Acceptable for most use cases
- Trade-off: Slightly lower accuracy

**Disable Unused Features:**
- Don't enable dog/baby detection unless needed
- Skip target detection if not required
- Process every Nth frame if full temporal resolution not needed

**Trim Aggressively:**
- Only process frames with interaction
- Skip static segments
- Reduce total frame count

**Hardware:**
- Close other applications
- Use SSD for data storage
- Ensure 8 GB+ RAM available

#### Improve Accuracy

**Use Model Complexity 2:**
- Best accuracy for pose estimation
- Useful for challenging poses or occlusions
- Trade-off: Slower processing

**Adjust Confidence:**
- Lower threshold (0.3-0.4) if skeletons missing
- Raise threshold (0.6-0.8) if too many false detections

**Optimize Lighting:**
- Well-lit scenes work best
- Avoid strong shadows
- Consistent lighting across frames

**Camera Position:**
- Full body visible
- Avoid extreme angles
- Subject not too close or too far

### 8.3 Data Quality Checks

**Before Processing, Verify:**

✅ **Color Images:**
- Clear, not blurry
- Adequate lighting
- Subject fully visible
- Reasonable resolution (640x480+)

✅ **Depth Images:**
- Valid depth values (not all zeros)
- Reasonable range (0.5m - 5.0m typical)
- Aligned with color images
- No large missing patches

✅ **Frame Count:**
- Color and depth have same number of frames
- Frame numbering is sequential
- No corrupt or missing files

**Use Page 1 to inspect** depth visualization:
- Should show clear depth gradients
- Closer objects = warmer colors
- Check depth values in info panel

### 8.4 Common Pitfalls to Avoid

❌ **Don't skip Page 1**
- Always load through Page 1 first
- Ensures proper standardization
- Creates necessary metadata

❌ **Don't manually edit trial_input/ or trial_output/**
- System manages these automatically
- Manual edits may break syncing
- Original data is source of truth

❌ **Don't process before detecting targets**
- Targets should be detected in Page 1
- Ground plane needed for accurate 3D poses
- Can re-detect targets after processing if needed

❌ **Don't mix Python versions**
- Stick to Python 3.12 (or 3.8-3.11)
- Python 3.13+ will NOT work
- Virtual environment isolates dependencies

❌ **Don't forget to save detections**
- Clicking "Detect Targets" only shows overlays
- Must click "Save Detections" to persist
- Check for green success message

### 8.5 Keyboard Shortcuts

**Currently not implemented**, but you can navigate efficiently:

**Frame Navigation:**
- Use slider for large jumps
- Use ◀ / ▶ buttons for single frame steps
- Use -10 / +10 buttons for moderate jumps

**Tip**: Click in slider track (not on handle) to jump to that position

---

## 9. Troubleshooting

### 9.1 Installation Issues

#### Python Version Error

**Error:**
```
mediapipe requires Python <3.13
```

**Cause**: Using Python 3.13 or higher

**Solution:**
```bash
# Install Python 3.12
brew install python@3.12  # macOS

# Recreate virtual environment
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### tkinter Not Found

**Error:**
```
ModuleNotFoundError: No module named 'tkinter'
```

**Cause**: Python built without tkinter

**Solution:**
- **macOS**: Install Python from python.org (includes tkinter)
- **Linux**: `sudo apt-get install python3-tk`
- **Windows**: Reinstall Python, check "tcl/tk and IDLE"

#### Dependency Build Errors

**Error:**
```
Failed building wheel for <package>
```

**Solution:**

**macOS:**
```bash
xcode-select --install
```

**Linux:**
```bash
sudo apt-get install build-essential python3-dev
```

**Windows:**
- Install Visual Studio Build Tools
- Download from visualstudio.microsoft.com

### 9.2 Runtime Errors

#### Model Not Found

**Error:**
```
Model not found: step0_data_loading/best.pt
```

**Solution:**
1. Verify file exists:
   ```bash
   ls -lh step0_data_loading/best.pt
   ```
2. If missing, copy from backup or obtain from data provider
3. Check file size is ~21 MB
4. Ensure file is not corrupt

#### No Frames Found

**Error:**
```
No frames found in color folder
```

**Causes & Solutions:**

**Wrong folder selected:**
- Browse to parent folder containing trials
- Don't select trial folder directly

**Unsupported structure:**
- Check structure matches Section 6.1
- Ensure `color/` or `Color/` subfolder exists
- Verify image files have correct extensions

**Empty folder:**
- Check folder actually contains images
- Use Finder/Explorer to verify

#### Depth Loading Fails

**Error:**
```
Failed to load depth for frame X
```

**Causes & Solutions:**

**Wrong format:**
- System expects NPY or RAW
- Convert depth to supported format

**Wrong shape:**
- Depth should match color resolution
- Check depth array dimensions

**Corrupt file:**
- Try re-copying from original source
- Check file size is reasonable

### 9.3 Detection Issues

#### No Skeleton Detected

**Problem**: "Human Detected: No" for frames where person is visible

**Solutions:**

1. **Lower confidence threshold** (Page 2 settings)
   - Try 0.3 or 0.4

2. **Change model complexity**
   - Try Complexity 2 for better accuracy

3. **Check frame quality**
   - Ensure good lighting
   - Person should be fully visible
   - Not too close or too far

4. **Verify MediaPipe installation**
   ```bash
   python -c "import mediapipe as mp; mp.solutions.pose.Pose()"
   ```

#### No Targets Detected

**Problem**: "Found 0 target(s)" when cups are visible

**Solutions:**

1. **Check frame selection**
   - Choose frame with clear, unobstructed view
   - Before interaction/occlusion

2. **Verify YOLO model**
   ```bash
   ls -lh step0_data_loading/best.pt
   ```

3. **Check lighting**
   - Targets should be well-lit
   - Avoid harsh shadows

4. **Try different frame**
   - Some frames may work better than others

#### Wrong Arm Selected

**Problem**: System selects wrong arm as pointing arm

**Solutions:**

1. **Manual override** (Page 2 settings)
   - Select "Left Arm" or "Right Arm" radio button
   - Reprocess current frame to verify

2. **Check skeleton quality**
   - Low-quality detection may confuse arm selection
   - Improve detection first (see above)

### 9.4 Performance Issues

#### Very Slow Processing

**Problem**: Takes >5 minutes for 100 frames

**Solutions:**

1. **Reduce model complexity** (Page 2)
   - Use Complexity 0 instead of 1 or 2

2. **Disable subject detection**
   - Uncheck dog/baby if not needed

3. **Check system resources**
   ```bash
   # Check CPU usage
   top  # macOS/Linux
   # Check available RAM
   ```

4. **Close other applications**
   - Free up RAM and CPU

5. **Use SSD**
   - Move data to SSD if on HDD

#### UI Freezes

**Problem**: Application becomes unresponsive

**Solutions:**

1. **Wait for processing**
   - Batch processing blocks UI
   - Progress bar shows it's working

2. **Check console output**
   - Run from terminal to see errors
   - Look for Python exceptions

3. **Restart application**
   - Close and relaunch
   - Don't force quit during processing

#### Memory Errors

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Process fewer frames**
   - Use trimming to reduce frame count

2. **Reduce resolution** (if possible)
   - Resize images before processing

3. **Increase system RAM**
   - Close other applications
   - Upgrade hardware if consistently hitting limits

4. **Process in batches**
   - Manually process subsets of frames

### 9.5 Output Issues

#### Results Not Syncing

**Problem**: Files in `trial_output/` but not in original folder

**Solutions:**

1. **Check metadata**
   - Ensure trial loaded through Page 1 first
   - Metadata links original and processed paths

2. **Verify original path still exists**
   - Don't move/rename original folder during processing

3. **Check permissions**
   - Ensure write access to original folder

4. **Re-run processing**
   - Sometimes sync fails due to timing
   - Re-processing may fix it

#### Corrupt Output Files

**Problem**: JSON files can't be opened or parsed

**Solutions:**

1. **Don't interrupt processing**
   - Let batch processing complete fully

2. **Check disk space**
   ```bash
   df -h
   ```

3. **Re-run processing**
   - Delete corrupt files
   - Process again

#### Missing Visualizations

**Problem**: `fig/` folder empty or missing frames

**Solutions:**

1. **Ensure processing completed**
   - Check progress bar reached 100%
   - Look for success message

2. **Check 3D reconstruction**
   - Requires valid depth data
   - Requires target detections (for ground plane)

3. **Verify matplotlib**
   ```bash
   python -c "import matplotlib; print('OK')"
   ```

---

## 10. Appendix

### 10.1 Glossary of Terms

**RGB-D**: RGB (color) + D (depth) camera data

**MediaPipe**: Google's pose estimation framework (33 keypoints)

**YOLO**: "You Only Look Once" - real-time object detection

**Skeleton**: Set of keypoints and connections representing body pose

**Keypoint**: Specific body landmark (e.g., wrist, shoulder, knee)

**Landmark**: Synonym for keypoint

**Intrinsics**: Camera parameters (focal length, principal point)

**Ground Plane**: Horizontal surface (floor/table) reference

**Pointing Vector**: 3D direction from shoulder through wrist to infinity

**Batch Processing**: Processing all frames automatically

**Trial**: Single recording session/sequence

**Frame**: Single time instant (one image pair)

**Depth Map**: 2D array of distance values (meters)

**NPY**: NumPy binary format for arrays

**RAW**: Uncompressed binary depth data

### 10.2 File Format Specifications

#### NPY Depth Format

**Structure:**
- Binary NumPy array
- Dtype: `float32` (meters) or `uint16` (millimeters)
- Shape: (height, width)
- Byte order: Little-endian (default)

**Loading in Python:**
```python
import numpy as np
depth = np.load('depth/frame_000001.npy')  # Shape: (720, 1280)
```

#### RAW Depth Format

**Structure:**
- Binary unsigned 16-bit integers
- Units: Millimeters
- Byte order: Big-endian or little-endian (auto-detected)
- Size: width × height × 2 bytes

**Loading in Python:**
```python
import numpy as np
depth_raw = np.fromfile('Depth/_Depth_0001.raw', dtype=np.uint16)
depth_raw = depth_raw.reshape((height, width))
depth_m = depth_raw / 1000.0  # Convert to meters
```

### 10.3 Camera Intrinsics Configuration

**Default Auto-Detection** (in `main_ui.py`, lines 203-219):

```python
# Detect camera intrinsics based on resolution
h, w = sample_img.shape[:2]

if w == 640 and h == 480:
    self.page2_ui.fx = self.page2_ui.fy = 615.0
    self.page2_ui.cx = 320.0
    self.page2_ui.cy = 240.0
elif w == 1280 and h == 720:
    self.page2_ui.fx = self.page2_ui.fy = 922.5
    self.page2_ui.cx = 640.0
    self.page2_ui.cy = 360.0
elif w == 1920 and h == 1080:
    self.page2_ui.fx = self.page2_ui.fy = 1383.75
    self.page2_ui.cx = 960.0
    self.page2_ui.cy = 540.0
else:
    # Fallback: approximate from image size
    self.page2_ui.fx = self.page2_ui.fy = w * 0.9
    self.page2_ui.cx = w / 2.0
    self.page2_ui.cy = h / 2.0
```

**To Use Custom Intrinsics:**

1. Edit `main_ui.py`
2. Add new resolution case or modify existing
3. Example for custom 800x600:
   ```python
   elif w == 800 and h == 600:
       self.page2_ui.fx = 750.5  # Your calibrated fx
       self.page2_ui.fy = 749.8  # Your calibrated fy
       self.page2_ui.cx = 400.2  # Your calibrated cx
       self.page2_ui.cy = 300.1  # Your calibrated cy
   ```

### 10.4 Coordinate Systems

**Camera Coordinate System:**
- Origin: Camera optical center
- X-axis: Right (positive direction)
- Y-axis: Down (positive direction)
- Z-axis: Forward/depth (positive direction)
- Units: Meters

**Pixel Coordinate System:**
- Origin: Top-left corner
- X-axis: Right (column index)
- Y-axis: Down (row index)
- Units: Pixels

**Ground-Plane Coordinate System:**
- Origin: Centroid of targets
- X-axis: Horizontal (right)
- Y-axis: Horizontal (forward)
- Z-axis: Vertical (up)
- Units: Meters

**Transformation:**
- Rotation matrix in `ground_plane_transform.json`
- Apply to camera coordinates to get ground-plane coordinates

### 10.5 MediaPipe Keypoint Index

**33 Keypoints (0-indexed):**

```
Face (0-10):
  0: nose
  1: left_eye_inner
  2: left_eye
  3: left_eye_outer
  4: right_eye_inner
  5: right_eye
  6: right_eye_outer
  7: left_ear
  8: right_ear
  9: mouth_left
  10: mouth_right

Upper Body (11-16):
  11: left_shoulder
  12: right_shoulder
  13: left_elbow
  14: right_elbow
  15: left_wrist
  16: right_wrist

Hands (17-22):
  17: left_pinky
  18: right_pinky
  19: left_index
  20: right_index
  21: left_thumb
  22: right_thumb

Lower Body (23-28):
  23: left_hip
  24: right_hip
  25: left_knee
  26: right_knee
  27: left_ankle
  28: right_ankle

Feet (29-32):
  29: left_heel
  30: right_heel
  31: left_foot_index
  32: right_foot_index
```

**For 3D Pointing:**
- Left arm: shoulder (11), elbow (13), wrist (15)
- Right arm: shoulder (12), elbow (14), wrist (16)

### 10.6 Common Error Messages

**Error**: `"No module named 'cv2'"`
- **Meaning**: OpenCV not installed
- **Fix**: `pip install opencv-python`

**Error**: `"CUDA out of memory"`
- **Meaning**: GPU memory exhausted (if using GPU)
- **Fix**: Use CPU instead or reduce batch size

**Error**: `"JSON decode error"`
- **Meaning**: Corrupt JSON file
- **Fix**: Delete file and reprocess

**Error**: `"Permission denied"`
- **Meaning**: No write access to folder
- **Fix**: Check folder permissions, run with appropriate privileges

**Error**: `"Depth shape mismatch"`
- **Meaning**: Depth array doesn't match color image size
- **Fix**: Check depth format, verify data integrity

### 10.7 Performance Benchmarks

**Typical Processing Times** (MacBook Pro M1, 8 GB RAM):

| Task | Frames | Settings | Time |
|------|--------|----------|------|
| Load trial | 100 | - | 2-3 sec |
| Detect targets | 1 | Default | 1-2 sec |
| Process single frame | 1 | Complexity 1 | 0.3 sec |
| Batch process | 100 | Complexity 0 | 30-45 sec |
| Batch process | 100 | Complexity 1 | 60-90 sec |
| Batch process | 100 | Complexity 2 | 120-180 sec |
| Dog detection | 100 | Enabled | +60-90 sec |

**Scaling:**
- Linear with frame count
- Complexity 2 ≈ 2-3x slower than Complexity 0
- Subject detection adds ~50-100% overhead

### 10.8 Frequently Asked Questions

**Q: Can I process videos instead of image sequences?**
A: Not directly. Use ffmpeg to extract frames first:
```bash
ffmpeg -i video.mp4 -start_number 1 color/frame_%06d.png
```

**Q: How accurate is the 3D reconstruction?**
A: Depends on depth sensor quality, typically ±1-5 cm at 1-3m range.

**Q: Can I use this without a depth camera?**
A: Partially. You can detect 2D skeletons, but 3D reconstruction requires depth.

**Q: What's the maximum number of frames I can process?**
A: Limited by RAM and disk space. Tested up to 5,000 frames without issues.

**Q: Can I detect multiple people?**
A: MediaPipe detects one person per frame. For multi-person, process separately.

**Q: How do I cite this system?**
A: See README.md for citation information.

**Q: Is the system open source?**
A: Check LICENSE file and repository README.

**Q: Can I run this on a server without GUI?**
A: Not with main_ui.py. Use command-line scripts in `step2_skeleton_extraction/`.

### 10.9 Additional Resources

**Documentation:**
- README.md - Project overview
- Individual step README files for technical details

**External References:**
- MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose
- YOLOv8: https://docs.ultralytics.com/
- OpenCV: https://docs.opencv.org/

**Data Processing:**
- NumPy: https://numpy.org/doc/
- Pandas: https://pandas.pydata.org/docs/

### 10.10 Version History

**Version 2.0** (Current):
- Unified main_ui.py with 2-page workflow
- Auto-detection of folder structures
- Automatic result syncing
- Dog and baby detection
- Improved 3D visualization
- Comprehensive documentation

**Version 1.0**:
- Initial release
- Separate UIs for each step
- Manual data standardization
- Basic skeleton detection

---

## End of User Guide

For questions or issues not covered in this guide:

1. Check console output for error messages
2. Verify Python version: `python --version`
3. Verify dependencies: `pip list`
4. Check model file exists: `ls -lh step0_data_loading/best.pt`
5. Review this guide's Troubleshooting section (Section 9)

**Thank you for using the Pointing Detection System!**
