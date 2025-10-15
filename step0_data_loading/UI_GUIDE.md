# Data Loading & Target Detection UI Guide

Complete guide for using the data loading and target detection GUI.

## Launching the UI

```bash
# From project root
python step0_data_loading/ui_data_loader.py sample_raw_data

# Or launch and browse to folder
python step0_data_loading/ui_data_loader.py
```

---

## Step-by-Step Usage

### 1. Loading Data

#### Option A: Launch with Folder Path
```bash
python step0_data_loading/ui_data_loader.py sample_raw_data
```
The UI will automatically discover all trials in the folder.

#### Option B: Browse to Folder
1. Launch UI: `python step0_data_loading/ui_data_loader.py`
2. Click **"Browse..."** button
3. Select folder containing trials (e.g., `sample_raw_data`)
4. UI will auto-discover all trials

#### What Happens During Loading
- ✅ Scans folder for trial subdirectories
- ✅ Auto-detects folder structure (multi-camera vs single-camera)
- ✅ Identifies available cameras for each trial
- ✅ Lists all trials in dropdown menu

---

### 2. Selecting Trial and Camera

#### Select Trial
1. Click **"Trial"** dropdown
2. Choose trial (e.g., `trial_1` or `1`)

#### Select Camera
1. Click **"Camera"** dropdown
2. For multi-camera: Choose camera (e.g., `cam1`, `cam2`)
3. For single-camera: Will auto-select `None`

#### What Happens After Selection
- ✅ Processes trial to standardized `trial_input/` folder
- ✅ Converts all frames to PNG (color) and .npy (depth, meters)
- ✅ Displays frame count
- ✅ Loads first frame automatically

**Status Bar Message**: `✅ Frame 1 from trial_input/ (1/89)`

---

### 3. Browsing Frames

#### Navigation Controls
- **Slider**: Drag to jump to any frame
- **◀ Prev**: Previous frame
- **Next ▶**: Next frame
- **◀◀ -10**: Jump back 10 frames
- **+10 ▶▶**: Jump forward 10 frames
- **🔄 Reload Frame**: Reload current frame

#### Display Windows
- **Left Panel**: Color image (RGB)
- **Right Panel**: Depth image (colorized)

#### Frame Information
The info box shows:
```
📂 Source: trial_input/ (standardized)
Trial: trial_1
Camera: cam1
Frame: 31
Color: (480, 640, 3), PNG
Depth: (480, 640), .npy, [0.123-3.456]m
```

---

### 4. Detecting Targets

#### Running Detection

1. **Load a frame** (browse to desired frame)
2. Click **"🎯 Detect Targets"** button
3. Wait for YOLO model to load (first time only)
4. Detections will appear on both color and depth images

#### What Gets Displayed

**On Color Image:**
- Green bounding boxes around detected cups
- Label with confidence (e.g., "cup 0.95")
- Green dot at center

**On Depth Image:**
- Green bounding boxes
- Label with average depth (e.g., "cup avg:0.850m")
- Median depth below (e.g., "med:0.845m")
- Green dot at center

#### Detection Info
Status shows: `✅ Detected 4 target(s)`
Label shows: `Found 4 target(s)` in green

---

### 5. Saving Detections

#### Save to JSON

1. After running detection, click **"💾 Save Detections"**
2. File is saved to: `trial_output/<trial>/<camera>/target_detections.json`

#### Output Format

```json
[
  {
    "bbox": [897, 525, 964, 575],
    "center_px": [930, 550],
    "avg_depth_m": 2.682,
    "x": 0.834,
    "y": 0.515,
    "z": 2.682,
    "label": "target_1"
  },
  {
    "bbox": [727, 496, 793, 547],
    "center_px": [760, 521],
    "avg_depth_m": 2.841,
    "x": 0.36,
    "y": 0.457,
    "z": 2.841,
    "label": "target_2"
  }
]
```

#### Field Descriptions

| Field | Description |
|-------|-------------|
| `bbox` | [x1, y1, x2, y2] bounding box in pixels |
| `center_px` | [x, y] center point in pixels |
| `avg_depth_m` | Average depth within bbox (meters) |
| `x`, `y`, `z` | 3D position in camera frame (meters) |
| `label` | `target_1` to `target_4` (right to left) |

**Important**: Targets are sorted **right to left** by x-coordinate:
- `target_1` = rightmost cup
- `target_4` = leftmost cup

---

### 6. Clearing Detections

Click **"🗑️ Clear"** to remove detection overlay and start fresh.

---

## Menu Options

### File Menu

- **Open Folder...**: Browse to new data folder
- **Save Config**: Save current trial configuration to JSON
- **Load Config**: Load previously saved configuration
- **Exit**: Close application

---

## Keyboard Shortcuts

None currently implemented. Use mouse/buttons for all interactions.

---

## Output Files

### trial_input/ (Auto-Generated)

Standardized format, created automatically when you select a trial:

```
trial_input/
└── trial_1_cam1/
    ├── color/
    │   ├── frame_000001.png
    │   ├── frame_000002.png
    │   └── ...
    └── depth/
        ├── frame_000001.npy
        ├── frame_000002.npy
        └── ...
```

### trial_output/ (Manual Save)

Created when you click "Save Detections":

```
trial_output/
└── trial_1/
    └── cam1/
        └── target_detections.json
```

---

## Troubleshooting

### "Model not found" Error

**Problem**: YOLO model `best.pt` not found

**Solution**:
1. Check that `step0_data_loading/best.pt` exists
2. If not, copy from `step1_calibration_process/target_detection/automatic_mode/best.pt`

### "No frames found" Warning

**Problem**: Trial folder structure not recognized

**Solution**:
1. Check folder structure matches supported formats (see README.md)
2. Ensure color/depth folders exist with image files

### Depth Image All Black

**Problem**: No valid depth values in frame

**Solution**:
1. Check that depth files exist and are not corrupted
2. Try different frame (some frames may have invalid depth)

### Detection Shows No Targets

**Problem**: No cups detected in frame

**Possible Causes**:
1. No cups visible in frame
2. Cups too small/far away
3. Poor lighting/image quality
4. Confidence threshold too high

**Solution**:
- Try different frames with clearer cup visibility
- Adjust confidence threshold in code if needed (default: 0.5)

---

## Tips and Best Practices

### ✅ DO
- Browse through multiple frames to find clear views of targets
- Verify detections visually before saving
- Check both color and depth views to ensure accurate depth
- Save detections for frames with all targets visible

### ❌ DON'T
- Don't save detections with missing targets
- Don't use frames with heavily occluded cups
- Don't rely on detections with very low confidence (<0.5)

---

## Example Workflow

1. **Launch**: `python step0_data_loading/ui_data_loader.py sample_raw_data`
2. **Select**: Trial `trial_1`, Camera `cam1`
3. **Wait**: System processes to `trial_input/` (~10 seconds)
4. **Browse**: Use slider to find frame with clear cup view
5. **Detect**: Click "🎯 Detect Targets"
6. **Verify**: Check green boxes on both color and depth
7. **Save**: Click "💾 Save Detections"
8. **Result**: `trial_output/trial_1/cam1/target_detections.json` created

---

## Technical Details

### Camera Intrinsics (Auto-Detected)

| Resolution | fx/fy | cx | cy |
|------------|-------|----|----|
| 640×480    | 615.0 | 320 | 240 |
| 1280×720   | 922.5 | 640 | 360 |
| 1920×1080  | 1383.75 | 960 | 540 |

### 3D Coordinate Frame

- **X-axis**: Right (positive = right side of image)
- **Y-axis**: Down (positive = bottom of image)
- **Z-axis**: Forward (positive = away from camera, depth)
- **Units**: Meters
- **Origin**: Camera optical center

### Detection Model

- **Model**: YOLOv8n (nano)
- **Training**: Custom dataset of cups
- **Input**: 640×640 RGB image
- **Output**: Bounding boxes with confidence scores
- **Confidence Threshold**: 0.5

---

## Next Steps

After saving target detections, you can proceed to:
- **Step 1**: Camera calibration using AprilTags
- **Step 2**: Skeleton extraction (human/dog pose)
- **Step 3**: Subject extraction and segmentation
- **Step 4**: Pointing calculation and analysis
