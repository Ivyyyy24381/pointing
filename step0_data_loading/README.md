# Step 0: Data Loading & Target Detection

Comprehensive data loading and target detection system with GUI for browsing trials, detecting targets, and exporting standardized data.

## Features

- ✅ **Flexible Data Loading**: Auto-detects multi-camera and single-camera folder structures
- ✅ **Auto-Standardization**: Converts all trials to unified format in `trial_input/`
- ✅ **Target Detection**: YOLO-based automatic cup detection with 3D localization
- ✅ **Interactive GUI**: Browse trials, visualize depth, detect targets
- ✅ **Point Cloud Generation**: Create 3D point clouds from RGB-D data
- ✅ **Interactive Depth Viewer**: Hover to see depth values

## Quick Start

### GUI Mode (Recommended)

```bash
# Launch data loading UI
python step0_data_loading/ui_data_loader.py sample_raw_data

# Or launch without arguments and browse to folder
python step0_data_loading/ui_data_loader.py
```

### Command Line Mode

```bash
# Process entire trial (multi-camera)
python step0_data_loading/process_trial.py sample_raw_data/trial_1 cam1

# Process entire trial (single-camera)
python step0_data_loading/process_trial.py sample_raw_data/1

# Load and view specific frame
python step0_data_loading/view_depth.py trial_1 cam1 100

# Create point cloud
python step0_data_loading/test_point_cloud.py trial_1 cam1 100
```

## Supported Folder Structures

### Multi-Camera Structure
```
trial_1/
├── cam1/
│   ├── color/
│   │   └── frame_000001.png
│   └── depth/
│       └── frame_000001.npy
└── cam2/
    ├── color/
    └── depth/
```

### Single-Camera Structure
```
1/
├── Color/
│   └── _Color_0001.png
├── Depth/  (or Depth_Color)
│   └── _Depth_0001.raw
└── Depth_Color/
    └── _Depth_Color_0001.png
```

**Note**: System auto-detects which folder contains true depth (prioritizes `.raw` files).

## Output Structure

All processed data is saved to `trial_input/` in standardized format:

```
trial_input/
└── trial_1_cam1/
    ├── color/
    │   └── frame_000001.png  (PNG format)
    └── depth/
        └── frame_000001.npy  (NumPy array, meters)
```

Target detections are saved to `trial_output/`:

```
trial_output/
└── trial_1/
    └── cam1/
        └── target_detections.json
```

## Target Detection

### Detection Format (JSON)

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
  }
]
```

Targets are labeled `target_1` to `target_4` from **right to left** by x-coordinate.

### Model

- **Location**: `step0_data_loading/best.pt`
- **Type**: YOLOv8 trained on cups
- **Confidence**: 0.5 threshold (configurable)

## Key Files

### GUI & Visualization
- `ui_data_loader.py` - Main GUI for loading, browsing, and target detection
- `view_depth.py` - Interactive depth viewer with hover tooltips
- `test_point_cloud.py` - Point cloud generation and visualization

### Core Loading
- `load_trial_data_flexible.py` - Flexible loader for all folder structures
- `process_trial.py` - Batch processor for entire trials
- `trial_input_manager.py` - Manager for standardized `trial_input/` data
- `data_manager.py` - Multi-trial discovery and management

### Target Detection
- `target_detector.py` - YOLO wrapper with 3D localization
- `best.pt` - YOLOv8 model weights

### Utilities
- `../utils/point_cloud_utils.py` - Point cloud creation utilities

## Workflow

1. **Load Data**: Use GUI or command line to process raw trial data
2. **Auto-Standardize**: System converts to `trial_input/` format
3. **Detect Targets**: Click "Detect Targets" button in GUI
4. **Verify**: Visualize detections on color and depth images
5. **Export**: Save to `trial_output/trial_name/camera/target_detections.json`

## Camera Intrinsics (Auto-Detected)

| Resolution | fx/fy | cx | cy |
|------------|-------|----|----|
| 640×480    | 615.0 | 320 | 240 |
| 1280×720   | 922.5 | 640 | 360 |
| 1920×1080  | 1383.75 | 960 | 540 |

For other resolutions, intrinsics are estimated as `fx = fy = width * 0.96`.

## Test Results

✅ Tested with `sample_raw_data/trial_1` (89 frames, multi-camera)
✅ Tested with `sample_raw_data/1` (166 frames, single-camera)
✅ Depth auto-detection: 480p, 720p, 1080p
✅ Target detection: 3D localization with depth

## Documentation

- `QUICK_START.md` - Quick start guide
- `INTEGRATED_WORKFLOW.md` - Complete workflow documentation
- `UI_GUIDE.md` - User interface guide
