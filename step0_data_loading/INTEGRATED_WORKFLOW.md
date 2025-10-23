# Integrated Workflow: Auto-Standardizing Data Loader

## Overview

The UI now **automatically standardizes** all data to `trial_input/` as you browse. This ensures:

✅ **Downstream tasks** only read from `trial_input/`
✅ **Naming independence** - always `frame_XXXXXX.png` / `frame_XXXXXX.npy`
✅ **Format consistency** - PNG for color, .npy for depth (in meters)
✅ **No manual processing** - happens automatically when you switch trials

---

## How It Works

### 1. Launch UI

```bash
python step0_data_loading/ui_data_loader.py sample_raw_data
```

### 2. UI Workflow (Auto-Standardization)

```
User selects trial + camera
          ↓
UI checks: Does trial_input/<trial_camera>/ exist?
          ↓
     NO ──────→ Auto-process trial to trial_input/
          ↓       (Standardize naming, convert to .npy)
     YES ──────→ Skip processing (use existing)
          ↓
Load frames from trial_input/ (standardized location)
          ↓
Display in UI
```

### 3. Result

**`trial_input/` folder structure:**
```
trial_input/
├── trial_1_cam1/                 ← Auto-created when you browse
│   ├── color/
│   │   ├── frame_000001.png      ← Standardized PNG
│   │   ├── frame_000002.png
│   │   └── ...
│   ├── depth/
│   │   ├── frame_000001.npy      ← Standardized .npy (meters)
│   │   ├── frame_000002.npy
│   │   └── ...
│   └── metadata.json             ← Processing info
└── 1/                             ← Single-camera trial
    ├── color/
    ├── depth/
    └── metadata.json
```

---

## Downstream Pipeline Integration

### All Tasks Read from `trial_input/`

**Example: Step 2 - Target Detection**
```python
# NO need for structure detection!
# NO need for format handling!
# Just read from trial_input/

import numpy as np
import cv2

trial_path = "trial_input/trial_1_cam1"

# Load color
color = cv2.imread(f"{trial_path}/color/frame_000031.png")

# Load depth
depth = np.load(f"{trial_path}/depth/frame_000031.npy")

# Depth is GUARANTEED to be:
# - float32
# - in meters
# - same shape as color (or original depth shape)
```

**Example: Step 3 - Skeleton Extraction**
```python
import glob

trial_path = "trial_input/trial_1_cam1"

# Find all frames
frame_files = sorted(glob.glob(f"{trial_path}/color/frame_*.png"))

for frame_file in frame_files:
    # Extract frame number from standardized naming
    frame_num = int(frame_file.split("_")[-1].split(".")[0])

    # Load
    color = cv2.imread(f"{trial_path}/color/frame_{frame_num:06d}.png")
    depth = np.load(f"{trial_path}/depth/frame_{frame_num:06d}.npy")

    # Process...
```

---

## Benefits

### 1. **Naming Independence**

Your original data can have ANY naming:
- `_Color_0230.png` → `frame_000230.png`
- `_Depth_Color_0230.raw` → `frame_000230.npy`
- `frame_000001.png` → `frame_000001.png` (unchanged)

### 2. **Format Consistency**

Original depth formats:
- `.raw` (1280x720, uint16) → `.npy` (720, 1280), float32, meters
- `.png` (colored depth) → `.npy` float32, meters
- `.npy` (mm) → `.npy` meters

### 3. **Zero Configuration**

Downstream tasks don't need:
- ❌ `detect_folder_structure()`
- ❌ `find_depth_file()`
- ❌ Format detection
- ❌ Unit conversion

Just:
```python
color = cv2.imread("trial_input/trial/color/frame_XXXXXX.png")
depth = np.load("trial_input/trial/depth/frame_XXXXXX.npy")
```

---

## UI Features

### Status Bar Messages

- `"Auto-processing to trial_input/..."` - Processing trial
- `"✅ Ready: 89 frames from trial_input/"` - Ready to use
- `"✅ Frame 31 from trial_input/ (31/89)"` - Viewing frame

### Info Panel Shows

```
📂 Source: trial_input/ (standardized)
Trial: trial_1
Camera: cam1
Frame: 31
Color: (480, 640, 3), PNG
Depth: (480, 640), .npy, [1.918-20.404]m
```

### Auto-Skip Reprocessing

If `trial_input/trial_1_cam1/` already exists:
- ✅ Skips processing (fast)
- Uses existing standardized data
- Prints: `"✅ Trial already processed: trial_input/trial_1_cam1"`

---

## Workflow Comparison

### ❌ OLD: Manual Processing

```bash
# Step 1: Process manually
python process_trial.py sample_raw_data/trial_1 cam1

# Step 2: Remember to read from trial_input/
# Step 3: Hope downstream tasks do it correctly
```

### ✅ NEW: Auto-Integrated

```bash
# Step 1: Just browse with UI
python ui_data_loader.py sample_raw_data

# That's it! trial_input/ is auto-created as you browse
# Downstream tasks just read from trial_input/
```

---

## Advanced: TrialInputManager

For programmatic access to standardized data:

```python
from trial_input_manager import TrialInputManager

manager = TrialInputManager("trial_input")

# Ensure trial is processed
manager.ensure_trial_processed("sample_raw_data/trial_1", "cam1")

# Load frame
color, depth = manager.load_frame("trial_1", "cam1", 31)

# Find frames
frames = manager.find_available_frames("trial_1", "cam1")

# List all trials
trials = manager.list_available_trials()
```

---

## Summary

| Component | Purpose | Output |
|-----------|---------|--------|
| UI (`ui_data_loader.py`) | Browse + Auto-standardize | `trial_input/` |
| TrialInputManager | Programmatic access | Load from `trial_input/` |
| Downstream Tasks | Process data | Read from `trial_input/` |

**Key Point**: Everything flows through `trial_input/` - one source of truth, standardized format! 🎯

---
