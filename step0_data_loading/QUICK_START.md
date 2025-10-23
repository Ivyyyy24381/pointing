# Quick Start: Data Loading

## UI Part 1: Load and Browse Data

### Launch GUI

```bash
python step0_data_loading/ui_data_loader.py sample_raw_data
```

**Features:**
- 📁 Browse folders with multiple trials
- 🎥 Auto-detect trial structure and cameras
- 🖼️ View color and depth side-by-side
- ⏯️ Navigate frames with slider
- 💾 Save/load configurations

---

## Command Line Usage

### 1. Discover Trials

```bash
python step0_data_loading/data_manager.py sample_raw_data
```

**Output:**
- Discovers all trials in folder
- Detects structure (multi-camera vs single-camera)
- Saves `data_config.json`
- Shows interactive demo

---

### 2. Process Entire Trial

```bash
# Multi-camera
python step0_data_loading/process_trial.py sample_raw_data/trial_1 cam1

# Single-camera
python step0_data_loading/process_trial.py sample_raw_data/1

# All cameras
python step0_data_loading/process_trial.py sample_raw_data/trial_1 --all
```

**Output:** Saves to `trial_input/<trial_name>/` with standardized format

---

### 3. Load Individual Frame

```bash
# Multi-camera
python step0_data_loading/load_trial_data_flexible.py sample_raw_data/trial_1 cam1 31

# Single-camera
python step0_data_loading/load_trial_data_flexible.py sample_raw_data/1 230
```

---

## Python API

### Quick Example

```python
from step0_data_loading.data_manager import DataManager

# Initialize
dm = DataManager("sample_raw_data")

# List trials
print(dm.list_trials())  # ['1', 'calib_1', 'trial_1']

# Load a frame
color, depth = dm.load_frame("trial_1", "cam1", 31)

# Find available frames
frames = dm.find_available_frames("trial_1", "cam1")
print(f"Found {len(frames)} frames")
```

---

## File Structure

```
step0_data_loading/
├── ui_data_loader.py              # GUI (UI Part 1) ⭐
├── data_manager.py                # High-level manager ⭐
├── process_trial.py               # Batch processor
├── load_trial_data_flexible.py   # Flexible loader
└── README.md                      # Full documentation
```

---

## Next Steps

After loading data with UI Part 1:

1. **Step 1**: Camera Calibration (optional)
2. **Step 2**: Target Detection
3. **Step 3**: Skeleton Extraction
4. **Step 4**: Subject Extraction
5. **Step 5**: Pointing Calculation

---

## Keyboard Shortcuts (GUI)

- `Ctrl+O` - Open folder
- `Arrow Keys` - Navigate frames (when slider focused)
- `Space` - Reload frame

---

## Tested With

✅ `sample_raw_data/trial_1/` - Multi-camera (3 cams, 89 frames each)
✅ `sample_raw_data/calib_1/` - Multi-camera (3 cams, ~56 frames each)
✅ `sample_raw_data/1/` - Single-camera (166 frames)
