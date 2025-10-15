# Step 0: Data Loading - Usage Flow

## Purpose

Standardize different trial folder structures into a unified format for downstream pipeline steps.

## The Problem

Your raw data comes in different structures:
- **Multi-camera**: `trial_1/cam1/color/frame_000001.png`
- **Single-camera**: `1/Color/_Color_0230.png`
- **Depth folders**: Sometimes `Depth/` is actually colored depth, and `Depth_Color/` contains the real .raw depth data

## The Solution: Two-Stage Process

### Stage 1: Browse & Explore (UI)

**Tool**: `ui_data_loader.py`

```bash
python step0_data_loading/ui_data_loader.py sample_raw_data
```

**Purpose**:
- Browse trials without copying/processing
- Inspect individual frames
- Verify data quality
- Select trials for processing

**Auto-detects**:
- ✅ Folder structure
- ✅ True depth folder (prioritizes .raw files)
- ✅ Depth dimensions (720p, 1080p, etc.)
- ✅ File naming conventions

---

### Stage 2: Standardize for Pipeline

**Tool**: `process_trial.py`

```bash
# Process one trial
python step0_data_loading/process_trial.py sample_raw_data/trial_1 cam1

# Process all cameras
python step0_data_loading/process_trial.py sample_raw_data/trial_1 --all
```

**What it does**:
1. Loads all frames from raw folder (handles all variations)
2. Saves to `trial_input/<trial_name>/` in **standardized format**:
   - `color/frame_XXXXXX.png` (always PNG)
   - `depth/frame_XXXXXX.npy` (always .npy in meters)
   - `metadata.json` (trial info)

**Standardized Output**:
```
trial_input/
├── trial_1_cam1/
│   ├── color/
│   │   ├── frame_000001.png    ← Standardized naming
│   │   ├── frame_000002.png
│   │   └── ...
│   ├── depth/
│   │   ├── frame_000001.npy    ← Always .npy, always meters
│   │   ├── frame_000002.npy
│   │   └── ...
│   └── metadata.json
└── 1/
    ├── color/                   ← Same structure
    ├── depth/                   ← Even for single-camera!
    └── metadata.json
```

---

## Workflow

### Option A: Process Everything Upfront

```bash
# 1. Process all trials first
python step0_data_loading/process_trial.py sample_raw_data/trial_1 --all
python step0_data_loading/process_trial.py sample_raw_data/1

# 2. Downstream steps read from trial_input/
# No need to worry about folder structure anymore!
```

### Option B: Browse Then Process

```bash
# 1. Browse data with UI
python step0_data_loading/ui_data_loader.py sample_raw_data

# 2. Process selected trials
python step0_data_loading/process_trial.py sample_raw_data/trial_1 cam1

# 3. Use standardized data in pipeline
```

---

## Key Features

### Automatic Detection

✅ **Depth Folder Detection**
- Finds folder with .raw files (most accurate depth)
- Handles `Depth/` vs `Depth_Color/` confusion
- Example: Detects `Depth_Color` has .raw files → uses that

✅ **Depth Shape Auto-Detection**
- Reads .raw file size
- Matches to common resolutions (480p, 720p, 1080p)
- No manual configuration needed

✅ **Depth Statistics**
```
✅ Loaded depth ((720, 1280)): .../Depth_Color/_Depth_Color_0230.raw
   Depth stats: median=3.593m, mean=3.587m, range=[0.710, 65.535]m
```

---

## Downstream Pipeline Benefits

After processing with `process_trial.py`, your downstream steps can assume:

1. **Consistent naming**: Always `frame_XXXXXX.png/npy`
2. **Consistent format**: Color=PNG, Depth=.npy
3. **Consistent units**: Depth always in meters
4. **Consistent structure**: Same folder layout for all trials

Example pipeline code becomes simple:
```python
# No need for structure detection!
color = cv2.imread(f"trial_input/trial_1_cam1/color/frame_{i:06d}.png")
depth = np.load(f"trial_input/trial_1_cam1/depth/frame_{i:06d}.npy")
# Depth is guaranteed to be in meters, float32
```

---

## Summary

| Tool | Purpose | Output |
|------|---------|--------|
| `ui_data_loader.py` | Browse & explore | None (read-only) |
| `data_manager.py` | Discover trials | Config file |
| `process_trial.py` | Standardize for pipeline | `trial_input/` |
| `load_trial_data_flexible.py` | Low-level loader | (internal use) |

**Recommendation**:
1. Use `process_trial.py` to create standardized `trial_input/`
2. Downstream pipeline reads from `trial_input/`
3. No more folder structure headaches!
