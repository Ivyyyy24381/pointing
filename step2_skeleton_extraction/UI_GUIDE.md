# Step 2: Skeleton Extraction UI Guide

## Overview

The Skeleton Extraction UI provides an interactive interface for:
- Loading and visualizing trial data
- Real-time skeleton detection with MediaPipe
- Batch processing multiple frames
- Visualizing 2D/3D skeletons and arm vectors
- Saving results in legacy-compatible format

## Launching the UI

```bash
# Using point_production environment
/opt/anaconda3/envs/point_production/bin/python \
    step2_skeleton_extraction/ui_skeleton_extractor.py

# Or use the pipeline runner
./run_pipeline.sh 2
```

## UI Layout

### Left Panel: Controls

#### 1. Load Trial
- **Select Trial Folder**: Browse to select a trial folder (e.g., `trial_input/trial_1/cam1/`)
- Requirements:
  - Must contain `color/` folder with frame images
  - Optional `depth/` folder for 3D extraction
- The UI will automatically:
  - Detect number of frames
  - Detect camera intrinsics from image resolution
  - Load first frame

#### 2. Detector Settings
Configure MediaPipe detection parameters:

**Model Complexity:**
- `0 (Fast)`: Fastest processing, lower accuracy
- `1 (Balanced)`: ⭐ Recommended - good balance
- `2 (Accurate)`: Highest accuracy, slower processing

**Detection Confidence:** (0.0 - 1.0)
- Lower values: Detect more poses (may include false positives)
- Higher values: Only high-confidence detections
- Default: 0.50

> **Tip:** Changes to settings automatically reinitialize the detector

#### 3. Frame Navigation
Navigate through frames:

- `⏮` First frame
- `◀` Previous frame
- `▶` Play/Pause (auto-play at 10 fps)
- `▶▶` Next frame
- `⏭` Last frame
- **Slider**: Jump to any frame directly
- **Frame counter**: Shows current/total frames

#### 4. Processing

**Process Current Frame:**
- Detect skeleton in currently displayed frame
- Updates visualization immediately
- Fast for single-frame analysis

**Process All Frames:**
- Batch process entire trial
- Shows progress bar
- Takes several minutes for 100+ frames
- Results stored in memory

**Save Results:**
- Export to JSON file
- Choose output directory
- Creates:
  - `skeleton_2d.json` - Full skeleton data
  - `skeleton_summary.txt` - Processing statistics

#### 5. Detection Info
Shows details about current frame:
- Number of landmarks detected
- Pointing arm (left/right/auto)
- Whether depth data is available
- 3D landmarks status
- Arm vectors (shoulder→wrist, elbow→wrist, etc.)
- Wrist 3D position in camera frame

### Right Panel: Visualization

#### Display Options
Toggle visualization layers:

- ☑️ **Show Skeleton**: Draw 2D skeleton overlay
  - Green lines: bone connections
  - Red dots: joint landmarks

- ☑️ **Show Arm Vectors**: Draw pointing arm vectors
  - 🟢 Green: shoulder→wrist
  - 🟠 Orange: elbow→wrist
  - 🟣 Magenta: eye→wrist
  - 🔵 Cyan: nose→wrist

- ☐ **Show Depth**: Display depth image instead of color
  - Uses JET colormap (blue=near, red=far)
  - Skeleton still overlaid on depth view

#### Canvas
- Main display area for images and skeleton
- Auto-resizes to fit window
- Maintains aspect ratio

## Workflow Examples

### 1. Quick Single-Frame Test

1. Click **Select Trial Folder** → Choose `trial_input/trial_1/cam1/`
2. Click **Process Current Frame**
3. View skeleton overlay on image
4. Check **Detection Info** panel for arm vectors
5. Use **Show Arm Vectors** to visualize pointing direction

### 2. Batch Process Entire Trial

1. Load trial folder
2. Adjust **Detector Settings** if needed
3. Click **Process All Frames**
4. Wait for progress bar to complete
5. Navigate through frames to review results
6. Click **Save Results** → Choose `trial_output/trial_1/cam1/`

### 3. Compare Different Settings

1. Load trial and process current frame
2. Note detection quality
3. Change **Model Complexity** to 2
4. Process same frame again
5. Compare accuracy vs speed

### 4. Visualize Depth + 3D

1. Load trial with depth data
2. Process current frame
3. Enable **Show Depth** checkbox
4. View skeleton overlaid on depth
5. Check **Detection Info** for 3D wrist location

### 5. Review Batch Results

1. After batch processing
2. Use **Play** button to animate through frames
3. Pause on specific frames to inspect
4. Use slider to jump to frames of interest

## Understanding the Output

### Visualization Colors

**Skeleton:**
- 🟢 Green lines: Bone connections
- 🔴 Red dots: Joint landmarks

**Arm Vectors (from wrist):**
- 🟢 shoulder→wrist: Primary pointing vector
- 🟠 elbow→wrist: Forearm direction
- 🟣 eye→wrist: Eye gaze alignment
- 🔵 nose→wrist: Head alignment

### Detection Info Panel

```
Frame 45
==============================

Landmarks detected: 33
Pointing arm: right
Has depth: True

3D Landmarks: ✓

Arm Vectors:
  shoulder_to_wrist:
    [-0.052, 0.701, 0.711]
  elbow_to_wrist:
    [-0.089, 0.842, 0.532]
  ...

Wrist 3D Location:
  X: -0.177 m
  Y: -0.733 m
  Z: 4.319 m
```

**Interpretation:**
- **Landmarks detected**: Always 33 for MediaPipe Pose
- **Pointing arm**: Which arm is extended/pointing
- **Arm Vectors**: Normalized unit vectors (length = 1.0)
- **Wrist 3D Location**: Position in camera frame (meters)
  - X: left(-) / right(+)
  - Y: down(-) / up(+)
  - Z: depth from camera

## Saved Output Format

### skeleton_2d.json
```json
{
  "frame_000001": {
    "frame": 1,
    "landmarks_2d": [[x, y, visibility], ...],
    "landmarks_3d": [[x, y, z], ...],
    "keypoint_names": ["nose", "left_eye", ...],
    "metadata": {
      "pointing_arm": "right",
      "model": "mediapipe_pose",
      "has_depth": true
    },
    "arm_vectors": {
      "shoulder_to_wrist": [nx, ny, nz],
      "elbow_to_wrist": [nx, ny, nz],
      "eye_to_wrist": [nx, ny, nz],
      "nose_to_wrist": [nx, ny, nz],
      "wrist_location": [x, y, z]
    }
  },
  "frame_000002": { ... }
}
```

### skeleton_summary.txt
```
Skeleton Extraction Summary
========================================
Trial: cam1
Total frames processed: 89
Detector: MediaPipe Pose
Model complexity: 1
Detection confidence: 0.50

Pointing arm distribution:
  auto: 45 frames (50.6%)
  left: 20 frames (22.5%)
  right: 24 frames (27.0%)
```

## Tips & Best Practices

### For Accuracy
- Use **Model Complexity 1 or 2** for better accuracy
- Ensure good lighting in source images
- Check that subject is fully visible in frame
- Higher **Detection Confidence** reduces false positives

### For Speed
- Use **Model Complexity 0** for real-time processing
- Process single frames first to test settings
- Batch process overnight for large trials

### For 3D Data
- Ensure depth folder exists and matches color frames
- Check wrist location values are reasonable (not 0,0,0)
- Invalid depth points will show as (0, 0, 0)

### Verification
- Use **Play** to quickly scan through results
- Look for consistent skeleton tracking
- Check arm vectors point in expected direction
- Verify wrist location depth (Z) is reasonable

## Troubleshooting

### "No pose detected"
- Subject may be too far from camera
- Poor lighting or occlusion
- Lower **Detection Confidence** threshold
- Try **Model Complexity 2** for harder cases

### Missing 3D landmarks
- Check if depth folder exists
- Verify depth files match frame numbers
- Depth image may have invalid values (0, NaN, inf)

### Slow performance
- Reduce **Model Complexity** to 0
- Process subset of frames first
- Close other applications
- Use batch processing instead of real-time

### Inaccurate arm vectors
- Ensure depth quality is good
- Check camera intrinsics detected correctly
- May need proper camera calibration for precision

## Keyboard Shortcuts

Currently none - use mouse/buttons. Future versions may add:
- `Space`: Play/Pause
- `←/→`: Previous/Next frame
- `P`: Process current frame
- `S`: Save results

## Advanced Usage

### Custom Camera Intrinsics

If you have calibrated camera parameters, modify the `load_trial()` method:

```python
# In ui_skeleton_extractor.py, after loading trial:
self.fx = 618.5  # Your calibrated fx
self.fy = 618.3  # Your calibrated fy
self.cx = 321.2  # Your calibrated cx
self.cy = 239.8  # Your calibrated cy
```

### Export for Step 4

Results are already in the correct format for Step 4 (Pointing Calculation):

```bash
# Save to trial_output
# Then run Step 4:
python step4_calculation/pointing_calculator.py \
    --skeleton trial_output/trial_1/cam1/skeleton_2d.json \
    --targets trial_output/trial_1/cam1/target_detections.json
```

## Feature Highlights

✅ **Real-time detection**: Process frames on demand
✅ **Batch processing**: Process entire trials automatically
✅ **3D visualization**: Shows skeleton with depth
✅ **Arm vectors**: Visualize pointing direction
✅ **Legacy compatible**: Output matches original format
✅ **Progress tracking**: Visual progress bar and status
✅ **Frame navigation**: Scrub through results easily
✅ **Flexible settings**: Adjust accuracy/speed tradeoff

## Future Enhancements

Potential additions:
- [ ] Export visualization video
- [ ] Side-by-side comparison of settings
- [ ] Manual arm selection override
- [ ] Confidence filtering UI
- [ ] Export to CSV for analysis
- [ ] 3D viewer for arm vectors
- [ ] Ground plane visualization
