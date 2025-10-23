# Step 2: Skeleton Extraction

Extract 2D and 3D skeleton keypoints from RGB-D images using MediaPipe (humans) or DeepLabCut (dogs).

## Features

- ✅ **MediaPipe Pose**: 33-keypoint human pose detection
- ✅ **3D Landmarks**: Depth-based 3D coordinate extraction
- ✅ **Arm Vectors**: Pointing direction vectors (shoulder→wrist, etc.)
- ✅ **Interactive UI**: Real-time visualization and batch processing
- ✅ **Batch Processing**: Process entire trial folders
- ✅ **Pointing Arm Detection**: Automatic left/right arm detection
- ✅ **Video Support**: Process videos or image sequences
- 🚧 **DeepLabCut**: Dog pose detection (TODO)

## Quick Start

### Interactive UI (Recommended)

```bash
# Launch UI for interactive processing
/opt/anaconda3/envs/point_production/bin/python \
    step2_skeleton_extraction/ui_skeleton_extractor.py
```

**Features:**
- Load and visualize trials
- Real-time skeleton detection
- Adjust detector settings on-the-fly
- View 2D/3D skeletons and arm vectors
- Batch process with progress tracking
- Save results in legacy format

See [UI_GUIDE.md](UI_GUIDE.md) for detailed instructions.

### Process Single Trial

```bash
# Process specific trial
python step2_skeleton_extraction/batch_processor.py \
  --trial trial_input/trial_1/cam1

# Output: trial_output/trial_1/cam1/skeleton_2d.json
```

### Process All Trials

```bash
# Process all trials in trial_input/
python step2_skeleton_extraction/batch_processor.py

# Or specify custom paths
python step2_skeleton_extraction/batch_processor.py \
  --trial_input trial_input \
  --output trial_output \
  --detector mediapipe
```

### Use in Python

```python
from step2_skeleton_extraction import MediaPipeHumanDetector

# Initialize detector
detector = MediaPipeHumanDetector(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process images
results = detector.detect_image_folder("trial_input/trial_1/cam1/color")

# Save results
detector.save_results(results, "output/skeleton_2d.json")

# Or process video
results = detector.detect_video("trial_1/Color.mp4")
```

## Output Format

### skeleton_2d.json

```json
{
  "frame_000001": {
    "frame": 1,
    "landmarks_2d": [
      [x, y, visibility],
      ...
    ],
    "landmarks_3d": [
      [x, y, z],
      ...
    ],
    "keypoint_names": ["nose", "left_eye", ...],
    "metadata": {
      "pointing_arm": "right",
      "model": "mediapipe_pose",
      "num_landmarks": 33
    }
  }
}
```

### Landmark Order (MediaPipe Pose)

33 landmarks in this order:
1. nose
2. left_eye_inner, left_eye, left_eye_outer
3. right_eye_inner, right_eye, right_eye_outer
4. left_ear, right_ear
5. mouth_left, mouth_right
6. left_shoulder, right_shoulder
7. left_elbow, right_elbow
8. **left_wrist, right_wrist** (indices 15, 16 - used for pointing)
9. left_pinky, right_pinky
10. left_index, right_index
11. left_thumb, right_thumb
12. left_hip, right_hip
13. left_knee, right_knee
14. left_ankle, right_ankle
15. left_heel, right_heel
16. left_foot_index, right_foot_index

## Pointing Arm Detection

The detector automatically determines which arm is pointing based on wrist height:

- **"left"**: Left wrist above left shoulder
- **"right"**: Right wrist above right shoulder
- **"auto"**: Both extended or neither extended

Override in GUI if needed.

## Dependencies

```bash
pip install mediapipe opencv-python numpy
```

Optional for DeepLabCut:
```bash
pip install deeplabcut
```

## Files

- `skeleton_base.py` - Abstract base class and result dataclass
- `mediapipe_human.py` - MediaPipe Pose detector
- `batch_processor.py` - Batch processing script
- `__init__.py` - Module exports

## Workflow

1. **Input**: `trial_input/trial_1/cam1/color/*.png`
2. **Process**: MediaPipe Pose detection on each frame
3. **Output**: `trial_output/trial_1/cam1/skeleton_2d.json`

## Troubleshooting

### "MediaPipe not installed"
```bash
pip install mediapipe
```

### No skeletons detected
- Check that images contain visible humans
- Lower `min_detection_confidence` (default: 0.5)
- Ensure good lighting and clear view of person

### Performance

- Use `model_complexity=0` for faster processing (less accurate)
- Use `model_complexity=2` for better accuracy (slower)
- Default is `1` (balanced)

## Next Steps

After skeleton extraction:
- **Step 3**: Subject extraction (optional SAM2 segmentation)
- **Step 4**: Pointing calculation and analysis
