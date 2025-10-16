# Multi-Subject Detection System

## Overview

The pipeline now supports detecting **humans, dogs, and babies** with appropriate detection methods for each.

## Detection Methods

### 1. **Human Detection** (Default)
- **Method**: MediaPipe Pose
- **Processing**: Full image
- **Keypoints**: 33 landmarks
- **Use case**: Adult human pointing gestures

### 2. **Baby Detection** 👶
- **Method**: MediaPipe Pose with lower-half cropping
- **Processing**: Lower 50% of image only
- **Keypoints**: 33 landmarks (same as human)
- **Why lower-half**: Avoids false detection of adults in frame
- **Use case**: Baby pointing gestures while adult present

### 3. **Dog Detection** 🐶
- **Method**: DeepLabCut SuperAnimal Quadruped
- **Processing**: Full video (not frame-by-frame)
- **Detector**: Built-in Faster R-CNN (automatic)
- **Keypoints**: 39 landmarks (nose, ears, paws, tail, etc.)
- **Use case**: Dog pointing/orientation analysis

## How to Use

### In the UI (Page 2):

1. **Load your trial** from `trial_input/`

2. **Select subject type**:
   - `👤 Human (default)` - Always checked (cannot uncheck)
   - `🐶 Dog (optional)` - Check for dog detection
   - `👶 Baby (optional)` - Check for baby detection

3. **Process frames**:
   - **Human/Baby**: Use "Process Current Frame" or "Process All Frames"
   - **Dog**: Only "Process All Frames" (video-based detection)

### Subject Type Priority:
If multiple types are checked, the system uses:
```
Dog > Baby > Human
```

## Implementation Details

### Baby Detection (Lower-Half Processing)

**Files Modified:**
- [mediapipe_human.py](step2_skeleton_extraction/mediapipe_human.py:51,86-87,110-144)
- [image_utils.py](step2_skeleton_extraction/image_utils.py) (new file)

**How it works:**
```python
# 1. Crop image to lower half
cropped_image, y_offset = crop_to_lower_half(image, crop_ratio=0.5)

# 2. Run MediaPipe on cropped image
results = pose.process(cropped_image)

# 3. Map coordinates back to original image
landmarks_original = map_coordinates_from_crop(landmarks_cropped, y_offset)
```

**Result:** Baby is detected in lower half, adults in upper half are ignored.

### Dog Detection (DeepLabCut)

**Files Created:**
- [deeplabcut_dog.py](step2_skeleton_extraction/deeplabcut_dog.py) (new file)

**How it works:**
```python
detector = DeepLabCutDogDetector()

# Process entire video with built-in Faster R-CNN detector
results = detector.detect_video(video_path)
# DeepLabCut automatically:
# - Finds the dog using Faster R-CNN
# - Tracks skeleton across all frames
# - Returns JSON with 39 keypoints per frame
```

**No manual segmentation needed!** The Faster R-CNN detector automatically locates the dog.

### UI Integration

**Detector Initialization** ([ui_skeleton_extractor.py:281-327](step2_skeleton_extraction/ui_skeleton_extractor.py:281-327)):
```python
if detect_dog:
    self.detector = None  # Video-only mode
elif detect_baby:
    self.detector = MediaPipeHumanDetector(lower_half_only=True)
else:
    self.detector = MediaPipeHumanDetector(lower_half_only=False)
```

**Subject Type Changes** ([ui_skeleton_extractor.py:376-414](step2_skeleton_extractor.py:376-414)):
- Checking dog/baby triggers `reinitialize_detector()`
- Detector is recreated with appropriate settings
- Status label updates to show active mode

## Data Saved

### Current Implementation

**JSON Output** (`trial_output/X/single_camera/skeleton_results.json`):
```json
{
  "frame_000001": {
    "frame_number": 1,
    "landmarks_2d": [[x, y, conf], ...],
    "landmarks_3d": [[x, y, z], ...],
    "hip_center_2d": [x, y],
    "hip_center_3d": [x, y, z],
    "arm_vectors": {...},
    "metadata": {
      "pointing_arm": "left" | "right",
      "subject_type": "human" | "dog" | "baby"  // TODO: Add this
    }
  }
}
```

### Legacy Format (To Implement)

**CSV Output** (`processed_subject_result_table.csv`):
- Frame-by-frame rows
- Individual joint columns (x, y, conf, x_m, y_m, z_m)
- Leg/limb averages
- Target distances (r, theta, phi)
- Orientation vectors (head, torso)
- 3D trace coordinates

See [dog_pose_visualize.py:503-779](step3_subject_extraction/dog_script/dog_pose_visualize.py:503-779) for format.

## File Structure

```
trial_input/X/single_camera/
├── color/
│   ├── frame_000001.png
│   └── ...
└── depth/
    ├── frame_000001.npy
    └── ...

trial_output/X/single_camera/
├── skeleton_results.json          # Current format (all subjects)
└── processed_subject_result_table.csv  # Legacy format (TODO)
```

## Advantages Over SAM2 Segmentation

| Aspect | SAM2 | Current Approach |
|--------|------|------------------|
| **User interaction** | Required | None (automatic) |
| **Processing time** | 5-10 minutes | Seconds |
| **Memory** | High (GPU) | Moderate |
| **Dependencies** | SAM2 + torch | DLC / MediaPipe |
| **Complexity** | High | Low |

## Technical Notes

### Lower-Half Cropping for Babies

**Why it works:**
- Babies typically sit/crawl on floor (lower half)
- Adults stand/sit in upper half
- Cropping naturally separates them spatially

**Crop ratio**: 50% (configurable in `MediaPipeHumanDetector.crop_ratio`)

**Coordinate mapping**: Keypoints are automatically mapped back to original image coordinates for depth lookups and visualization.

### DeepLabCut Built-in Detector

**Model**: `superanimal_quadruped`
- Pre-trained on thousands of quadruped videos
- 39 keypoints covering full body
- Generalizes to dogs, cats, horses, etc.

**Detector**: `fasterrcnn_resnet50_fpn_v2`
- Automatically localizes animal in frame
- No bounding box annotation needed
- `max_individuals=1` ensures single subject

## Known Limitations

### Dog Detection
- **Frame-by-frame not supported**: DeepLabCut processes entire video at once
- **UI limitation**: "Process Current Frame" button disabled in dog mode
- **Workaround**: Use "Process All Frames" to process full video

### Baby Detection
- **Requires spatial separation**: Baby must be in lower half, adult in upper half
- **May fail if**: Baby held by adult, or adult sitting on floor

## Future Enhancements

1. **Multi-subject tracking**: Support detecting human + dog simultaneously
2. **CSV export**: Implement legacy format export matching dog_pose_visualize.py
3. **Custom crop regions**: Allow user-defined ROI instead of fixed lower-half
4. **Dog frame-by-frame**: Explore frame-level DeepLabCut inference for UI

## Dependencies

- **MediaPipe**: `pip install mediapipe` (human/baby)
- **DeepLabCut**: `pip install deeplabcut` (dog)
- **Numpy, OpenCV**: Standard requirements

## References

- [DeepLabCut SuperAnimal](https://github.com/DeepLabCut/DeepLabCut)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- Legacy pipeline: [step3_subject_extraction/dog_script/](step3_subject_extraction/dog_script/)
