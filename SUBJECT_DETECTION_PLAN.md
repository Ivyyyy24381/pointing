# Subject Detection Architecture (Dog/Baby)

## Goals
- Separate pipeline from human detection
- Process only lower half of image
- Independent data storage (separate JSON)
- Independent visualization (separate overlays)
- Can run simultaneously with human detection

## Architecture

### 1. Data Storage Structure
```
trial_output/X/single_camera/
├── skeleton_results.json          # Human detection (existing)
├── dog_detection_results.json     # Dog detection (new)
└── baby_detection_results.json    # Baby detection (new)
```

### 2. Detection Pipeline

**Human (Existing)**:
- Full image → MediaPipe → 33 keypoints → 3D reconstruction

**Dog (New, Separate)**:
- Lower 50% → YOLO bbox → Crop → DeepLabCut → 39 keypoints → Save to dog_detection_results.json

**Baby (New, Separate)**:
- Lower 50% → MediaPipe → 33 keypoints → Save to baby_detection_results.json

### 3. UI Integration

**Checkboxes**:
- `👤 Human` - Default, always processes
- `🐶 Dog` - Optional, runs dog pipeline when checked
- `👶 Baby` - Optional, runs baby pipeline when checked

**Visualization Layers**:
1. Base: RGB/Depth image
2. Layer 1: Human skeleton (green) - if detected
3. Layer 2: Dog bbox + skeleton (blue) - if detected
4. Layer 3: Baby skeleton (yellow) - if detected

**3D Visualization**:
- Human: Full skeleton with arm vectors
- Dog: Center point + bbox outline (or full skeleton if implemented)
- Baby: Full skeleton (different color)

### 4. Implementation Plan

#### Step 1: Create Subject Detector Module
```python
# subject_detector.py
class SubjectDetector:
    def __init__(self, subject_type='dog'):
        self.subject_type = subject_type  # 'dog' or 'baby'
        self.crop_ratio = 0.5  # Lower 50%

    def detect_frame(self, image, frame_num):
        # Crop to lower half
        cropped, y_offset = crop_to_lower_half(image)

        if self.subject_type == 'dog':
            # YOLO bbox → DeepLabCut skeleton
            pass
        else:  # baby
            # MediaPipe on cropped image
            pass

        # Map coordinates back to original
        return result
```

#### Step 2: Update UI to Support Multiple Pipelines
```python
# ui_skeleton_extractor.py
def process_current_frame(self):
    # Human detection (existing)
    human_result = self.detector.detect_frame(...)

    # Dog detection (if enabled)
    dog_result = None
    if self.detect_dog_var.get():
        dog_result = self.dog_detector.detect_frame(...)

    # Baby detection (if enabled)
    baby_result = None
    if self.detect_baby_var.get():
        baby_result = self.baby_detector.detect_frame(...)

    # Store separately
    self.human_results[frame_key] = human_result
    self.dog_results[frame_key] = dog_result
    self.baby_results[frame_key] = baby_result
```

#### Step 3: Separate Visualization
```python
def update_display(self):
    display_img = self.current_color.copy()

    # Draw human (green)
    if self.human_result:
        display_img = self.draw_human_skeleton(display_img)

    # Draw dog (blue)
    if self.dog_result:
        display_img = self.draw_dog_bbox_and_skeleton(display_img)

    # Draw baby (yellow)
    if self.baby_result:
        display_img = self.draw_baby_skeleton(display_img)
```

#### Step 4: Separate Save Functions
```python
def save_results(self):
    # Save human results
    self.save_human_results()

    # Save dog results (if any)
    if self.dog_results:
        self.save_dog_results()

    # Save baby results (if any)
    if self.baby_results:
        self.save_baby_results()
```

### 5. Benefits of This Architecture

✅ **Clean Separation**: Each pipeline independent
✅ **Easy to Enable/Disable**: Just check/uncheck boxes
✅ **No Conflicts**: Separate data storage
✅ **Easy to Debug**: Each pipeline can be tested separately
✅ **Scalable**: Easy to add more subject types later

### 6. Data Format

**Human Results** (existing):
```json
{
  "frame_000001": {
    "landmarks_2d": [[x, y, conf], ...],  // 33 points
    "landmarks_3d": [[x, y, z], ...],
    "pointing_arm": "left"
  }
}
```

**Dog Results** (new):
```json
{
  "frame_000001": {
    "bbox": [x1, y1, x2, y2],
    "keypoints_2d": [[x, y, conf], ...],  // 39 points
    "subject_type": "dog",
    "detection_region": "lower_half"
  }
}
```

**Baby Results** (new):
```json
{
  "frame_000001": {
    "keypoints_2d": [[x, y, conf], ...],  // 33 points
    "subject_type": "baby",
    "detection_region": "lower_half"
  }
}
```

## Next Steps

1. Implement `SubjectDetector` class
2. Update UI to have three separate result dictionaries
3. Add drawing functions for each subject type
4. Add separate save functions
5. Test with human + dog, human + baby, all three together
