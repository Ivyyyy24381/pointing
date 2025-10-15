# 🦴 Step 3: Skeleton Extraction

**Purpose**: Detect 2D keypoints (joints) of humans or animals using pose estimation models.

---

## 📖 Overview

Extract skeleton keypoints from each frame:
- **Humans/Babies**: MediaPipe Pose (33 keypoints)
- **Dogs**: DeepLabCut (custom quadruped model)

Output: 2D pixel coordinates + confidence scores for each keypoint

---

## 🎯 What This Step Does

### Detection Pipeline
1. Load color images (optionally masked)
2. Run pose estimation model
3. Extract keypoint coordinates and confidence
4. Save per-frame skeleton data

### Keypoint Examples
```
Human (MediaPipe):
  - nose, left_eye, right_eye
  - left_shoulder, right_shoulder
  - left_elbow, right_elbow
  - left_wrist, right_wrist
  - left_hip, right_hip
  - etc. (33 total)

Dog (DeepLabCut):
  - nose, left_ear, right_ear
  - neck, tail_base
  - left_front_paw, right_front_paw
  - left_back_paw, right_back_paw
  - etc. (custom model)
```

---

## 📥 Inputs

### Required
```python
# Color images
images: List[np.ndarray]           # (H, W, 3) RGB images

# Subject type
subject_type: str                  # "human", "baby", or "dog"
```

### Optional
```python
# Segmentation masks (from Step 4)
masks: Optional[List[np.ndarray]] = None  # (H, W) bool masks

# Detection parameters
min_confidence: float = 0.5        # Minimum keypoint confidence
model_path: str = None             # Custom model path for DeepLabCut
```

---

## 📤 Outputs

```python
@dataclass
class Keypoint:
    x: float                          # Pixel X coordinate
    y: float                          # Pixel Y coordinate
    confidence: float                 # Detection confidence [0, 1]

@dataclass
class SkeletonFrame:
    frame_id: int
    keypoints: Dict[str, Keypoint]    # name → Keypoint
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    subject_type: str                 # "human", "baby", "dog"
```

Saved to: `trial_output/skeleton_2d.json`

```json
{
  "frames": [
    {
      "frame_id": 0,
      "keypoints": {
        "nose": {"x": 320.5, "y": 180.2, "confidence": 0.95},
        "left_wrist": {"x": 280.1, "y": 350.4, "confidence": 0.88},
        ...
      },
      "bbox": [250, 150, 150, 300]
    },
    ...
  ],
  "subject_type": "human",
  "model_used": "mediapipe_v0.10.8"
}
```

---

## 🧮 Theory

### MediaPipe Pose

**Architecture**: BlazePose - lightweight CNN
**Input**: RGB image (any resolution)
**Output**: 33 keypoints + visibility flags

**Keypoint IDs**:
```python
0: nose
11: left_shoulder, 12: right_shoulder
13: left_elbow, 14: right_elbow
15: left_wrist, 16: right_wrist
23: left_hip, 24: right_hip
25: left_knee, 26: right_knee
27: left_ankle, 28: right_ankle
```

**Advantages**:
- Real-time (30+ FPS)
- No training needed
- Works on CPU

### DeepLabCut

**Architecture**: ResNet-based pose estimation
**Input**: RGB image
**Output**: Custom keypoints (model-dependent)

**Requires**:
- Pre-trained model for quadrupeds
- Config file with keypoint names
- Higher computational cost (GPU recommended)

**Advantages**:
- Highly accurate for custom animals
- Flexible keypoint definitions
- Can handle occlusions

---

## 💻 Usage

### MediaPipe (Humans/Babies)
```python
from step3_skeleton_extraction import MediaPipeDetector

detector = MediaPipeDetector(
    min_confidence=0.5,
    static_image_mode=False  # True for single images, False for video
)

skeletons = detector.process_images(
    images=color_images,
    frame_ids=range(len(color_images))
)

# Save results
detector.save_skeletons(skeletons, "trial_output/skeleton_2d.json")
```

### DeepLabCut (Dogs)
```python
from step3_skeleton_extraction import DeepLabCutDetector

detector = DeepLabCutDetector(
    config_path="models/dog_model/config.yaml",
    model_path="models/dog_model/snapshot-100000"
)

skeletons = detector.process_images(
    images=color_images,
    frame_ids=range(len(color_images))
)
```

### With Segmentation Masks
```python
# Use masks to improve detection accuracy
from step3_skeleton_extraction import apply_masks

masked_images = apply_masks(color_images, masks)
skeletons = detector.process_images(masked_images, frame_ids)
```

### Visualize Results
```python
from step3_skeleton_extraction import visualize_skeleton

for i, skeleton in enumerate(skeletons):
    vis_img = visualize_skeleton(
        image=color_images[i],
        skeleton=skeleton,
        connections=True,  # Draw connections between joints
        show_confidence=True
    )
    cv2.imwrite(f"debug/skeleton_{i:06d}.png", vis_img)
```

---

## 🔬 Reference: Legacy Implementation

See `legeacy_code/dog_script/detect_dog_skeleton.py` for:
- `detect_dog()` - DeepLabCut wrapper
- `run_mediapipe_json()` - MediaPipe batch processing
- Coordinate scaling and correction logic

---

## ⚙️ Parameters

### MediaPipe Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_detection_confidence` | `float` | `0.5` | Minimum detection confidence |
| `min_tracking_confidence` | `float` | `0.5` | Minimum tracking confidence (video mode) |
| `static_image_mode` | `bool` | `False` | True for images, False for video |
| `model_complexity` | `int` | `1` | 0=Lite, 1=Full, 2=Heavy |

### DeepLabCut Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `str` | Required | Path to DLC config |
| `model_path` | `str` | Required | Path to trained model |
| `pcutoff` | `float` | `0.1` | Minimum confidence threshold |
| `gputouse` | `int` | `0` | GPU device ID (None=CPU) |

---

## 📊 Validation

### Check Detection Quality
```python
from step3_skeleton_extraction import validate_skeletons

metrics = validate_skeletons(
    skeletons=skeletons,
    min_confidence=0.5,
    min_keypoints=10  # Minimum keypoints per frame
)

print(f"Frames with detections: {metrics['detection_rate']:.1%}")
print(f"Mean confidence: {metrics['mean_confidence']:.2f}")
print(f"Missing keypoints: {metrics['missing_keypoints']}")
```

Good detection:
- Detection rate > 90%
- Mean confidence > 0.7
- Key joints (nose, shoulders, hips) detected in >95% of frames

---

## 🧪 Testing

```bash
# Test MediaPipe
pytest step3_skeleton_extraction/tests/test_mediapipe.py

# Test DeepLabCut (requires model)
pytest step3_skeleton_extraction/tests/test_deeplabcut.py \
  --config tests/data/test_config.yaml

# Integration test
python step3_skeleton_extraction/tests/test_integration.py \
  --images tests/data/sample_frames/ \
  --type human \
  --output tests/output/
```

---

## 🐛 Troubleshooting

### "No pose detected"
- Check subject is clearly visible
- Try lowering `min_detection_confidence`
- Ensure good lighting and contrast
- For video: set `static_image_mode=False`

### "Keypoints jittering/jumping"
- Enable tracking mode (`static_image_mode=False`)
- Increase `min_tracking_confidence`
- Apply temporal smoothing (see Step 5)

### "DeepLabCut model not found"
- Verify `config_path` points to valid config.yaml
- Ensure model snapshot exists
- Check DLC installation: `python -c "import deeplabcut; print(deeplabcut.__version__)"`

### "Low confidence scores"
- Check image quality (resolution, blur, occlusion)
- For DLC: Retrain model with more diverse data
- Try different `model_complexity` for MediaPipe

---

## 🎯 Best Practices

### For Humans
```python
# Use Full model for accuracy
detector = MediaPipeDetector(model_complexity=1)

# Key joints for pointing analysis:
key_joints = [
    "nose",
    "left_wrist", "right_wrist",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow"
]
```

### For Dogs
```python
# Ensure model trained on similar breed/size
# Key joints for trajectory:
key_joints = [
    "nose",           # Primary tracking point
    "neck",           # Upper body center
    "tail_base",      # Orientation reference
    "front_paws",     # Contact points
    "back_paws"
]
```

### Performance Optimization
```python
# Batch processing (faster)
images = [cv2.imread(f) for f in image_paths]
skeletons = detector.process_images(images)

# VS single image (slower)
for img_path in image_paths:
    img = cv2.imread(img_path)
    skeleton = detector.process_image(img)
```

---

## 📚 Resources

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [DeepLabCut Documentation](http://www.mackenziemathislab.org/deeplabcut)
- [BlazePose Paper](https://arxiv.org/abs/2006.10204)

---

## ✅ Checklist

Before moving to Step 4:
- [ ] Skeletons detected for >90% of frames
- [ ] Key joints have confidence >0.5
- [ ] Skeleton data saved to JSON
- [ ] Visualizations look correct
- [ ] Unit tests passing

---

## 🔜 Next Step

→ **Step 4: Subject Extraction** (Optional) - Segment subject for improved skeleton accuracy

OR skip to:

→ **Step 5: Calculation** - Project 2D keypoints to 3D and compute metrics
