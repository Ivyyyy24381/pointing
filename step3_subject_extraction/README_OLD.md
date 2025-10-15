# 🎭 Step 4: Subject Extraction (Optional)

**Purpose**: Segment subject (human/dog) from background using SAM2 for improved skeleton detection.

⚠️ **Note**: This step is optional. Use if skeleton detection quality is poor or background is cluttered.

---

## 📖 Overview

SAM2 (Segment Anything Model 2) provides:
- **Interactive segmentation**: User clicks → subject mask
- **Temporal propagation**: Mask automatically applied to entire video
- **High quality**: State-of-the-art segmentation accuracy

Benefits:
- Removes background distractions → better skeleton detection
- Isolates moving subject → cleaner 3D reconstruction
- Handles occlusions better

---

## 🎯 What This Step Does

### Segmentation Pipeline
1. User provides foreground/background points
2. SAM2 generates mask for first frame
3. Propagate mask to all frames (temporal coherence)
4. Save masks for downstream use

### Output
- Binary masks (H, W) for each frame
- Masked color images (optional)
- Segmentation metadata (scale factors, prompts)

---

## 📥 Inputs

### Required
```python
# Color images
images: List[np.ndarray]           # (H, W, 3) RGB images

# User prompts (first frame only)
foreground_points: List[Tuple[int, int]]  # Green dots
background_points: List[Tuple[int, int]]  # Red dots
```

### Optional
```python
# SAM2 model
model_checkpoint: str = "sam2_hiera_large.pt"
model_config: str = "sam2_hiera_l.yaml"

# Processing parameters
max_size: int = 360                # Resize long edge (memory)
confidence_threshold: float = 0.5
```

---

## 📤 Outputs

```python
@dataclass
class SegmentationResult:
    masks: List[np.ndarray]           # (N, H, W) bool masks
    masked_images: List[np.ndarray]   # (N, H, W, 3) images with background removed
    scale_metadata: Dict              # Scaling info for coordinate correction
```

Saved files:
- `trial_output/masks/mask_{frame_id:06d}.png` - Binary masks
- `trial_output/masked_video.mp4` - Video with background removed
- `trial_output/sam2_scale_metadata.json` - Important for coordinate correction!

```json
{
  "original_size": [640, 480],
  "sam2_size": [360, 270],
  "scale_factors": [1.778, 1.778],
  "foreground_points": [[320, 240], [350, 280]],
  "background_points": [[50, 50], [600, 450]]
}
```

---

## 🧮 Theory

### SAM2 Architecture

```
Input Image → Image Encoder (ViT) → Image Embeddings
                                           ↓
User Prompts → Prompt Encoder      → Prompt Embeddings
                                           ↓
                          Mask Decoder → Segmentation Mask
```

### Temporal Propagation

```
Frame 0: User provides points → SAM2 → Mask 0
         ↓
Frame 1: Use Mask 0 + optical flow → Propagate → Mask 1
         ↓
Frame 2: Use Mask 1 + optical flow → Propagate → Mask 2
         ...
```

### Scale Factor Correction

⚠️ **Critical**: SAM2 resizes images internally!

```python
# Original image: 640×480
# SAM2 processes: 360×270 (to fit GPU memory)
# Scale factor: 640/360 = 1.778

# Skeleton keypoints from masked image need correction:
x_corrected = x_detected * scale_factor_x
y_corrected = y_detected * scale_factor_y
```

---

## 💻 Usage

### Interactive GUI Mode
```python
from step4_subject_extraction import segment_with_sam2_gui

result = segment_with_sam2_gui(
    images=color_images,
    checkpoint="models/sam2_hiera_large.pt",
    output_dir="trial_output/"
)

# GUI controls:
# - Left click: Add foreground point (green)
# - Right click: Add background point (red)
# - Press 's': Start segmentation
# - Press 'r': Reset points
# - Press 'q': Quit
```

### Programmatic Mode
```python
from step4_subject_extraction import SAM2Segmenter

segmenter = SAM2Segmenter(
    checkpoint="models/sam2_hiera_large.pt",
    config="models/sam2_hiera_l.yaml"
)

# Define prompts
prompts = {
    'foreground': [(320, 240), (350, 280)],
    'background': [(50, 50), (600, 450)]
}

# Segment
result = segmenter.process_video(
    images=color_images,
    prompts=prompts,
    max_size=360
)

# Save
segmenter.save_results(result, "trial_output/")
```

### Apply Masks
```python
from step4_subject_extraction import apply_masks

# Option 1: Set background to black
masked_images = apply_masks(color_images, result.masks, bg_color=(0, 0, 0))

# Option 2: Set background to white
masked_images = apply_masks(color_images, result.masks, bg_color=(255, 255, 255))

# Option 3: Blur background
masked_images = apply_masks(color_images, result.masks, blur_bg=True)
```

---

## 🔬 Reference: Legacy Implementation

See `legeacy_code/dog_script/segmentation.py` for:
- `SAM2VideoSegmenter` class
- Interactive point selection GUI
- Scale metadata tracking
- Video saving with masks

---

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint` | `str` | Required | SAM2 model checkpoint path |
| `config` | `str` | Required | SAM2 config YAML |
| `max_size` | `int` | `360` | Max image dimension (memory) |
| `min_mask_area` | `int` | `100` | Min mask size (pixels²) |
| `confidence_threshold` | `float` | `0.5` | Mask confidence cutoff |
| `fps` | `int` | `10` | Output video FPS |

---

## 📊 Validation

### Check Segmentation Quality
```python
from step4_subject_extraction import validate_masks

metrics = validate_masks(
    masks=result.masks,
    images=color_images
)

print(f"Mean mask coverage: {metrics['mean_coverage']:.1%}")
print(f"Temporal consistency: {metrics['iou_stability']:.2f}")
print(f"Frames with valid mask: {metrics['valid_frames']}")
```

Good segmentation:
- Mask coverage: 10-40% (subject fills reasonable portion)
- IoU stability: >0.85 (masks don't jump between frames)
- Valid frames: 100%

---

## 🧪 Testing

```bash
# Test SAM2 installation
python -c "from sam2.build_sam import build_sam2; print('✅ SAM2 OK')"

# Test segmentation
pytest step4_subject_extraction/tests/test_sam2.py

# Integration test with GUI
python step4_subject_extraction/sam2_segmenter.py \
  --images tests/data/sample_frames/ \
  --checkpoint models/sam2_hiera_large.pt \
  --output tests/output/
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
# Reduce image size
segmenter.process_video(images, prompts, max_size=256)  # Default 360

# Or use CPU (slower)
segmenter = SAM2Segmenter(checkpoint, config, device='cpu')
```

### "Mask leaks into background"
- Add more background points (red)
- Click on problematic areas
- Increase `confidence_threshold`

### "Subject not fully segmented"
- Add more foreground points (green)
- Click on all parts of subject
- Try different first frame

### "Masks jump/flicker between frames"
- Motion too fast (reduce FPS)
- Re-run with more prompts
- Post-process with temporal smoothing

---

## 🎯 Best Practices

### Choosing First Frame
```python
# Good first frame:
# 1. Subject fully visible
# 2. Subject in center
# 3. Clear contrast with background
# 4. Subject not moving too fast

# Usually frame 10-20 (not frame 0)
first_frame_idx = 15
```

### Prompt Strategy
```python
# Foreground (green) - click ON subject:
# - Head/nose
# - Torso center
# - Limbs (all 4 legs for dog)

# Background (red) - click AWAY from subject:
# - Far corners (all 4)
# - Areas subject will never enter
# - Any static objects

foreground = [
    (320, 200),  # Head
    (340, 280),  # Torso
    (300, 350),  # Left leg
    (380, 350)   # Right leg
]

background = [
    (50, 50),    # Top-left corner
    (600, 50),   # Top-right
    (50, 450),   # Bottom-left
    (600, 450)   # Bottom-right
]
```

### When to Skip This Step
- Background is clean (single color, empty)
- Skeleton detection already works well
- Subject stays in one area (minimal occlusion)
- Processing time is critical

---

## 📚 Resources

- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [Download Checkpoints](https://github.com/facebookresearch/segment-anything-2#download-checkpoints)

---

## ✅ Checklist

Before moving to Step 5:
- [ ] Masks generated for all frames
- [ ] Visual inspection looks good
- [ ] Scale metadata saved (important!)
- [ ] Masked video created (optional)
- [ ] IoU stability >0.85

---

## 🔜 Next Step

→ **Step 5: Calculation** - Project 2D keypoints to 3D using masks (if available)

**Remember**: Apply scale factor correction in Step 5!
