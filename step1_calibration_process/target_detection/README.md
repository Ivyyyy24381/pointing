# 🎯 Step 2: Target Detection

**Purpose**: Detect and localize target objects (cups) in 3D space using manual labeling and depth projection.

---

## 📖 Overview

This step allows users to:
1. Click on targets (cups) in a 2D image
2. Sample depth values at click locations
3. Project 2D pixels + depth → 3D world coordinates
4. Save target positions for downstream analysis

---

## 🎯 What This Step Does

### Interactive Labeling
- Display reference frame from trial
- User clicks on 4 cup centers
- System samples depth in neighborhood (5×5 window)
- Projects to 3D using camera calibration

### Output
- 3D position of each target
- Pixel coordinates for visualization
- Depth values for validation

---

## 📥 Inputs

### Required
```python
# Reference image
color_image: np.ndarray            # (H, W, 3) RGB image

# Depth data
depth_image: np.ndarray            # (H, W) float32 in meters

# Camera calibration (from Step 1)
camera_params: CameraParams        # fx, fy, cx, cy

# User interaction
num_targets: int = 4               # Number of targets to label
```

### Optional
```python
# Pre-existing labels (for refinement)
existing_targets: List[Target] = None
```

---

## 📤 Outputs

```python
@dataclass
class Target:
    id: int                           # 0, 1, 2, 3
    label: str                        # "target_1", "target_2", ...
    pixel_coords: Tuple[int, int]     # (x, y) click location
    world_coords: Tuple[float, float, float]  # (X, Y, Z) in meters
    depth_m: float                    # Depth at click (meters)
    confidence: float                 # 1.0 for manual labels
```

Saved to: `trial_output/target_coordinates.json`

```json
{
  "targets": [
    {
      "id": 0,
      "label": "target_1",
      "pixel_coords": [320, 240],
      "world_coords": [0.5, 0.3, 2.0],
      "depth_m": 2.0
    },
    ...
  ],
  "reference_frame": 150,
  "camera_params": {...}
}
```

---

## 🧮 Theory

### Depth Sampling Strategy

```
    ← 5×5 window →
┌─────────────────┐
│ • • • • •      │ ← Sample multiple depths
│ • • X • •      │   around click point X
│ • • • • •      │   to handle noise
└─────────────────┘

depth_final = median(valid_depths)
```

Why median?
- Robust to outliers
- Handles depth noise better than mean
- Filters out invalid (0) depth values

### 2D → 3D Projection

```python
# Given click (x_px, y_px) and depth Z:
X = (x_px - cx) * Z / fx
Y = (y_px - cy) * Z / fy
Z = Z

# In camera coordinates
position_cam = [X, Y, Z]

# Transform to world (if extrinsics available)
position_world = R @ position_cam + t
```

---

## 💻 Usage

### Interactive GUI Mode
```python
from step2_target_detection import label_targets_gui

targets = label_targets_gui(
    color_path="trial_input/trial_1/cam1/color/frame_000150.png",
    depth_path="trial_input/trial_1/cam1/depth/frame_000150.npy",
    camera_params=params,
    num_targets=4
)

# GUI opens:
# - Click on 4 cups in order
# - See real-time 3D coordinates
# - Press 's' to save, 'r' to reset, 'q' to quit
```

### Programmatic Mode
```python
from step2_target_detection import detect_targets_from_clicks

click_coords = [(320, 240), (450, 230), (190, 250), (580, 260)]

targets = detect_targets_from_clicks(
    color_image=color_img,
    depth_image=depth_img,
    camera_params=params,
    click_coords=click_coords
)
```

### Load Existing Targets
```python
from step2_target_detection import load_targets

targets = load_targets("trial_output/target_coordinates.json")
print(f"Loaded {len(targets)} targets")
```

---

## 🔬 Reference: Legacy Implementation

See `legeacy_code/calibrate_target.py` for:
- `calibrate_targets()` - Interactive ROI selection
- Depth sampling and averaging
- Offset correction for missing depths

---

## 🎨 GUI Controls

| Key | Action |
|-----|--------|
| **Left Click** | Mark target location |
| **Right Click** | Remove last target |
| **'r'** | Reset all targets |
| **'s'** | Save and continue |
| **'q'** | Quit without saving |
| **'h'** | Show help overlay |

---

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_targets` | `int` | `4` | Number of targets to label |
| `window_size` | `int` | `5` | Depth sampling window (pixels) |
| `depth_method` | `str` | `"median"` | How to aggregate depths (`"median"` or `"mean"`) |
| `min_depth_m` | `float` | `0.1` | Minimum valid depth |
| `max_depth_m` | `float` | `10.0` | Maximum valid depth |

---

## 📊 Validation

### Check Target Quality
```python
from step2_target_detection import validate_targets

metrics = validate_targets(
    targets=targets,
    depth_image=depth_img,
    expected_depth_range=(1.0, 4.0)  # meters
)

print(f"Valid depth samples: {metrics['valid_ratio']:.1%}")
print(f"Depth variance: {metrics['depth_std']:.3f} m")
```

Good targets:
- Valid depth ratio > 90%
- Depth variance < 0.05m (within 5cm)

---

## 🧪 Testing

```bash
# Run unit tests
pytest step2_target_detection/tests/

# Test GUI with sample data
python step2_target_detection/manual_labeling.py \
  --color test_data/color.png \
  --depth test_data/depth.npy \
  --config config/camera_config.yaml
```

---

## 🐛 Troubleshooting

### "No valid depth at click location"
- Depth hole (IR occlusion)
- Try clicking nearby
- Increase `window_size` to 7 or 9

### "Target seems off / unrealistic position"
- Check camera calibration is correct
- Verify depth values are in meters (not millimeters)
- Ensure coordinate system is right-handed

### "Can't see targets in image"
- Choose a frame where targets are clearly visible
- Adjust window brightness/contrast
- Use frame from middle of trial (not start/end)

---

## 🎯 Best Practices

### Choosing Reference Frame
```python
# Use frame where:
# 1. All targets visible
# 2. Subject NOT in frame (clear view)
# 3. Good lighting
# 4. Targets in focus

reference_frame = total_frames // 2  # Middle frame usually good
```

### Target Order
```
Label targets consistently:
  target_1: Left-most
  target_2: Center-left
  target_3: Center-right
  target_4: Right-most

Or use custom labeling scheme
```

### Quality Check
```python
# After labeling, verify:
1. All targets have depth > 0
2. Targets form reasonable spatial pattern
3. Distances between targets make sense

# Example: Targets should be ~0.5-1.0m apart
for i in range(len(targets)-1):
    dist = np.linalg.norm(
        np.array(targets[i].world_coords) -
        np.array(targets[i+1].world_coords)
    )
    print(f"Distance {i}→{i+1}: {dist:.2f}m")
```

---

## 📚 Resources

- [Depth Camera Principles](https://en.wikipedia.org/wiki/Depth_camera)
- [RealSense Depth Quality](https://www.intelrealsense.com/depth-camera-d435/)

---

## ✅ Checklist

Before moving to Step 3:
- [ ] 4 targets labeled
- [ ] All targets have valid depth (>0)
- [ ] Target positions saved to JSON
- [ ] Visualized 3D positions look reasonable
- [ ] Unit tests passing

---

## 🔜 Next Step

→ **Step 3: Skeleton Extraction** - Detect human/animal keypoints for trajectory analysis
