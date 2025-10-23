# 🔧 Step 1: Camera Calibration

**Purpose**: Load or compute camera intrinsic and extrinsic parameters for accurate 3D reconstruction.

---

## 📖 Overview

Camera calibration provides the mathematical model needed to:
1. **Project 2D pixels → 3D world coordinates** using depth data
2. **Transform between multiple camera coordinate systems** (extrinsics)
3. **Correct lens distortion** for accurate measurements

---

## 🎯 What This Step Does

### Intrinsics (Required)
- Focal lengths: `fx`, `fy` (pixels)
- Principal point: `cx`, `cy` (optical center in pixels)
- Image dimensions: `width`, `height`
- Optional: Distortion coefficients

### Extrinsics (Optional)
- Rotation matrix `R` (3×3)
- Translation vector `t` (3×1)
- Transforms camera coords → world coords

---

## 📥 Inputs

### Method 1: Load from Config File
```yaml
# config/camera_config.yaml
cam1:
  fx: 385.716
  fy: 385.254
  cx: 320.872
  cy: 239.636
  width: 640
  height: 480
  distortion_coeffs: [...]  # Optional
```

### Method 2: Calibrate with AprilTag
```python
# Images containing AprilTag of known size
images: List[np.ndarray]
tag_size_m: float = 0.16  # Tag size in meters
```

### Method 3: Manual Input
```python
# User provides parameters directly
fx, fy, cx, cy: float
```

---

## 📤 Outputs

```python
@dataclass
class CameraParams:
    fx: float                        # Focal length X (pixels)
    fy: float                        # Focal length Y (pixels)
    cx: float                        # Principal point X (pixels)
    cy: float                        # Principal point Y (pixels)
    width: int                       # Image width
    height: int                      # Image height
    distortion: Optional[np.ndarray] # [k1, k2, p1, p2, k3]
    extrinsics: Optional[np.ndarray] # 4×4 transformation matrix
```

Saved to: `trial_output/calibration.json`

---

## 🧮 Theory

### Pinhole Camera Model

```
┌─────────────────────────────────┐
│  3D World Point (X, Y, Z)       │
│           ↓                     │
│  Camera Transformation          │
│  [R|t] (extrinsics)            │
│           ↓                     │
│  Camera Coordinates (Xc, Yc, Zc)│
│           ↓                     │
│  Projection (intrinsics)        │
│  x = fx * (Xc/Zc) + cx         │
│  y = fy * (Yc/Zc) + cy         │
│           ↓                     │
│  2D Image Point (x, y)          │
└─────────────────────────────────┘
```

### Inverse Projection (2D + Depth → 3D)

Given pixel `(x, y)` and depth `Z`:
```python
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
Z = Z  # depth value
```

---

## 💻 Usage

### Method 1: Load from Config
```python
from step1_calibration_process import load_calibration

params = load_calibration("config/camera_config.yaml", camera_id="cam1")
print(f"fx={params.fx}, fy={params.fy}")
```

### Method 2: Detect AprilTag
```python
from step1_calibration_process import calibrate_with_apriltag

images = [cv2.imread(f"calib_{i}.png") for i in range(5)]
params = calibrate_with_apriltag(
    images=images,
    tag_family="tag36h11",
    tag_size_m=0.16
)
```

### Method 3: Manual Input
```python
from step1_calibration_process import CameraParams

params = CameraParams(
    fx=385.7, fy=385.2,
    cx=320.0, cy=240.0,
    width=640, height=480
)
```

### Save Calibration
```python
from step1_calibration_process import save_calibration

save_calibration(params, "trial_output/calibration.json")
```

---

## 🔬 Reference: Legacy Implementation

See `legeacy_code/calibration.py` for:
- `find_realsense_intrinsics()` - Extract from ROS bag
- `calibrate_camera_to_world()` - Umeyama algorithm for R, t
- `calibrate_side_to_front()` - Multi-camera alignment

---

## 📊 Validation

### Check Calibration Quality
```python
from step1_calibration_process import validate_calibration

metrics = validate_calibration(
    params=params,
    test_images=test_images,
    ground_truth_3d=known_points
)

print(f"Reprojection error: {metrics['mean_error_px']:.2f} pixels")
```

Good calibration: < 1.0 pixel error

---

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `str` | `config/camera_config.yaml` | Path to config file |
| `camera_id` | `str` | `"cam1"` | Which camera to load |
| `tag_family` | `str` | `"tag36h11"` | AprilTag family |
| `tag_size_m` | `float` | `0.16` | Physical tag size (meters) |

---

## 🧪 Testing

```bash
# Run unit tests
pytest step1_calibration_process/tests/test_intrinsics.py

# Test with sample data
python step1_calibration_process/tests/test_integration.py \
  --config config/camera_config.yaml \
  --validate
```

---

## 🐛 Troubleshooting

### "Focal length seems wrong"
- Check image resolution matches config (should be 640×480)
- Verify `fx`, `fy` are in pixels, not millimeters
- Typical range: 300-700 pixels for RealSense

### "AprilTag not detected"
- Ensure tag is clearly visible and well-lit
- Try larger tag size (>10cm)
- Check tag family matches detector

### "Reprojection error too high"
- Re-calibrate with more images
- Check for motion blur in calibration images
- Verify depth data is accurate

---

## 📚 Resources

- [Camera Calibration Theory](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [AprilTag](https://april.eecs.umich.edu/software/apriltag)
- [RealSense SDK](https://github.com/IntelRealSense/librealsense)

---

## ✅ Checklist

Before moving to Step 2:
- [ ] Camera parameters loaded successfully
- [ ] Validation error < 1.0 pixel
- [ ] Config saved to `trial_output/calibration.json`
- [ ] Unit tests passing
- [ ] Documentation updated

---

## 🔜 Next Step

→ **Step 2: Target Detection** - Use calibration to convert 2D clicks to 3D positions
