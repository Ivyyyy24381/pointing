# 🏗️ Project Architecture

**Clean, modular pipeline for pointing gesture analysis**

---

## 📐 Design Principles

1. **Separation of Concerns**: Each step is independent and testable
2. **Single Responsibility**: One module = one clear purpose
3. **Data Flow**: Explicit inputs/outputs for each step
4. **Reusability**: Utility functions shared across steps
5. **Testability**: Each component has unit tests
6. **Documentation**: Every step has comprehensive README

---

## 📂 Directory Structure

```
pointing/
│
├── step0_data_loading/              # Load and standardize input data
│   ├── README.md                    # Data format and loading docs
│   ├── data_loader.py               # Load images from trial folders
│   ├── data_validator.py            # Validate data integrity
│   └── tests/                       # Unit tests
│
├── step1_calibration_process/       # Camera calibration
│   ├── README.md                    # Calibration theory and usage
│   ├── intrinsics.py                # Load/compute camera intrinsics
│   ├── extrinsics.py                # Compute camera poses (AprilTag)
│   └── tests/
│
├── step2_target_detection/          # Detect and localize targets (cups)
│   ├── README.md                    # Target detection methods
│   ├── manual_labeling.py           # GUI for manual target labeling
│   ├── projection.py                # 2D click → 3D position
│   └── tests/
│
├── step3_skeleton_extraction/       # Detect human/animal skeleton
│   ├── README.md                    # Skeleton detection docs
│   ├── mediapipe_detector.py        # MediaPipe for humans/babies
│   └── tests/
│
├── step4_subject_extraction/        # Segment subject from background
│   ├── README.md                    # Segmentation methods
│   ├── sam2_segmenter.py            # SAM2 foreground segmentation
│   ├── deeplabcut_detector.py       # DeepLabCut for dogs
│   ├── mask_utils.py                # Mask processing utilities
│   └── tests/
│
├── step5_calculation/               # Compute metrics and analysis
│   ├── README.md                    # Calculation methods
│   ├── trajectory.py                # 3D trajectory computation
│   ├── distances.py                 # Distance/angle calculations
│   ├── pointing_inference.py        # Infer pointing direction
│   └── tests/
│
├── utils/                           # Shared utility functions
│   ├── 2d_3d_conversion/           # Coordinate transformations
│   │   ├── projection.py            # 2D ↔ 3D conversions
│   │   └── README.md
│   ├── trial_data_structure_cleanup/  # Data cleaning utilities
│   ├── visualization/               # Plotting and visualization
│   └── io/                          # File I/O helpers
│
├── web_ui/                          # Flask web interface
│   ├── app.py                       # Main Flask app
│   ├── templates/                   # HTML templates
│   ├── static/                      # CSS/JS/images
│   └── README.md                    # Web UI documentation
│
├── trial_input/                     # Example input data
│   └── .gitkeep
│
├── trial_output/                    # Processing outputs
│   └── .gitkeep
│
├── legeacy_code/                    # Old implementation (reference only)
│   └── (archived code)
│
└── thirdparty/                      # External dependencies
    └── (external libraries)
```

---

## 🔄 Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 0: Data Loading                                       │
│  Load images, validate format, standardize resolution       │
└────────────────────┬────────────────────────────────────────┘
                     │ Color images (640×480)
                     │ Depth images (.npy or .png)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Calibration                                         │
│  Load camera intrinsics (fx, fy, cx, cy)                   │
│  Optional: Compute extrinsics with AprilTag                 │
└────────────────────┬────────────────────────────────────────┘
                     │ Camera parameters
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Target Detection                                    │
│  User clicks on targets → 2D pixels                         │
│  Project to 3D using depth + camera params                  │
└────────────────────┬────────────────────────────────────────┘
                     │ Target 3D positions [X, Y, Z]
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Skeleton Extraction                                 │
│  Detect keypoints: nose, wrists, shoulders, etc.           │
│  MediaPipe (humans) or DeepLabCut (dogs)                   │
└────────────────────┬────────────────────────────────────────┘
                     │ 2D keypoints per frame
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Subject Extraction (Optional)                       │
│  Segment subject from background using SAM2                 │
│  Improve skeleton detection accuracy                        │
└────────────────────┬────────────────────────────────────────┘
                     │ Segmentation masks
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Calculation & Analysis                             │
│  Project 2D keypoints → 3D positions                        │
│  Compute distances, angles, trajectories                    │
│  Infer pointing direction and target selection              │
└────────────────────┬────────────────────────────────────────┘
                     │ Final results CSV + visualizations
                     ↓
                  OUTPUT
```

---

## 📊 Module Interfaces

### Step 0: Data Loading
```python
# Input
trial_dir: Path                    # e.g., "trial_input/trial_1/"

# Output
DataBatch {
    color_images: List[np.ndarray]      # (H, W, 3) uint8
    depth_images: List[np.ndarray]      # (H, W) float32 in meters
    frame_ids: List[int]
    metadata: Dict                       # Original resolution, etc.
}
```

### Step 1: Calibration
```python
# Input
config_file: Path                  # "config/camera_config.yaml"
# OR
calibration_images: List[np.ndarray]  # Images with AprilTag

# Output
CameraParams {
    fx, fy: float                      # Focal lengths
    cx, cy: float                      # Principal point
    width, height: int                 # Image dimensions
    distortion: Optional[np.ndarray]   # Distortion coefficients
}
```

### Step 2: Target Detection
```python
# Input
image: np.ndarray                  # Reference image
depth_image: np.ndarray
camera_params: CameraParams
user_clicks: List[Tuple[int, int]]  # [(x, y), ...]

# Output
List[Target] {
    id: int
    label: str                         # "target_1", "target_2", ...
    pixel_coords: Tuple[int, int]      # (x, y)
    world_coords: Tuple[float, float, float]  # (X, Y, Z)
    depth_m: float
}
```

### Step 3: Skeleton Extraction
```python
# Input
images: List[np.ndarray]
detector_type: str                 # "mediapipe" or "deeplabcut"

# Output
List[SkeletonFrame] {
    frame_id: int
    keypoints: Dict[str, Keypoint]     # name → (x, y, confidence)
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
}
```

### Step 4: Subject Extraction
```python
# Input
images: List[np.ndarray]
prompts: List[Point]               # Foreground/background points

# Output
List[np.ndarray]                   # Segmentation masks (H, W) bool
```

### Step 5: Calculation
```python
# Input
skeletons: List[SkeletonFrame]
depth_images: List[np.ndarray]
targets: List[Target]
camera_params: CameraParams

# Output
AnalysisResults {
    trajectory_3d: np.ndarray          # (N, 3) positions over time
    distances_to_targets: np.ndarray   # (N, num_targets)
    closest_target_per_frame: List[int]
    pointing_direction: Optional[np.ndarray]  # For human gestures
    orientation_vectors: np.ndarray    # Subject orientation
}
```

---

## 🧩 Utility Modules

### 2D ↔ 3D Conversion
```python
from utils.projection import pixel_to_3d, project_3d_to_2d

# Pixel → 3D
X, Y, Z = pixel_to_3d(x_px, y_px, depth_m, fx, fy, cx, cy)

# 3D → Pixel
x_px, y_px = project_3d_to_2d(X, Y, Z, fx, fy, cx, cy)
```

### Visualization
```python
from utils.visualization import plot_trajectory_3d, plot_skeleton

plot_trajectory_3d(positions, targets, output_path="trajectory.png")
plot_skeleton(image, keypoints, connections)
```

### I/O
```python
from utils.io import load_depth, save_results

depth = load_depth("depth.npy")  # Auto-handles .npy, .png, .raw
save_results(results, "output/analysis.json")
```

---

## 🧪 Testing Strategy

### Unit Tests
Each step has `tests/` folder with:
- `test_module.py` - Test individual functions
- `test_integration.py` - Test step end-to-end
- `test_data/` - Small sample datasets

### Running Tests
```bash
# Test specific step
pytest step1_calibration_process/tests/

# Test all
pytest

# With coverage
pytest --cov=. --cov-report=html
```

---

## 📝 Documentation Standards

Each `README.md` includes:

1. **Purpose**: What does this step do?
2. **Theory**: Background/algorithms used
3. **Inputs**: Expected input format
4. **Outputs**: Output format and files
5. **Usage**: Command-line examples
6. **Parameters**: Configuration options
7. **Examples**: Code snippets
8. **Troubleshooting**: Common issues

---

## 🔀 Branching Strategy

```
main                    # Stable, production-ready
├── develop             # Integration branch
    ├── feature/step1-calibration
    ├── feature/step2-targets
    ├── feature/web-ui
    └── bugfix/depth-loading
```

---

## 🚀 Development Workflow

### Adding a New Step

1. Create module structure:
   ```bash
   mkdir -p stepX_name/{tests,docs}
   touch stepX_name/{README.md,main.py,tests/test_main.py}
   ```

2. Write README with:
   - Purpose
   - Input/Output specs
   - Usage examples

3. Implement module with:
   - Clear function signatures
   - Type hints
   - Docstrings

4. Write tests:
   - Unit tests for each function
   - Integration test for full step

5. Update ARCHITECTURE.md

### Code Review Checklist

- [ ] Follows single responsibility principle
- [ ] Has comprehensive README
- [ ] Includes unit tests (>80% coverage)
- [ ] Type hints on all functions
- [ ] Docstrings for public API
- [ ] No hardcoded paths
- [ ] Error handling for edge cases
- [ ] Logging for debugging

---

## 🎯 Current Status

| Step | Status | Completion |
|------|--------|------------|
| Step 0: Data Loading | ✅ Documented | 20% |
| Step 1: Calibration | 📝 Planning | 0% |
| Step 2: Target Detection | 📝 Planning | 0% |
| Step 3: Skeleton Extraction | 📝 Planning | 0% |
| Step 4: Subject Extraction | 📝 Planning | 0% |
| Step 5: Calculation | 📝 Planning | 0% |
| Web UI | 📝 Template | 5% |
| Utils | 🔧 In Progress | 15% |

---

## 📧 Contributing

See individual step READMEs for implementation details.

When contributing:
1. Work on one step at a time
2. Write tests first (TDD)
3. Update documentation
4. Run full test suite before PR

---

**Last Updated**: 2025-01-09
**Version**: 2.0 (Refactored Architecture)
