# 🎯 Pointing Gesture Analysis - Web Interface

A Flask-based web interface for step-by-step human pointing gesture analysis.

## ✨ Features

**Simple Version (v1.0)**:
- ✅ Upload color + depth images
- ✅ Camera calibration (manual, AprilTag, or config file)
- ✅ AprilTag detection for world coordinate system
- ✅ Human detection using MediaPipe
- ✅ Interactive cup selection (click on image)
- ✅ Compute relative positions and pointing direction
- ✅ 3D visualization of results

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd web_pipeline
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

### 3. Open Browser

Navigate to: `http://localhost:5000`

## 📋 Step-by-Step Workflow

### Step 1: Upload Data
- Upload a color image (PNG/JPG)
- Upload a depth file (RAW format, uint16)

### Step 2: Camera Calibration
Choose one method:
- **Manual**: Enter fx, fy, cx, cy manually
- **AprilTag**: Auto-detect from AprilTag in scene
- **Config**: Load from `config/camera_config.yaml`

### Step 3: AprilTag Detection (Optional)
- Automatically detects AprilTag if present
- Uses tag to establish world coordinate system
- Skips if no tag found

### Step 4: Human Detection
- Uses MediaPipe Pose to detect human
- Extracts 3D positions of:
  - Nose
  - Left/Right wrist
  - Left/Right shoulder
  - Left/Right elbow

### Step 5: Cup Detection
- Click on the image to mark 4 cup positions
- System automatically computes 3D positions using depth

### Step 6: Compute Relative Positions
- Calculates pointing vector (shoulder → wrist)
- Computes distance from wrist to each cup
- Computes distance from pointing ray to each cup
- Identifies which cup human is pointing at

### Step 7: Visualize Results
- 3D scatter plot showing:
  - Human keypoints
  - 4 cups
  - Pointing ray
- Summary JSON with all measurements
- Downloadable results

## 📁 File Structure

```
web_pipeline/
├── app.py                  # Flask server
├── templates/
│   └── index.html          # Web interface
├── uploads/                # Temporary uploaded files
├── results/                # Generated visualizations & results
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎨 API Endpoints

### Pipeline Control
- `GET /` - Main interface
- `POST /reset` - Reset pipeline
- `GET /status` - Get current status

### Pipeline Steps
- `POST /step/upload` - Upload color + depth
- `POST /step/calibrate` - Camera calibration
- `POST /step/detect_apriltag` - Detect AprilTag
- `POST /step/detect_human` - Detect human pose
- `POST /step/detect_cups` - Detect cups from clicks
- `POST /step/compute_positions` - Compute relative positions
- `POST /step/visualize` - Generate final visualization

### Results
- `GET /results/<filename>` - Serve result files

## 📊 Output Files

After completing the pipeline:

### `results/intrinsics.json`
```json
{
  "fx": 385.7,
  "fy": 385.2,
  "cx": 320.0,
  "cy": 240.0,
  "width": 640,
  "height": 480
}
```

### `results/analysis_summary.json`
```json
{
  "human_keypoints": {
    "right_wrist": [0.123, 0.456, 1.234],
    ...
  },
  "cups": [
    {
      "label": "target_1",
      "position_m": [1.163, 0.818, 2.618]
    },
    ...
  ],
  "relative_positions": {
    "closest_cup": "target_2",
    "results": [...]
  }
}
```

### Visualizations
- `apriltag_detected.jpg` - AprilTag detection result
- `human_detected.jpg` - Human pose keypoints
- `cups_detected.jpg` - Cup positions marked
- `final_3d_visualization.png` - 3D plot of entire scene

## 🔧 Configuration

### Using AprilTag for Camera Calibration

If you have an AprilTag with known size in your scene:

1. Print an AprilTag (e.g., from https://april.eecs.umich.edu/)
2. Measure its size (e.g., 0.16m)
3. Place it in view of the camera
4. Select "AprilTag Auto-Calibration" in Step 2

### Manual Calibration

If you know your camera intrinsics:

```python
fx, fy = 385.7, 385.2  # Focal lengths in pixels
cx, cy = 320.0, 240.0  # Principal point (image center)
```

### Loading from Config

Place your camera config at:
```
../config/camera_config.yaml
```

Format:
```yaml
cam3:
  fx: 388.5696716308594
  fy: 388.11639404296875
  cx: 319.456787109375
  cy: 248.93724060058594
  width: 640
  height: 480
```

## 🐛 Troubleshooting

### "No AprilTag detected"
- Make sure AprilTag is clearly visible
- Ensure good lighting
- Try a larger tag size

### "No human detected"
- Ensure person is fully visible in frame
- Check lighting conditions
- Person should be facing camera

### "Invalid depth image"
- Verify depth file is in RAW format (uint16)
- Check resolution matches color image

### Import errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# For AprilTag on Mac:
brew install boost eigen
pip install apriltag
```

## 🚀 Future Enhancements

- [ ] Support for multiple people
- [ ] Automatic cup detection (using object detection)
- [ ] Real-time webcam mode
- [ ] Export to different formats (CSV, JSON, MAT)
- [ ] Batch processing multiple trials
- [ ] Integration with dog tracking pipeline

## 📝 Notes

- Depth files should be raw uint16 format (2 bytes per pixel)
- Images will be automatically resized to match camera intrinsics
- Results are saved in `results/` folder
- Pipeline state is reset when server restarts

## 📄 License

MIT License - Feel free to use and modify!