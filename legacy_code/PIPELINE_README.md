# Enhanced Computer Vision Pipeline with Scale Correction and Target Labeling

This document describes the enhanced computer vision pipeline that fixes coordinate mapping issues and adds manual target labeling capabilities.

## 🔧 Problems Fixed

### 1. SAM2VideoSegmenter Scale Issues
- **Problem**: Images resized to max_size=360 for SAM2 processing, but masks applied to original resolution
- **Solution**: Added scale metadata tracking and proper mask rescaling
- **Files**: `dog_script/segmentation.py`

### 2. Coordinate Mapping Misalignment  
- **Problem**: Pose keypoints from masked video didn't align with original depth data
- **Solution**: Added coordinate correction functions that detect and fix scale mismatches
- **Files**: `dog_script/dog_pose_visualize.py`

### 3. Static Target Limitations
- **Problem**: Fixed target positions in config files didn't reflect actual trial setups
- **Solution**: Created interactive target labeling tool for per-trial target annotation
- **Files**: `label_targets.py`

## 🚀 New Features

### 1. Interactive Target Labeling (`label_targets.py`)
- Manual target annotation using GUI interface
- Automatic 2D→3D coordinate conversion using depth data
- Per-trial target storage with metadata
- Handles coordinate scaling between display and original resolutions

### 2. Enhanced Distance Calculations
- Supports both manual and static targets
- Adds closest target information to outputs
- Tracks labeling method used (manual/static)
- Enhanced visualization with target method indicators

### 3. Comprehensive Pipeline Validation (`validate_pipeline.py`)
- Scale factor accuracy validation
- Depth alignment testing
- Target coordinate validation
- Distance calculation verification
- Coordinate transformation testing
- Pipeline output completeness checks

### 4. Integrated Pipeline Controller (`integrated_pipeline.py`)
- Coordinates all pipeline steps
- Handles scale metadata between components
- Provides force reprocessing options
- Comprehensive error handling and reporting

## 📁 File Structure

```
pointing/
├── dog_script/
│   ├── segmentation.py              # Enhanced SAM2 with scale metadata
│   └── dog_pose_visualize.py        # Enhanced pose processing with coordinate correction
├── config/
│   ├── camera_config.yaml           # Camera intrinsics
│   ├── skeleton_config.json         # Skeleton joint definitions
│   └── targets.yaml                 # Static target fallback
├── label_targets.py                 # Interactive target labeling tool
├── validate_pipeline.py             # Comprehensive validation suite
├── integrated_pipeline.py           # Main pipeline controller
└── PIPELINE_README.md              # This documentation
```

## 🎯 Usage Guide

### Quick Start - Full Pipeline
```bash
# Process all trials with enhanced pipeline
python integrated_pipeline.py --base_dir /path/to/trials

# Process single trial
python integrated_pipeline.py --base_dir /path/to/trials --trial_dir /path/to/specific/trial

# Force reprocessing and re-labeling
python integrated_pipeline.py --base_dir /path/to/trials --force_reprocess --force_relabel
```

### Step-by-Step Usage

#### 1. Target Labeling (Optional but Recommended)
```bash
# Label targets for all trials
python label_targets.py --base_dir /path/to/trials

# Label single trial
python label_targets.py --trial_dir /path/to/trial

# Force re-labeling existing targets
python label_targets.py --base_dir /path/to/trials --force
```

#### 2. Run Enhanced Segmentation
```bash
# The segmentation now saves scale metadata automatically
# Run through integrated pipeline or directly:
python dog_script/segmentation.py
```

#### 3. Pose Processing with Coordinate Correction
```bash
# Run enhanced pose processing (automatically detects manual targets)
python dog_script/dog_pose_visualize.py --json_path /path/to/skeleton.json
```

#### 4. Validate Results
```bash
# Validate single trial
python validate_pipeline.py --trial_dir /path/to/trial

# Validate all trials
python validate_pipeline.py --base_dir /path/to/trials
```

### Specific Pipeline Steps
```bash
# Run only specific steps
python integrated_pipeline.py --base_dir /path/to/trials --steps segmentation targets
python integrated_pipeline.py --base_dir /path/to/trials --steps pose validation
```

## 📊 Enhanced Outputs

### 1. Scale Metadata (`sam2_scale_metadata.json`)
```json
{
  "scale_factors": {
    "frame_001.jpg": [3.56, 2.0]
  },
  "original_dimensions": {
    "frame_001.png": [1280, 720]
  },
  "processing_dimensions": {
    "frame_001.jpg": [360, 360]
  },
  "processing_info": {
    "resized_for_sam2": true,
    "frame_count": 500
  }
}
```

### 2. Target Coordinates (`target_coordinates.json`)
```json
{
  "targets": [
    {
      "id": 1,
      "label": "target_1",
      "pixel_coords": [913, 552],
      "world_coords": [1.163, 0.818, 2.618],
      "depth_m": 2.618
    }
  ],
  "labeling_frame": "Color_000250.png",
  "original_dimensions": {"width": 1280, "height": 720},
  "camera_intrinsics": {...},
  "labeling_metadata": {...}
}
```

### 3. Enhanced CSV Output
New columns in `processed_subject_result_table.csv`:
- `target_1_r`, `target_2_r`, etc. (distances to each target)
- `closest_target_distance` (distance to nearest target)
- `closest_target_label` (which target is closest)
- `target_labeling_method` (manual/static)
- `interpolated_distances` (0/1 flag for interpolated values)

### 4. Enhanced Visualizations
- 3D plots show manual vs static target labeling
- Distance plots include target method indicators
- Validation reports with detailed diagnostics

## 🔍 Coordinate System Details

### Scale Factor Handling
1. **Original Images**: PNG files at full resolution (e.g., 1280×720)
2. **SAM2 Processing**: Resized to max_size=360 with preserved aspect ratio
3. **Scale Factors**: Stored as (width_scale, height_scale) ratios
4. **Coordinate Correction**: Applied when mapping between resolutions

### Target Coordinate Systems
- **Pixel Coordinates**: In original image resolution
- **World Coordinates**: 3D positions in meters (camera coordinate frame)
- **Depth Integration**: Uses camera intrinsics for 2D→3D conversion

### Validation Checks
- Coordinate transformation accuracy (< 0.1 pixel error)
- Depth alignment ratio (> 30% valid mappings)
- Target position sanity checks
- Scale factor consistency

## ⚠️ Important Notes

### Backward Compatibility
- Pipeline maintains compatibility with existing static targets
- Falls back gracefully when manual targets unavailable
- Existing skeleton detection workflows unchanged

### Performance Considerations
- Scale metadata adds minimal overhead
- Target labeling is one-time per trial
- Validation can be disabled for faster processing

### Error Handling
- Graceful degradation when components fail
- Comprehensive error reporting
- Force reprocessing options for recovery

## 🧪 Testing and Validation

### Validation Categories
1. **Scale Factor Accuracy**: Verifies coordinate transformations
2. **Depth Alignment**: Tests 2D keypoint → depth mapping  
3. **Target Coordinates**: Validates manual target positions
4. **Distance Calculations**: Checks output consistency
5. **Pipeline Outputs**: Ensures all expected files generated

### Quality Metrics
- Coordinate alignment ratio (target: >30%)
- Transformation accuracy (target: <0.1 pixel error)
- Target spacing validation (target: >0.1m minimum separation)
- Pipeline completion rate

## 📞 Troubleshooting

### Common Issues

#### Scale Metadata Missing
```bash
# Re-run segmentation to generate metadata
python integrated_pipeline.py --base_dir /path/to/trials --steps segmentation --force_reprocess
```

#### Poor Coordinate Alignment
- Check camera intrinsics accuracy
- Verify depth data quality
- Validate scale factor calculations

#### Target Labeling Issues
- Ensure depth data exists and is valid
- Check display scaling in labeling tool
- Verify camera intrinsics loaded correctly

#### Distance Calculation Errors
- Confirm target coordinates exist
- Check coordinate system consistency
- Validate 3D reconstruction accuracy

### Debug Information
Enable detailed logging by setting environment variables:
```bash
export PIPELINE_DEBUG=1
export COORDINATE_DEBUG=1
```

## 🔄 Migration from Old Pipeline

1. **Backup existing results**
2. **Run target labeling** on important trials
3. **Reprocess with new pipeline** using `--force_reprocess`
4. **Validate results** using validation tools
5. **Compare outputs** between old and new pipelines

## 📈 Future Enhancements

- Multi-camera coordinate system alignment
- Automated target detection algorithms
- Real-time coordinate validation
- Advanced interpolation methods
- GPU acceleration for large datasets

---

For questions or issues, please refer to the validation tools or create detailed error reports including trial directory structure and processing logs.