---
title: pointing_data markmap
markmap:
  colorFreezeLevel: 3
---

# pointing data
## pointing_production(front_view)
### trials (e.g. 1)
#### Color
##### Color_%(FRAME_ID).png - RGB color frames extracted from rosbag (640x480 or 1280x720)
##### f_%(FRAME_ID).png - Alternative naming format for color frames
#### Depth
##### Depth_%(FRAME_ID).png - Depth visualization frames as PNG images
##### Depth_%(FRAME_ID).raw - Raw depth data in uint16 format (matching color frame dimensions)
#### Depth_Color
##### Depth_Color_%(FRAME_ID).png - Colorized depth images for visualization
#### Color.mp4 - Compiled color video from all trial frames
#### Depth.mp4 - Compiled depth video from all trial frames  
#### Depth_Color.mp4 - Compiled colorized depth video
#### 2d_pointing_trace.png - Visualization of detected 2D pointing gesture traces over time
#### gesture_data.csv - Raw gesture detection data with keypoint coordinates and confidence scores
#### processed_gesture_data.csv - Post-processed gesture data with smoothing and filtering applied
### auto_split.csv - Automatic trial segmentation metadata with start/end timestamps
### Color.mp4 - Full session color video before trial splitting
### Depth.mp4 - Full session depth video before trial splitting
### Depth_Color.mp4 - Full session colorized depth video before trial splitting  
### rosbag_metadata.yaml - Camera intrinsics and recording parameters extracted from rosbag
### {subject_id}_front_gesture_data.csv - Combined gesture data across all trials for the subject

## pointing_comprehension(side_view)
### trials (e.g. 1)
#### Color
##### Color_%(FRAME_ID).png - RGB color frames standardized to 1280x720 to match camera intrinsics
#### Depth  
##### Depth_%(FRAME_ID).png - Depth visualization frames standardized to 1280x720
##### Depth_%(FRAME_ID).raw - Raw uint16 depth data standardized to 1280x720 dimensions
#### Depth_Color
##### Depth_Color_%(FRAME_ID).png - Colorized depth images standardized to 1280x720
#### segmented_color - Directory containing SAM2 segmentation visualization frames
#### Color.mp4 - Trial color video compiled from standardized frames at original framerate (15-25 FPS)
#### Depth.mp4 - Trial depth video compiled from standardized frames
#### Depth_Color.mp4 - Trial colorized depth video compiled from standardized frames
#### masked_video.mp4 - SAM2-segmented video with subject mask applied (10-12.5 FPS after sampling)
#### masked_video_annotated.mp4 - SAM2 segmentation with visual overlays showing mask boundaries
#### subject_annotated_video.mp4 - Final output with pose skeleton, bounding boxes, and target distance annotations (10-12.5 FPS)
#### masked_video_skeleton.json - MediaPipe pose detection results for each frame with bodypart coordinates and confidence scores
#### interactive_points.json - User-labeled foreground/background points for SAM2 segmentation training
#### target_coordinates.json - 3D world coordinates of reference targets for distance calculation
#### sam2_scale_metadata.json - Scale factors and processing metadata from SAM2 segmentation pipeline
#### processed_subject_result_distance_comparison.png - Plot showing distance to each target over time
#### processed_subject_result_trace.png - 2D trajectory plot of subject center-of-mass movement
#### processed_subject_result_trace3d.png - 3D trajectory visualization of subject movement in world coordinates
#### processed_subject_result_table.csv - Comprehensive frame-by-frame data including:
  - Frame timing (frame_index, time_sec, local_frame_index)
  - Target distances (human_r, target_N_r) and angles (theta, phi) 
  - Bounding box coordinates (bbox_x, bbox_y, bbox_w, bbox_h)
  - All MediaPipe keypoint coordinates in 2D pixels and 3D world meters
  - Confidence scores for each detected keypoint
  - Subject orientation vectors (head_orientation, torso_orientation)
  - 3D center-of-mass trajectory (trace3d_x, trace3d_y, trace3d_z)
### auto_split.csv - Automatic trial segmentation metadata with start/end timestamps
### Color.mp4 - Full session color video before trial splitting
### Depth.mp4 - Full session depth video before trial splitting  
### Depth_Color.mp4 - Full session colorized depth video before trial splitting
### rosbag_metadata.yaml - Camera intrinsics (fx, fy, ppx, ppy, width, height) and recording parameters
### {subject_id}_side_combined_result.csv - Aggregated results across all trials for comprehensive analysis

## File Processing Pipeline
### Stage 1: Rosbag Processing
- Extract rosbag_metadata.yaml (camera intrinsics, recording metadata)
- Generate Color.mp4, Depth.mp4, Depth_Color.mp4 (full session videos)
- Create auto_split.csv (trial boundary detection)
- Extract individual frame images to Color/, Depth/, Depth_Color/ directories

### Stage 2: Image Standardization
- Resize all color/depth images to 1280x720 to match camera intrinsics
- Maintain temporal alignment and preserve depth data accuracy
- Update metadata to reflect new dimensions

### Stage 3: SAM2 Segmentation (side_view only)
- Load interactive_points.json for user-labeled training points
- Apply frame sampling (every 2-3 frames) for memory management
- Generate masked_video.mp4 and masked_video_annotated.mp4
- Save sam2_scale_metadata.json with processing parameters

### Stage 4: Pose Detection (side_view only) 
- Run MediaPipe pose detection on masked_video.mp4
- Generate masked_video_skeleton.json with keypoint coordinates
- Apply coordinate corrections based on scale metadata

### Stage 5: 3D Analysis (side_view only)
- Convert 2D keypoints to 3D world coordinates using camera intrinsics
- Calculate distances to reference targets from target_coordinates.json  
- Generate trajectory plots and distance comparison visualizations
- Create comprehensive processed_subject_result_table.csv
- Produce subject_annotated_video.mp4 with all annotations overlaid