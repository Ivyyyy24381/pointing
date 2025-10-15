---
title: pointing_data markmap
markmap:
  colorFreezeLevel: 3
---

# Pointing Data

## 1. pointing_production (front_view)

### ▸ Trials (e.g. 1)

#### ▹ Color
- `Color_%(FRAME_ID).png`: RGB color frames extracted from rosbag (640×480 or 1280×720)
- `f_%(FRAME_ID).png`: Alternative naming format

#### ▹ Depth
- `Depth_%(FRAME_ID).png`: Depth visualization frames
- `Depth_%(FRAME_ID).raw`: Raw uint16 depth data (same resolution as color)

#### ▹ Depth_Color
- `Depth_Color_%(FRAME_ID).png`: Colorized depth images

#### ▹ Videos (per-trial)
- `Color.mp4`: Color video compiled from all trial frames
- `Depth.mp4`: Depth video compiled from all trial frames
- `Depth_Color.mp4`: Colorized depth video compiled from all trial frames

#### ▹ Gesture Analysis
- `2d_pointing_trace.png`: Visualization of detected 2D pointing traces
- `gesture_data.csv`: Raw keypoint coordinates and confidence scores
- `processed_gesture_data.csv`: Smoothed and filtered gesture data

### ▸ Metadata & Session Files
- `auto_split.csv`: Automatic trial segmentation with start/end timestamps
- `Color.mp4`: Full session color video before trial splitting
- `Depth.mp4`: Full session depth video before trial splitting
- `Depth_Color.mp4`: Full session colorized depth video before trial splitting
- `rosbag_metadata.yaml`: Camera intrinsics and rosbag parameters
- `{subject_id}_front_gesture_data.csv`: Combined gesture data across all trials

---

## 2. pointing_comprehension (side_view)

### ▸ Trials (e.g. 1)

#### ▹ Color
- `Color_%(FRAME_ID).png`: RGB color frames standardized to 1280×720

#### ▹ Depth
- `Depth_%(FRAME_ID).png`: Depth visualization (1280×720)
- `Depth_%(FRAME_ID).raw`: Raw uint16 depth (1280×720)

#### ▹ Depth_Color
- `Depth_Color_%(FRAME_ID).png`: Colorized depth images (1280×720)

#### ▹ Segmentation Outputs
- `segmented_color/`: SAM2 visualization frames
- `interactive_points.json`: User-labeled foreground/background points
- `sam2_scale_metadata.json`: Scale factors and SAM2 processing metadata

#### ▹ Videos (per-trial)
- `Color.mp4`: Trial color video (15–25 FPS)
- `Depth.mp4`: Trial depth video
- `Depth_Color.mp4`: Trial colorized depth video
- `masked_video.mp4`: SAM2-segmented video (10–12.5 FPS)
- `masked_video_annotated.mp4`: Segmentation + mask boundary overlays
- `subject_annotated_video.mp4`: Pose + bounding box + distance overlays

#### ▹ Pose / Target Analysis
- `masked_video_skeleton.json`: MediaPipe pose detections per frame
- `target_coordinates.json`: 3D world coordinates of reference targets
- `processed_subject_result_distance_comparison.png`: Distance-to-target over time
- `processed_subject_result_trace.png`: 2D center-of-mass trajectory
- `processed_subject_result_trace3d.png`: 3D trajectory in world coordinates

#### ▹ Comprehensive CSV Output
- `processed_subject_result_table.csv`: Frame-by-frame analysis including:
  - Frame timing (`frame_index`, `time_sec`, `local_frame_index`)
  - Target distances and angles (`human_r`, `target_N_r`, `theta`, `phi`)
  - Bounding box (`bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`)
  - 2D and 3D keypoints
  - Keypoint confidence scores
  - Orientation vectors (`head_orientation`, `torso_orientation`)
  - 3D trajectory (`trace3d_x/y/z`)

### ▸ Metadata & Session Files
- `auto_split.csv`: Automatic trial segmentation
- `Color.mp4`: Full session color video (pre-trial)
- `Depth.mp4`: Full session depth video (pre-trial)
- `Depth_Color.mp4`: Full session colorized depth video (pre-trial)
- `rosbag_metadata.yaml`: Intrinsics (fx, fy, ppx, ppy, width, height)
- `{subject_id}_side_combined_result.csv`: Aggregated results across all trials