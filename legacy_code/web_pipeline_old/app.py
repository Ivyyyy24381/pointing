"""
Flask Web Interface for Pointing Gesture Analysis Pipeline
Author: Claude + Ivy
Date: 2025-01-09

Simple Version: Human Detection + Cup Detection + Relative Position
With AprilTag camera calibration
Supports .raw and .npy depth formats
"""

import os
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import yaml
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create necessary folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Global state to track pipeline progress
pipeline_state = {
    'current_step': 0,
    'steps': [
        'upload_data',
        'calibrate_camera',
        'detect_apriltag',
        'detect_human',
        'detect_cups',
        'compute_relative_position',
        'visualize_results'
    ],
    'data': {}
}


# ============================================================================
# Utility Functions
# ============================================================================
def load_depth_file(depth_path, width, height):
    """
    Load depth file - supports .raw (uint16) and .npy formats
    Returns depth array in millimeters (uint16) or meters (float)
    """
    ext = os.path.splitext(depth_path)[1].lower()

    if ext == '.raw':
        # Load RAW format (uint16, in millimeters)
        with open(depth_path, 'rb') as f:
            depth_data = np.frombuffer(f.read(), dtype=np.uint16)

        # Try to reshape - check if size matches
        expected_size = width * height
        if depth_data.size != expected_size:
            # Try common resolutions
            common_resolutions = [
                (640, 480), (848, 480), (1280, 720), (1920, 1080)
            ]
            for w, h in common_resolutions:
                if depth_data.size == w * h:
                    print(f"⚠️ Depth resolution mismatch. Using {w}x{h} instead of {width}x{height}")
                    depth_img = depth_data.reshape((h, w))
                    # Resize to match color image
                    depth_img = cv2.resize(depth_img, (width, height), interpolation=cv2.INTER_NEAREST)
                    return depth_img

            raise ValueError(f"Cannot reshape depth data. Expected {expected_size}, got {depth_data.size}")

        depth_img = depth_data.reshape((height, width))

    elif ext == '.npy':
        # Load NPY format (can be uint16 or float)
        depth_img = np.load(depth_path)

        # Check if resize needed
        if depth_img.shape != (height, width):
            print(f"⚠️ Depth shape mismatch. Resizing from {depth_img.shape} to {(height, width)}")
            depth_img = cv2.resize(depth_img, (width, height), interpolation=cv2.INTER_NEAREST)

        # Convert to uint16 if float (assuming meters → millimeters)
        if depth_img.dtype == np.float32 or depth_img.dtype == np.float64:
            depth_img = (depth_img * 1000).astype(np.uint16)

    else:
        raise ValueError(f"Unsupported depth format: {ext}")

    return depth_img


def depth_to_meters(depth_img):
    """Convert depth image to meters (float32)"""
    if depth_img.dtype == np.uint16:
        return depth_img.astype(np.float32) / 1000.0
    return depth_img.astype(np.float32)


@app.route('/')
def index():
    """Main page - step-by-step workflow"""
    return render_template('index.html', state=pipeline_state)


@app.route('/reset', methods=['POST'])
def reset_pipeline():
    """Reset the entire pipeline"""
    global pipeline_state
    pipeline_state['current_step'] = 0
    pipeline_state['data'] = {}

    # Clean up uploads and results
    import shutil
    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        Path(folder).mkdir(exist_ok=True)

    return jsonify({'success': True, 'message': 'Pipeline reset successfully'})


# ============================================================================
# Step 1: Upload Data
# ============================================================================
@app.route('/step/upload', methods=['POST'])
def upload_data():
    """Upload color image, depth image (.raw or .npy)"""
    if 'color_image' not in request.files or 'depth_file' not in request.files:
        return jsonify({'success': False, 'error': 'Missing required files'})

    color_file = request.files['color_image']
    depth_file = request.files['depth_file']

    # Determine depth format from filename
    depth_filename = depth_file.filename
    depth_ext = os.path.splitext(depth_filename)[1].lower()

    if depth_ext not in ['.raw', '.npy']:
        return jsonify({'success': False, 'error': f'Depth file must be .raw or .npy, got {depth_ext}'})

    # Save files
    color_path = os.path.join(app.config['UPLOAD_FOLDER'], 'color.png')
    depth_path = os.path.join(app.config['UPLOAD_FOLDER'], f'depth{depth_ext}')

    color_file.save(color_path)
    depth_file.save(depth_path)

    # Load and validate color image
    color_img = cv2.imread(color_path)
    if color_img is None:
        return jsonify({'success': False, 'error': 'Invalid color image'})

    height, width = color_img.shape[:2]

    # Validate depth file
    try:
        depth_img = load_depth_file(depth_path, width, height)
        if depth_img is None:
            return jsonify({'success': False, 'error': 'Failed to load depth file'})

        # Get depth statistics
        valid_depth = depth_img[depth_img > 0]
        if len(valid_depth) > 0:
            depth_stats = {
                'min_mm': int(valid_depth.min()),
                'max_mm': int(valid_depth.max()),
                'mean_mm': int(valid_depth.mean()),
                'valid_pixels': len(valid_depth),
                'total_pixels': depth_img.size,
                'coverage_percent': float(len(valid_depth) / depth_img.size * 100)
            }
        else:
            depth_stats = {'error': 'No valid depth data'}

    except Exception as e:
        return jsonify({'success': False, 'error': f'Depth file error: {str(e)}'})

    # Store info
    pipeline_state['data']['color_path'] = color_path
    pipeline_state['data']['depth_path'] = depth_path
    pipeline_state['data']['depth_format'] = depth_ext
    pipeline_state['data']['image_size'] = [width, height]
    pipeline_state['current_step'] = 1

    return jsonify({
        'success': True,
        'message': f'Uploaded: {width}x{height}, depth: {depth_ext}',
        'depth_stats': depth_stats,
        'next_step': 'calibrate_camera'
    })


# ============================================================================
# Step 2: Camera Calibration (Manual or AprilTag)
# ============================================================================
@app.route('/step/calibrate', methods=['POST'])
def calibrate_camera():
    """
    Camera calibration using:
    1. Manual input (fx, fy, cx, cy)
    2. AprilTag detection (if present in image)
    3. Load from existing config
    """
    method = request.json.get('method', 'manual')

    if method == 'manual':
        # Manual intrinsics input
        intrinsics = {
            'fx': float(request.json.get('fx', 385.7)),
            'fy': float(request.json.get('fy', 385.2)),
            'cx': float(request.json.get('cx', 320.0)),
            'cy': float(request.json.get('cy', 240.0)),
            'width': pipeline_state['data']['image_size'][0],
            'height': pipeline_state['data']['image_size'][1]
        }

    elif method == 'apriltag':
        # Detect AprilTag for automatic calibration
        intrinsics = detect_apriltag_calibration()
        if intrinsics is None:
            return jsonify({'success': False, 'error': 'No AprilTag detected'})

    elif method == 'config':
        # Load from config file
        config_path = '../config/camera_config.yaml'
        if not os.path.exists(config_path):
            return jsonify({'success': False, 'error': 'Config file not found'})

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Use cam3 (side camera) by default
            intrinsics = {
                'fx': config['cam3']['fx'],
                'fy': config['cam3']['fy'],
                'cx': config['cam3']['cx'],
                'cy': config['cam3']['cy'],
                'width': config['cam3']['width'],
                'height': config['cam3']['height']
            }

    # Save intrinsics
    pipeline_state['data']['intrinsics'] = intrinsics

    # Save to file
    intrinsics_path = os.path.join(app.config['RESULTS_FOLDER'], 'intrinsics.json')
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics, f, indent=2)

    pipeline_state['current_step'] = 2

    return jsonify({
        'success': True,
        'intrinsics': intrinsics,
        'next_step': 'detect_apriltag'
    })


# ============================================================================
# Step 3: Detect AprilTag (for world coordinate system)
# ============================================================================
@app.route('/step/detect_apriltag', methods=['POST'])
def detect_apriltag():
    """
    Detect AprilTag to establish world coordinate system
    AprilTag provides known size and pose for camera-to-world transform
    """
    try:
        import apriltag
    except ImportError:
        # Skip AprilTag if not installed
        pipeline_state['data']['has_apriltag'] = False
        pipeline_state['current_step'] = 3
        return jsonify({
            'success': True,
            'has_apriltag': False,
            'message': 'AprilTag library not installed, skipping',
            'next_step': 'detect_human'
        })

    color_path = pipeline_state['data']['color_path']
    intrinsics = pipeline_state['data']['intrinsics']

    # Load image
    img = cv2.imread(color_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detector = apriltag.Detector()
    detections = detector.detect(gray)

    if len(detections) == 0:
        # No AprilTag - skip this step, use manual world coordinates
        pipeline_state['data']['has_apriltag'] = False
        pipeline_state['current_step'] = 3
        return jsonify({
            'success': True,
            'has_apriltag': False,
            'message': 'No AprilTag detected, will use manual coordinates',
            'next_step': 'detect_human'
        })

    # Process first detection
    detection = detections[0]
    tag_id = detection.tag_id
    corners = detection.corners

    # Estimate pose (assuming tag size = 0.16m)
    tag_size = 0.16
    camera_params = [intrinsics['fx'], intrinsics['fy'],
                     intrinsics['cx'], intrinsics['cy']]

    pose, _, _ = detector.detection_pose(detection, camera_params, tag_size)

    # Extract rotation and translation
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Visualize
    vis_img = img.copy()
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i+1)%4].astype(int))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)

    center = corners.mean(axis=0).astype(int)
    cv2.putText(vis_img, f'Tag {tag_id}', tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    vis_path = os.path.join(app.config['RESULTS_FOLDER'], 'apriltag_detected.jpg')
    cv2.imwrite(vis_path, vis_img)

    # Save results
    pipeline_state['data']['has_apriltag'] = True
    pipeline_state['data']['apriltag'] = {
        'tag_id': int(tag_id),
        'corners': corners.tolist(),
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'tag_size': tag_size
    }
    pipeline_state['current_step'] = 3

    return jsonify({
        'success': True,
        'has_apriltag': True,
        'tag_id': int(tag_id),
        'translation': t.tolist(),
        'visualization': 'apriltag_detected.jpg',
        'next_step': 'detect_human'
    })


# ============================================================================
# Step 4: Detect Human (MediaPipe)
# ============================================================================
@app.route('/step/detect_human', methods=['POST'])
def detect_human():
    """
    Detect human using MediaPipe Pose
    Extract wrist, shoulder, elbow positions with depth
    """
    import mediapipe as mp

    color_path = pipeline_state['data']['color_path']
    depth_path = pipeline_state['data']['depth_path']
    intrinsics = pipeline_state['data']['intrinsics']

    # Load images
    img = cv2.imread(color_path)
    height, width = img.shape[:2]

    # Load depth with proper format handling
    depth_img_mm = load_depth_file(depth_path, width, height)
    depth_img_m = depth_to_meters(depth_img_mm)

    # MediaPipe Pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return jsonify({'success': False, 'error': 'No human detected'})

    # Extract key points
    landmarks = results.pose_landmarks.landmark

    # Get 2D keypoints and project to 3D
    keypoints_2d = {}
    keypoints_3d = {}

    key_parts = {
        'nose': mp_pose.PoseLandmark.NOSE,
        'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
        'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
        'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
        'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    }

    for name, landmark_id in key_parts.items():
        lm = landmarks[landmark_id]
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)

        # Get depth
        if 0 <= x_px < width and 0 <= y_px < height:
            depth_m = depth_img_m[y_px, x_px]

            # Project to 3D
            if depth_m > 0:
                X = (x_px - intrinsics['cx']) * depth_m / intrinsics['fx']
                Y = (y_px - intrinsics['cy']) * depth_m / intrinsics['fy']
                Z = depth_m

                keypoints_2d[name] = [x_px, y_px]
                keypoints_3d[name] = [float(X), float(Y), float(Z)]

    # Visualize
    mp_drawing = mp.solutions.drawing_utils
    vis_img = img.copy()
    mp_drawing.draw_landmarks(vis_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Add depth colormap overlay
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img_mm, alpha=0.03),
        cv2.COLORMAP_JET
    )
    vis_combined = cv2.addWeighted(vis_img, 0.7, depth_colormap, 0.3, 0)

    vis_path = os.path.join(app.config['RESULTS_FOLDER'], 'human_detected.jpg')
    cv2.imwrite(vis_path, vis_combined)

    # Save results
    pipeline_state['data']['human'] = {
        'keypoints_2d': keypoints_2d,
        'keypoints_3d': keypoints_3d
    }
    pipeline_state['current_step'] = 4

    return jsonify({
        'success': True,
        'keypoints_2d': keypoints_2d,
        'keypoints_3d': keypoints_3d,
        'visualization': 'human_detected.jpg',
        'next_step': 'detect_cups'
    })


# ============================================================================
# Step 5: Detect Cups (Manual Selection)
# ============================================================================
@app.route('/step/detect_cups', methods=['POST'])
def detect_cups():
    """
    Detect 4 cups using manual ROI selection
    User clicks on web interface to mark cup centers
    """
    cup_clicks = request.json.get('cup_clicks', [])

    if len(cup_clicks) != 4:
        return jsonify({'success': False, 'error': f'Need 4 cups, got {len(cup_clicks)}'})

    color_path = pipeline_state['data']['color_path']
    depth_path = pipeline_state['data']['depth_path']
    intrinsics = pipeline_state['data']['intrinsics']

    # Load images
    img = cv2.imread(color_path)
    height, width = img.shape[:2]

    depth_img_mm = load_depth_file(depth_path, width, height)
    depth_img_m = depth_to_meters(depth_img_mm)

    cups_3d = []
    vis_img = img.copy()

    for i, click in enumerate(cup_clicks):
        x_px = int(click['x'])
        y_px = int(click['y'])

        # Sample depth in 5x5 window around click
        x1, y1 = max(0, x_px - 2), max(0, y_px - 2)
        x2, y2 = min(width, x_px + 3), min(height, y_px + 3)
        depth_roi = depth_img_m[y1:y2, x1:x2]

        # Average non-zero depths
        valid_depths = depth_roi[depth_roi > 0]
        if len(valid_depths) > 0:
            depth_m = float(np.median(valid_depths))
        else:
            depth_m = 2.0  # Default fallback

        # Project to 3D
        X = (x_px - intrinsics['cx']) * depth_m / intrinsics['fx']
        Y = (y_px - intrinsics['cy']) * depth_m / intrinsics['fy']
        Z = depth_m

        cups_3d.append({
            'label': f'target_{i+1}',
            'pixel': [x_px, y_px],
            'position_m': [float(X), float(Y), float(Z)],
            'depth_m': float(depth_m)
        })

        # Visualize
        cv2.circle(vis_img, (x_px, y_px), 10, (0, 255, 0), 2)
        cv2.putText(vis_img, f'Cup {i+1}', (x_px + 15, y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    vis_path = os.path.join(app.config['RESULTS_FOLDER'], 'cups_detected.jpg')
    cv2.imwrite(vis_path, vis_img)

    # Save results
    pipeline_state['data']['cups'] = cups_3d
    pipeline_state['current_step'] = 5

    return jsonify({
        'success': True,
        'cups': cups_3d,
        'visualization': 'cups_detected.jpg',
        'next_step': 'compute_relative_position'
    })


# ============================================================================
# Step 6: Compute Relative Positions
# ============================================================================
@app.route('/step/compute_positions', methods=['POST'])
def compute_relative_positions():
    """
    Compute:
    1. Distance from human wrists to each cup
    2. Pointing direction (wrist - shoulder vector)
    3. Which cup is closest to pointing ray
    """
    human = pipeline_state['data']['human']
    cups = pipeline_state['data']['cups']

    # Get wrist positions (use right wrist as default)
    wrist_3d = human['keypoints_3d'].get('right_wrist')
    shoulder_3d = human['keypoints_3d'].get('right_shoulder')
    elbow_3d = human['keypoints_3d'].get('right_elbow')

    if not wrist_3d or not shoulder_3d:
        # Try left side
        wrist_3d = human['keypoints_3d'].get('left_wrist')
        shoulder_3d = human['keypoints_3d'].get('left_shoulder')
        elbow_3d = human['keypoints_3d'].get('left_elbow')

    if not wrist_3d or not shoulder_3d:
        return jsonify({'success': False, 'error': 'Cannot find wrist/shoulder'})

    wrist = np.array(wrist_3d)
    shoulder = np.array(shoulder_3d)

    # Compute pointing vector
    pointing_vec = wrist - shoulder
    pointing_vec_norm = pointing_vec / np.linalg.norm(pointing_vec)

    # Compute distances to each cup
    results = []

    for cup in cups:
        cup_pos = np.array(cup['position_m'])

        # Euclidean distance
        dist = np.linalg.norm(cup_pos - wrist)

        # Distance to pointing ray (perpendicular distance)
        wrist_to_cup = cup_pos - wrist
        projection = np.dot(wrist_to_cup, pointing_vec_norm)
        perpendicular = wrist_to_cup - projection * pointing_vec_norm
        ray_dist = np.linalg.norm(perpendicular)

        results.append({
            'cup_label': cup['label'],
            'euclidean_distance_m': float(dist),
            'ray_distance_m': float(ray_dist),
            'projection_along_ray_m': float(projection)
        })

    # Find closest cup (by ray distance)
    closest_cup = min(results, key=lambda x: x['ray_distance_m'])

    # Save results
    pipeline_state['data']['relative_positions'] = {
        'pointing_vector': pointing_vec.tolist(),
        'wrist_position': wrist.tolist(),
        'shoulder_position': shoulder.tolist(),
        'results': results,
        'closest_cup': closest_cup['cup_label']
    }

    pipeline_state['current_step'] = 6

    return jsonify({
        'success': True,
        'pointing_vector': pointing_vec.tolist(),
        'results': results,
        'closest_cup': closest_cup['cup_label'],
        'next_step': 'visualize_results'
    })


# ============================================================================
# Step 7: Visualize Results
# ============================================================================
@app.route('/step/visualize', methods=['POST'])
def visualize_results():
    """
    Create final visualization:
    1. 3D scatter plot of human + cups
    2. Pointing ray
    3. Distance annotations
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    human = pipeline_state['data']['human']
    cups = pipeline_state['data']['cups']
    rel_pos = pipeline_state['data']['relative_positions']

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cups
    for cup in cups:
        pos = cup['position_m']
        color = 'red' if cup['label'] == rel_pos['closest_cup'] else 'orange'
        ax.scatter(pos[0], pos[1], pos[2], c=color, marker='o', s=200, label=cup['label'])
        ax.text(pos[0], pos[1], pos[2], cup['label'], fontsize=10)

    # Plot human keypoints
    for name, pos in human['keypoints_3d'].items():
        ax.scatter(pos[0], pos[1], pos[2], c='blue', marker='x', s=100)

    # Plot pointing ray
    wrist = np.array(rel_pos['wrist_position'])
    pointing_vec = np.array(rel_pos['pointing_vector'])

    # Extend ray to 2 meters
    ray_end = wrist + pointing_vec / np.linalg.norm(pointing_vec) * 2.0
    ax.plot([wrist[0], ray_end[0]],
            [wrist[1], ray_end[1]],
            [wrist[2], ray_end[2]], 'g-', linewidth=3, label='Pointing Ray')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Human Pointing Analysis - Closest Cup: {rel_pos["closest_cup"]}')
    ax.legend()

    vis_path = os.path.join(app.config['RESULTS_FOLDER'], 'final_3d_visualization.png')
    plt.savefig(vis_path, dpi=150)
    plt.close()

    # Save summary JSON
    summary = {
        'human_keypoints': human['keypoints_3d'],
        'cups': cups,
        'relative_positions': rel_pos,
        'closest_cup': rel_pos['closest_cup']
    }

    summary_path = os.path.join(app.config['RESULTS_FOLDER'], 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    pipeline_state['current_step'] = 7

    return jsonify({
        'success': True,
        'visualization': 'final_3d_visualization.png',
        'summary': summary,
        'complete': True
    })


# ============================================================================
# Helper Functions
# ============================================================================
def detect_apriltag_calibration():
    """Detect AprilTag and estimate camera intrinsics"""
    # This is a simplified version - full implementation would use
    # multiple views of AprilTag to estimate intrinsics
    return None


@app.route('/results/<path:filename>')
def serve_result(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/status')
def get_status():
    """Get current pipeline status"""
    return jsonify({
        'current_step': pipeline_state['current_step'],
        'current_step_name': pipeline_state['steps'][pipeline_state['current_step']],
        'total_steps': len(pipeline_state['steps']),
        'has_data': len(pipeline_state['data']) > 0
    })


if __name__ == '__main__':
    print("🚀 Starting Pointing Analysis Web Interface")
    print("📍 Open http://localhost:5000 in your browser")
    print("📊 Supported depth formats: .raw (uint16) and .npy")
    app.run(debug=True, host='0.0.0.0', port=5000)
