"""
Flask Web Interface for Pointing Gesture Analysis Pipeline
Step-by-step implementation template

Author: Ivy
Date: 2025-01-09
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Create folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Global state
pipeline_state = {
    'current_step': 0,
    'data': {}
}


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/status')
def status():
    """Get pipeline status"""
    return jsonify(pipeline_state)


@app.route('/reset', methods=['POST'])
def reset():
    """Reset pipeline"""
    global pipeline_state
    pipeline_state = {'current_step': 0, 'data': {}}

    # TODO: Clean up uploaded files

    return jsonify({'success': True, 'message': 'Pipeline reset'})


# ============================================================================
# TODO: Implement steps one by one
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """
    Step 1: Upload color and depth files
    TODO: Implement file upload and validation
    """
    return jsonify({'success': False, 'error': 'Not implemented yet'})


@app.route('/api/calibrate', methods=['POST'])
def calibrate_camera():
    """
    Step 2: Camera calibration
    TODO: Implement calibration (manual input or from config)
    """
    return jsonify({'success': False, 'error': 'Not implemented yet'})


@app.route('/api/detect_human', methods=['POST'])
def detect_human():
    """
    Step 3: Detect human using MediaPipe
    TODO: Implement human detection
    """
    return jsonify({'success': False, 'error': 'Not implemented yet'})


@app.route('/api/detect_cups', methods=['POST'])
def detect_cups():
    """
    Step 4: Detect cups from user clicks
    TODO: Implement cup detection with depth projection
    """
    return jsonify({'success': False, 'error': 'Not implemented yet'})


@app.route('/api/compute_positions', methods=['POST'])
def compute_positions():
    """
    Step 5: Compute relative positions and pointing direction
    TODO: Implement distance and ray computation
    """
    return jsonify({'success': False, 'error': 'Not implemented yet'})


@app.route('/api/visualize', methods=['POST'])
def visualize():
    """
    Step 6: Generate 3D visualization
    TODO: Implement 3D plotting
    """
    return jsonify({'success': False, 'error': 'Not implemented yet'})


# ============================================================================
# Serve static files
# ============================================================================

@app.route('/results/<path:filename>')
def serve_result(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Pointing Gesture Analysis - Web Interface")
    print("=" * 60)
    print(f"📍 URL: http://localhost:5000")
    print(f"📁 Uploads: {app.config['UPLOAD_FOLDER']}")
    print(f"📊 Results: {app.config['RESULTS_FOLDER']}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
