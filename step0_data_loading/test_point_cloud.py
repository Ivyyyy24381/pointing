#!/usr/bin/env python3
"""
Quick test script to create a point cloud from color and depth images.

Usage:
    python test_point_cloud.py <trial_name> <camera_id> <frame_number>
    python test_point_cloud.py trial_1 cam1 100
    python test_point_cloud.py 1 None 50  # Single camera trial
"""

import sys
from pathlib import Path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step0_data_loading.trial_input_manager import TrialInputManager
from utils.point_cloud_utils import create_point_cloud, visualize_point_cloud, save_point_cloud


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    trial_name = sys.argv[1]
    camera_id = sys.argv[2] if sys.argv[2] != 'None' else None
    frame_number = int(sys.argv[3])

    # Initialize manager
    manager = TrialInputManager()

    # Load frame from trial_input/ (will auto-process if needed)
    print(f"Loading frame {frame_number} from {trial_name} (camera: {camera_id})...")
    color, depth = manager.load_frame(trial_name, camera_id, frame_number)

    if color is None or depth is None:
        print("❌ Failed to load frame")
        sys.exit(1)

    print(f"✅ Loaded color: {color.shape}, depth: {depth.shape}")
    print(f"   Depth range: [{depth[depth > 0].min():.3f}, {depth.max():.3f}] meters")

    # Create point cloud (intrinsics auto-estimated)
    print("Creating point cloud...")
    pcd = create_point_cloud(color, depth)

    print(f"✅ Created point cloud with {len(pcd.points)} points")

    # Save point cloud
    output_dir = Path("trial_input") / f"{trial_name}_{camera_id if camera_id else 'single'}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"frame_{frame_number:06d}.ply"

    save_point_cloud(pcd, str(output_file))

    # Visualize
    print("🎨 Visualizing point cloud (close window to exit)...")
    visualize_point_cloud(pcd, window_name=f"{trial_name} - Frame {frame_number}")


if __name__ == "__main__":
    main()
