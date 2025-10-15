"""
Data Manager - High-level interface for loading multiple trials

This module provides a unified interface to discover, configure, and load
data from folders containing multiple trials with different structures.

Features:
- Auto-discovers trials in a folder
- Detects structure and configuration for each trial
- Loads frames from any trial with consistent API
- Caches metadata for fast access

Usage:
    from data_manager import DataManager

    # Initialize with root folder
    dm = DataManager("sample_raw_data")

    # List available trials
    print(dm.list_trials())

    # Load a frame
    color, depth = dm.load_frame("trial_1", "cam1", 31)
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Import our flexible loader
sys.path.insert(0, os.path.dirname(__file__))
from load_trial_data_flexible import (
    detect_folder_structure,
    load_trial_data_flexible,
    list_available_cameras,
    load_color_flexible,
    load_depth_flexible,
    find_color_file,
    find_depth_file
)


class TrialInfo:
    """Container for trial metadata"""

    def __init__(self, trial_name: str, trial_path: str, structure: str, cameras: List[str]):
        self.trial_name = trial_name
        self.trial_path = trial_path
        self.structure = structure  # 'multi_camera' or 'single_camera'
        self.cameras = cameras
        self._frame_cache = {}  # Cache frame numbers per camera

    def __repr__(self):
        cam_str = ', '.join(self.cameras) if self.cameras else 'None'
        return f"TrialInfo('{self.trial_name}', structure={self.structure}, cameras=[{cam_str}])"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trial_name': self.trial_name,
            'trial_path': self.trial_path,
            'structure': self.structure,
            'cameras': self.cameras
        }


class DataManager:
    """
    High-level data manager for multiple trials

    Automatically discovers trials, detects their structure, and provides
    a unified interface for loading data.
    """

    def __init__(self, root_folder: str, auto_discover: bool = True):
        """
        Initialize DataManager

        Args:
            root_folder: Root folder containing trials (e.g., 'sample_raw_data')
            auto_discover: If True, automatically discover trials on init
        """
        self.root_folder = os.path.abspath(root_folder)
        self.trials: Dict[str, TrialInfo] = {}

        if not os.path.isdir(self.root_folder):
            raise ValueError(f"Root folder does not exist: {self.root_folder}")

        if auto_discover:
            self.discover_trials()

    def discover_trials(self) -> List[str]:
        """
        Discover all trials in the root folder

        Returns:
            List of trial names
        """
        print(f"🔍 Discovering trials in: {self.root_folder}")

        # Look for subdirectories that could be trials
        for item in os.listdir(self.root_folder):
            item_path = os.path.join(self.root_folder, item)

            # Skip non-directories and hidden files
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue

            # Try to detect structure
            structure = detect_folder_structure(item_path)

            if structure == 'unknown':
                print(f"  ⚠️  Skipping {item}: Unknown structure")
                continue

            # Get cameras
            cameras = list_available_cameras(item_path)

            # Create TrialInfo
            trial_info = TrialInfo(
                trial_name=item,
                trial_path=item_path,
                structure=structure,
                cameras=cameras
            )

            self.trials[item] = trial_info

            if structure == 'multi_camera':
                print(f"  ✅ {item}: {structure} ({len(cameras)} cameras)")
            else:
                print(f"  ✅ {item}: {structure}")

        print(f"\n📊 Discovered {len(self.trials)} trials")
        return list(self.trials.keys())

    def list_trials(self) -> List[str]:
        """Get list of all trial names"""
        return list(self.trials.keys())

    def get_trial_info(self, trial_name: str) -> Optional[TrialInfo]:
        """Get TrialInfo for a specific trial"""
        return self.trials.get(trial_name)

    def get_trial_cameras(self, trial_name: str) -> List[str]:
        """Get list of cameras for a trial"""
        trial_info = self.trials.get(trial_name)
        if trial_info:
            return trial_info.cameras
        return []

    def load_frame(self, trial_name: str, camera_id: Optional[str], frame_number: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a frame from any trial

        Args:
            trial_name: Name of the trial
            camera_id: Camera ID (None for single-camera trials)
            frame_number: Frame number

        Returns:
            (color_img, depth_img) tuple

        Raises:
            ValueError: If trial not found
        """
        trial_info = self.trials.get(trial_name)

        if trial_info is None:
            raise ValueError(f"Trial '{trial_name}' not found. Available: {list(self.trials.keys())}")

        # Load using flexible loader
        return load_trial_data_flexible(trial_info.trial_path, camera_id, frame_number)

    def load_color(self, trial_name: str, camera_id: Optional[str], frame_number: int) -> np.ndarray:
        """Load only color image"""
        trial_info = self.trials.get(trial_name)
        if trial_info is None:
            raise ValueError(f"Trial '{trial_name}' not found")

        return load_color_flexible(trial_info.trial_path, camera_id, frame_number)

    def load_depth(self, trial_name: str, camera_id: Optional[str], frame_number: int) -> np.ndarray:
        """Load only depth data"""
        trial_info = self.trials.get(trial_name)
        if trial_info is None:
            raise ValueError(f"Trial '{trial_name}' not found")

        return load_depth_flexible(trial_info.trial_path, camera_id, frame_number)

    def find_available_frames(self, trial_name: str, camera_id: Optional[str] = None) -> List[int]:
        """
        Find all available frame numbers for a trial

        Args:
            trial_name: Name of the trial
            camera_id: Camera ID (None for single-camera or auto-select)

        Returns:
            Sorted list of frame numbers
        """
        trial_info = self.trials.get(trial_name)
        if trial_info is None:
            raise ValueError(f"Trial '{trial_name}' not found")

        # Use process_trial's frame finding logic
        from process_trial import find_all_frames
        return find_all_frames(trial_info.trial_path, camera_id)

    def print_summary(self):
        """Print a summary of all discovered trials"""
        print("\n" + "="*70)
        print("📊 DATA MANAGER SUMMARY")
        print("="*70)
        print(f"Root folder: {self.root_folder}")
        print(f"Total trials: {len(self.trials)}")
        print()

        for trial_name, trial_info in sorted(self.trials.items()):
            print(f"  📁 {trial_name}")
            print(f"     Structure: {trial_info.structure}")
            if trial_info.cameras:
                print(f"     Cameras: {', '.join(trial_info.cameras)}")
            print()

        print("="*70)

    def save_config(self, output_path: str = "data_config.json"):
        """
        Save trial configuration to JSON file

        Args:
            output_path: Path to save config file
        """
        config = {
            'root_folder': self.root_folder,
            'trials': {name: info.to_dict() for name, info in self.trials.items()}
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Saved configuration to: {output_path}")

    @classmethod
    def from_config(cls, config_path: str) -> 'DataManager':
        """
        Load DataManager from config file

        Args:
            config_path: Path to config JSON file

        Returns:
            DataManager instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        dm = cls(config['root_folder'], auto_discover=False)

        # Reconstruct trials
        for trial_name, trial_dict in config['trials'].items():
            trial_info = TrialInfo(
                trial_name=trial_dict['trial_name'],
                trial_path=trial_dict['trial_path'],
                structure=trial_dict['structure'],
                cameras=trial_dict['cameras']
            )
            dm.trials[trial_name] = trial_info

        print(f"✅ Loaded configuration from: {config_path}")
        return dm


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Insufficient arguments")
        print("\nUsage: python data_manager.py <root_folder>")
        print("\nExample:")
        print("  python data_manager.py sample_raw_data")
        sys.exit(1)

    root_folder = sys.argv[1]

    if not os.path.isdir(root_folder):
        print(f"❌ Error: Folder does not exist: {root_folder}")
        sys.exit(1)

    # Create DataManager
    dm = DataManager(root_folder)

    # Print summary
    dm.print_summary()

    # Save config
    config_path = os.path.join(root_folder, "data_config.json")
    dm.save_config(config_path)

    # Interactive demo
    print("\n" + "="*70)
    print("📺 INTERACTIVE DEMO")
    print("="*70)

    trials = dm.list_trials()
    if not trials:
        print("No trials found")
        return

    # Pick first trial
    trial_name = trials[0]
    trial_info = dm.get_trial_info(trial_name)

    print(f"\nTrying to load from: {trial_name}")

    # Get camera if multi-camera
    camera_id = None
    if trial_info.structure == 'multi_camera' and trial_info.cameras:
        camera_id = trial_info.cameras[0]
        print(f"Using camera: {camera_id}")

    # Find frames
    try:
        frames = dm.find_available_frames(trial_name, camera_id)
        if frames:
            frame_num = frames[len(frames) // 2]  # Pick middle frame
            print(f"Loading frame {frame_num} (middle of {len(frames)} frames)")

            # Load frame
            color, depth = dm.load_frame(trial_name, camera_id, frame_num)

            if color is not None:
                print(f"✅ Color loaded: {color.shape}")
            if depth is not None:
                print(f"✅ Depth loaded: {depth.shape}")

            # Visualize
            if color is not None or depth is not None:
                from load_trial_data_flexible import visualize_rgbd
                print("\n📺 Displaying frame (press any key to close)...")
                visualize_rgbd(color, depth, window_name=f"{trial_name} - Frame {frame_num}")
        else:
            print("No frames found")

    except Exception as e:
        print(f"❌ Error loading frame: {e}")


if __name__ == "__main__":
    main()
