"""
Trial Input Manager

Manages standardized trial_input/ folder.
All downstream tasks read from this location.

This ensures:
- Consistent naming: frame_XXXXXX.png / frame_XXXXXX.npy
- Consistent format: PNG for color, .npy for depth
- Consistent units: Depth always in meters
- No dependency on original folder structure
"""

import os
import sys
import numpy as np
import cv2
from typing import List, Optional, Tuple
import json

sys.path.insert(0, os.path.dirname(__file__))
from process_trial import process_trial


class TrialInputManager:
    """
    Manager for standardized trial_input/ folder

    Ensures all data is standardized before use by downstream tasks.
    """

    def __init__(self, trial_input_folder: str = "trial_input"):
        """
        Initialize TrialInputManager

        Args:
            trial_input_folder: Path to trial_input folder (default: "trial_input")
        """
        self.trial_input_folder = os.path.abspath(trial_input_folder)
        os.makedirs(self.trial_input_folder, exist_ok=True)

    def ensure_trial_processed(self, trial_path: str, camera_id: Optional[str] = None, force_reprocess: bool = False) -> str:
        """
        Ensure trial is processed and available in trial_input/

        If not already processed, processes it automatically.

        Args:
            trial_path: Path to original trial folder
            camera_id: Camera ID (None for single-camera)
            force_reprocess: If True, reprocess even if already exists

        Returns:
            output_path: Path to processed trial in trial_input/
        """
        # Determine output folder name (nested structure)
        trial_name = os.path.basename(os.path.normpath(trial_path))
        if camera_id:
            output_path = os.path.join(self.trial_input_folder, trial_name, camera_id)
        else:
            output_path = os.path.join(self.trial_input_folder, trial_name, "single_camera")

        # Check if already processed
        if os.path.exists(output_path) and not force_reprocess:
            # Verify it has color and depth folders
            color_folder = os.path.join(output_path, "color")
            depth_folder = os.path.join(output_path, "depth")

            if os.path.isdir(color_folder) and os.path.isdir(depth_folder):
                print(f"✅ Trial already processed: {output_path}")
                return output_path

        # Process trial
        print(f"⚙️ Processing trial to trial_input/...")
        output_path = process_trial(
            trial_path=trial_path,
            camera_id=camera_id,
            output_base=self.trial_input_folder,
            frame_range=None
        )

        return output_path

    def get_trial_path(self, trial_name: str, camera_id: Optional[str] = None) -> str:
        """
        Get path to processed trial in trial_input/

        Args:
            trial_name: Name of the trial
            camera_id: Camera ID (if multi-camera)

        Returns:
            Full path to trial folder in trial_input/
        """
        if camera_id:
            return os.path.join(self.trial_input_folder, trial_name, camera_id)
        else:
            return os.path.join(self.trial_input_folder, trial_name, "single_camera")

    def load_frame(self, trial_name: str, camera_id: Optional[str], frame_number: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load frame from trial_input/

        Args:
            trial_name: Name of the trial
            camera_id: Camera ID (if applicable)
            frame_number: Frame number

        Returns:
            (color_img, depth_img) tuple
        """
        trial_path = self.get_trial_path(trial_name, camera_id)

        # Load color
        color_path = os.path.join(trial_path, "color", f"frame_{frame_number:06d}.png")
        color_img = None
        if os.path.exists(color_path):
            color_img = cv2.imread(color_path)

        # Load depth
        depth_path = os.path.join(trial_path, "depth", f"frame_{frame_number:06d}.npy")
        depth_img = None
        if os.path.exists(depth_path):
            depth_img = np.load(depth_path)

        return color_img, depth_img

    def batch_load_frames(self, trial_name: str, camera_id: Optional[str], frame_numbers: List[int]) -> dict:
        """
        Load multiple frames efficiently (batch loading with threading)

        Args:
            trial_name: Name of the trial
            camera_id: Camera ID (if applicable)
            frame_numbers: List of frame numbers to load

        Returns:
            Dictionary mapping frame_number -> (color_img, depth_img)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        trial_path = self.get_trial_path(trial_name, camera_id)
        color_folder = os.path.join(trial_path, "color")
        depth_folder = os.path.join(trial_path, "depth")

        def load_single_frame(frame_num):
            """Load a single frame (color + depth)"""
            color_path = os.path.join(color_folder, f"frame_{frame_num:06d}.png")
            depth_path = os.path.join(depth_folder, f"frame_{frame_num:06d}.npy")

            color_img = None
            depth_img = None

            if os.path.exists(color_path):
                color_img = cv2.imread(color_path)

            if os.path.exists(depth_path):
                depth_img = np.load(depth_path)

            return frame_num, (color_img, depth_img)

        frames = {}

        # Load frames in parallel using thread pool (8 workers for disk I/O)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(load_single_frame, num): num for num in frame_numbers}

            for future in as_completed(futures):
                frame_num, data = future.result()
                frames[frame_num] = data

        return frames

    def find_available_frames(self, trial_name: str, camera_id: Optional[str] = None) -> List[int]:
        """
        Find all available frame numbers in trial_input/

        Args:
            trial_name: Name of the trial
            camera_id: Camera ID (if applicable)

        Returns:
            Sorted list of frame numbers
        """
        trial_path = self.get_trial_path(trial_name, camera_id)
        color_folder = os.path.join(trial_path, "color")

        if not os.path.isdir(color_folder):
            return []

        # Find all frame_XXXXXX.png files
        import glob
        import re

        frame_files = glob.glob(os.path.join(color_folder, "frame_*.png"))
        frame_numbers = []

        for f in frame_files:
            basename = os.path.basename(f)
            match = re.search(r'frame_(\d+)\.png', basename)
            if match:
                frame_numbers.append(int(match.group(1)))

        return sorted(frame_numbers)

    def get_metadata(self, trial_name: str, camera_id: Optional[str] = None) -> dict:
        """
        Get trial metadata from trial_input/

        Args:
            trial_name: Name of the trial
            camera_id: Camera ID (if applicable)

        Returns:
            Metadata dictionary
        """
        trial_path = self.get_trial_path(trial_name, camera_id)
        metadata_path = os.path.join(trial_path, "metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return {}

    def list_available_trials(self) -> List[str]:
        """
        List all trials available in trial_input/

        Returns:
            List of trial folder names
        """
        if not os.path.exists(self.trial_input_folder):
            return []

        trials = []
        for item in os.listdir(self.trial_input_folder):
            item_path = os.path.join(self.trial_input_folder, item)
            if os.path.isdir(item_path):
                # Check if it has color and depth folders
                color_folder = os.path.join(item_path, "color")
                depth_folder = os.path.join(item_path, "depth")
                if os.path.isdir(color_folder) and os.path.isdir(depth_folder):
                    trials.append(item)

        return sorted(trials)

    def print_summary(self):
        """Print summary of trial_input/"""
        trials = self.list_available_trials()

        print("\n" + "="*70)
        print("📂 TRIAL_INPUT SUMMARY")
        print("="*70)
        print(f"Location: {self.trial_input_folder}")
        print(f"Total trials: {len(trials)}")
        print()

        for trial in trials:
            frames = self.find_available_frames(trial, None)
            metadata = self.get_metadata(trial, None)

            print(f"  📁 {trial}")
            print(f"     Frames: {len(frames)}")
            if frames:
                print(f"     Range: {min(frames)}-{max(frames)}")
            if metadata:
                print(f"     Structure: {metadata.get('structure', 'unknown')}")
            print()

        print("="*70)


def main():
    """Command line interface"""
    manager = TrialInputManager()
    manager.print_summary()


if __name__ == "__main__":
    main()
