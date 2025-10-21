"""
DeepLabCut-based dog skeleton detection using SuperAnimal Quadruped model
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json


class DeepLabCutDogDetector:
    """
    Dog skeleton detector using DeepLabCut SuperAnimal Quadruped model.

    Uses DeepLabCut's built-in Faster R-CNN detector to find dogs automatically.
    """

    def __init__(self):
        """Initialize DeepLabCut dog detector."""
        try:
            import deeplabcut
            self.dlc = deeplabcut
            print("✅ DeepLabCut loaded successfully")
        except ImportError:
            raise ImportError(
                "DeepLabCut not installed. Install with:\n"
                "pip install deeplabcut"
            )

        # SuperAnimal Quadruped keypoint names (39 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'throat', 'withers', 'tail_base', 'left_front_paw', 'right_front_paw',
            'left_back_paw', 'right_back_paw', 'left_front_knee', 'right_front_knee',
            'left_back_knee', 'right_back_knee', 'left_front_elbow', 'right_front_elbow',
            'left_back_elbow', 'right_back_elbow', 'neck', 'tail_mid', 'tail_tip',
            'left_front_wrist', 'right_front_wrist', 'left_back_ankle', 'right_back_ankle',
            'left_front_pad', 'right_front_pad', 'left_back_pad', 'right_back_pad',
            'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
            'left_front_toe', 'right_front_toe', 'left_back_toe', 'right_back_toe'
        ]

    def detect_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        Run DeepLabCut detection on a video.

        Args:
            video_path: Path to input video
            output_dir: Optional output directory (defaults to video directory)

        Returns:
            Dictionary with detection results
        """
        video_path = str(Path(video_path).expanduser())

        if output_dir is None:
            output_dir = str(Path(video_path).parent)

        print(f"\n{'='*60}")
        print(f"🐶 DEEPLABCUT DOG DETECTION")
        print(f"{'='*60}")
        print(f"Input video: {video_path}")
        print(f"Output dir: {output_dir}")

        # Run SuperAnimal Quadruped detection with built-in detector
        print("\n🔄 Running SuperAnimal Quadruped detection...")
        try:
            # API: video_inference_superanimal(videos, superanimal_name, **kwargs)
            output = self.dlc.video_inference_superanimal(
                [video_path],
                'superanimal_quadruped',  # superanimal_name (not model_type!)
                model_name="hrnet_w32",
                detector_name="fasterrcnn_resnet50_fpn_v2",  # Built-in detector!
                plot_trajectories=True,
                video_adapt=False,
                pcutoff=0.3  # Confidence threshold
            )

            print(f"✅ Detection complete!")
            print(f"   Output: {output}")

            # Parse JSON output
            json_path = self._find_output_json(output_dir)
            if json_path:
                print(f"📄 Loading results from: {json_path}")
                results = self._parse_deeplabcut_json(json_path)
                print(f"✅ Parsed {len(results)} frames")
                return results
            else:
                print("⚠️  Warning: Could not find output JSON")
                return {}

        except Exception as e:
            print(f"❌ DeepLabCut detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _find_output_json(self, output_dir: str) -> Optional[Path]:
        """Find the DeepLabCut output JSON file."""
        output_path = Path(output_dir)

        # Look for JSON files with DeepLabCut naming pattern
        json_files = list(output_path.glob("*.json"))

        # Filter for DLC output files (contain 'DLC' or 'skeleton')
        dlc_jsons = [f for f in json_files if 'DLC' in f.name or 'skeleton' in f.name]

        if dlc_jsons:
            # Return most recent
            return max(dlc_jsons, key=lambda p: p.stat().st_mtime)

        return None

    def _parse_deeplabcut_json(self, json_path: Path) -> Dict:
        """
        Parse DeepLabCut JSON output to standard format.

        Args:
            json_path: Path to DeepLabCut JSON output

        Returns:
            Dictionary mapping frame indices to detection results
        """
        with open(json_path, 'r') as f:
            dlc_data = json.load(f)

        results = {}

        # DeepLabCut JSON structure varies, handle different formats
        if isinstance(dlc_data, list):
            # Frame-by-frame list format
            for frame_idx, frame_data in enumerate(dlc_data):
                results[frame_idx] = self._parse_frame_data(frame_data)

        elif isinstance(dlc_data, dict):
            # Dictionary format with frame keys
            for frame_key, frame_data in dlc_data.items():
                if frame_key.startswith('frame_'):
                    frame_idx = int(frame_key.split('_')[1])
                    results[frame_idx] = self._parse_frame_data(frame_data)

        return results

    def _parse_frame_data(self, frame_data: Dict) -> Dict:
        """
        Parse single frame data from DeepLabCut output.

        Args:
            frame_data: DeepLabCut frame data

        Returns:
            Standardized frame result with bodyparts, bboxes, bbox_scores
        """
        result = {
            'bodyparts': [],
            'bboxes': [],
            'bbox_scores': []
        }

        # Extract keypoints from DeepLabCut format
        bodyparts = frame_data.get('bodyparts', [])
        if bodyparts:
            result['bodyparts'] = bodyparts

        # Extract bounding boxes
        bboxes = frame_data.get('bboxes', [])
        if bboxes:
            result['bboxes'] = bboxes

        # Extract bbox scores
        scores = frame_data.get('bbox_scores', [])
        if scores:
            result['bbox_scores'] = scores
        elif bboxes:
            # Default score of 1.0 if not provided
            result['bbox_scores'] = [1.0] * len(bboxes)

        return result


if __name__ == "__main__":
    # Test the detector
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()

    detector = DeepLabCutDogDetector()
    results = detector.detect_video(args.video)

    print(f"\n✅ Detection complete!")
    print(f"   Frames processed: {len(results)}")
