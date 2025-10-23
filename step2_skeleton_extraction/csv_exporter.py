"""
CSV export for pointing gesture analysis.

Exports processed results to CSV with all required columns.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List


def export_pointing_analysis_to_csv(results_dict: Dict, analyses_dict: Dict,
                                    output_path: Path, global_start_frame: int = 0):
    """
    Export pointing analysis results to CSV.

    Args:
        results_dict: Dictionary of {frame_key: DetectionResult}
        analyses_dict: Dictionary of {frame_key: analysis_dict from analyze_pointing_frame}
        output_path: Path to save CSV file
        global_start_frame: Starting frame number for global_frame column.
                           This is used when multiple trials are concatenated - the first
                           frame of trial 2 won't start at 1, but will continue from where
                           trial 1 ended. Frame is local to this trial, global_frame is
                           across all trials.

    CSV Columns:
        frame: Frame number within this trial
        global_frame: Frame number across all trials (frame + global_start_frame)
        pointing_arm: Which arm is pointing (left/right)
        wrist_location: 3D position of wrist
        *_ground_intersection: Where each vector intersects ground plane
        *_vec: Direction vector for each pointing vector
        *_dist_to_target_1-4: Euclidean distance from intersection to each target
        head_orientation_*: Head facing direction metrics
        landmarks: 3D skeleton landmarks
        confidence: Average MediaPipe detection confidence (mean of all landmark visibilities)
    """
    rows = []

    # Sort frame keys to ensure proper ordering
    sorted_keys = sorted(results_dict.keys())

    for frame_key in sorted_keys:
        result = results_dict[frame_key]
        analysis = analyses_dict.get(frame_key)

        if not analysis:
            continue

        # Extract frame number from frame_key (e.g., "frame_000139" -> 139)
        frame_num = int(frame_key.split('_')[-1])

        # Build row
        row = {
            'frame': frame_num,
            'global_frame': frame_num + global_start_frame,
            'pointing_arm': result.metadata.get('pointing_arm', 'unknown'),
            'wrist_location': analysis.get('wrist_location'),
            'eye_to_wrist_ground_intersection': analysis.get('eye_to_wrist_ground_intersection'),
            'shoulder_to_wrist_ground_intersection': analysis.get('shoulder_to_wrist_ground_intersection'),
            'elbow_to_wrist_ground_intersection': analysis.get('elbow_to_wrist_ground_intersection'),
            'nose_to_wrist_ground_intersection': analysis.get('nose_to_wrist_ground_intersection'),
            'ground_intersection': analysis.get('ground_intersection'),
            'head_orientation_ground_intersection': analysis.get('head_orientation_ground_intersection'),
        }

        # Add vector and distance columns for each arm vector
        for vec_name in ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist']:
            row[f'{vec_name}_vec'] = analysis.get(f'{vec_name}_vec')

            for i in range(1, 5):
                row[f'{vec_name}_dist_to_target_{i}'] = analysis.get(f'{vec_name}_dist_to_target_{i}')

        # Add head orientation distance columns
        for i in range(1, 5):
            row[f'head_orientation_dist_to_target_{i}'] = analysis.get(f'head_orientation_dist_to_target_{i}')

        # Add head orientation vector and origin
        row['head_orientation_vector'] = analysis.get('head_orientation_vector')
        row['head_orientation_origin'] = analysis.get('head_orientation_origin')

        # Add landmarks
        row['landmarks'] = result.landmarks_3d

        # Compute average confidence (visibility) from 2D landmarks
        # landmarks_2d is [(x, y, visibility), ...]
        if result.landmarks_2d:
            import numpy as np
            visibilities = [v for x, y, v in result.landmarks_2d]
            avg_confidence = np.mean(visibilities) if visibilities else None
            row['confidence'] = avg_confidence
        else:
            row['confidence'] = None

        rows.append(row)

    # Create DataFrame with explicit column order
    columns = [
        'frame', 'global_frame', 'pointing_arm', 'wrist_location',
        'eye_to_wrist_ground_intersection',
        'shoulder_to_wrist_ground_intersection',
        'elbow_to_wrist_ground_intersection',
        'nose_to_wrist_ground_intersection',
        'ground_intersection',
        'head_orientation_ground_intersection',
        'eye_to_wrist_vec', 'eye_to_wrist_dist_to_target_1', 'eye_to_wrist_dist_to_target_2',
        'eye_to_wrist_dist_to_target_3', 'eye_to_wrist_dist_to_target_4',
        'shoulder_to_wrist_vec', 'shoulder_to_wrist_dist_to_target_1', 'shoulder_to_wrist_dist_to_target_2',
        'shoulder_to_wrist_dist_to_target_3', 'shoulder_to_wrist_dist_to_target_4',
        'elbow_to_wrist_vec', 'elbow_to_wrist_dist_to_target_1', 'elbow_to_wrist_dist_to_target_2',
        'elbow_to_wrist_dist_to_target_3', 'elbow_to_wrist_dist_to_target_4',
        'nose_to_wrist_vec', 'nose_to_wrist_dist_to_target_1', 'nose_to_wrist_dist_to_target_2',
        'nose_to_wrist_dist_to_target_3', 'nose_to_wrist_dist_to_target_4',
        'head_orientation_dist_to_target_1', 'head_orientation_dist_to_target_2',
        'head_orientation_dist_to_target_3', 'head_orientation_dist_to_target_4',
        'head_orientation_vector', 'head_orientation_origin',
        'landmarks', 'confidence'
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Exported pointing analysis to: {output_path}")
    print(f"   Frames: {len(df)}")

    return df
