#!/usr/bin/env python3
"""
Pipeline Validation and Testing Tool

This script validates the coordinate transformations and scale corrections
throughout the computer vision pipeline.
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dog_script.segmentation import SAM2VideoSegmenter
import yaml


def validate_scale_factors(trial_dir):
    """Validate scale factor accuracy between resolutions."""
    print(f"\n🔍 Validating scale factors for {trial_dir}")
    
    color_dir = os.path.join(trial_dir, "Color")
    scale_metadata_path = os.path.join(trial_dir, "sam2_scale_metadata.json")
    
    if not os.path.exists(scale_metadata_path):
        print("❌ No scale metadata found")
        return False
        
    with open(scale_metadata_path, 'r') as f:
        scale_info = json.load(f)
    
    # Check a few sample images
    png_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])
    sample_files = png_files[::len(png_files)//3][:3]  # Sample 3 files
    
    for png_file in sample_files:
        # Load original image
        orig_img = cv2.imread(os.path.join(color_dir, png_file))
        orig_h, orig_w = orig_img.shape[:2]
        
        # Get expected dimensions
        jpg_name = png_file.replace('.png', '.jpg')
        if jpg_name in scale_info['scale_factors']:
            sx, sy = scale_info['scale_factors'][jpg_name]
            expected_w = int(orig_w / sx)
            expected_h = int(orig_h / sy)
            
            print(f"  📏 {png_file}: {orig_w}x{orig_h} -> {expected_w}x{expected_h} (scale: {sx:.3f}, {sy:.3f})")
            
            # Validate scale accuracy
            if abs(sx - sy) > 0.01:
                print(f"    ⚠️ Non-uniform scaling detected")
        else:
            print(f"  ❌ No scale info for {jpg_name}")
    
    return True


def validate_depth_alignment(trial_dir):
    """Validate that 2D keypoints align with depth data."""
    print(f"\n🎯 Validating depth alignment for {trial_dir}")
    
    # Load pose processing results
    results_file = os.path.join(trial_dir, "processed_subject_result.json")
    if not os.path.exists(results_file):
        print("❌ No pose processing results found")
        return False
        
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load a sample depth frame
    depth_dir = os.path.join(trial_dir, "Depth")
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.raw')])
    
    if not depth_files:
        print("❌ No depth files found")
        return False
        
    # Test first few frames
    valid_alignments = 0
    total_tests = 0
    
    for i, result in enumerate(results[:10]):  # Test first 10 frames
        if not result.get('bodyparts') or i >= len(depth_files):
            continue
            
        # Load depth frame
        depth_file = os.path.join(depth_dir, depth_files[i])
        with open(depth_file, 'rb') as f:
            raw_depth = np.frombuffer(f.read(), dtype=np.uint16)
            # Assume standard dimensions - should get from metadata
            depth_frame = raw_depth.reshape((720, 1280)) / 1000.0
            
        # Check keypoint alignment
        keypoints = result['bodyparts'][0]
        for kp in keypoints[:5]:  # Test first 5 keypoints
            if len(kp) >= 2:
                x, y = int(kp[0]), int(kp[1])
                total_tests += 1
                
                if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                    depth_val = depth_frame[y, x]
                    if depth_val > 0:
                        valid_alignments += 1
    
    alignment_ratio = valid_alignments / total_tests if total_tests > 0 else 0
    print(f"  📊 Alignment: {valid_alignments}/{total_tests} ({alignment_ratio:.1%})")
    
    if alignment_ratio < 0.3:
        print("  ❌ Poor depth alignment detected")
        return False
    else:
        print("  ✅ Good depth alignment")
        return True


def validate_target_coordinates(trial_dir):
    """Validate manually labeled target coordinates."""
    print(f"\n🎯 Validating target coordinates for {trial_dir}")
    
    target_file = os.path.join(trial_dir, "target_coordinates.json")
    if not os.path.exists(target_file):
        print("  ℹ️ No manual target coordinates found")
        return True
        
    with open(target_file, 'r') as f:
        target_data = json.load(f)
    
    targets = target_data.get('targets', [])
    print(f"  📍 Found {len(targets)} targets")
    
    # Validate target positions
    for target in targets:
        label = target['label']
        pixel_coords = target['pixel_coords']
        world_coords = target['world_coords']
        
        print(f"    {label}: pixel({pixel_coords[0]:.0f}, {pixel_coords[1]:.0f}) -> world({world_coords[0]:.3f}, {world_coords[1]:.3f}, {world_coords[2]:.3f})")
        
        # Basic sanity checks
        if world_coords[2] <= 0:  # Negative or zero depth
            print(f"      ❌ Invalid depth: {world_coords[2]}")
            return False
            
        if abs(world_coords[0]) > 5 or abs(world_coords[1]) > 5:  # Unreasonable X/Y
            print(f"      ⚠️ Large X/Y coordinates: {world_coords[0]}, {world_coords[1]}")
    
    # Check target spacing
    if len(targets) >= 2:
        positions = np.array([t['world_coords'] for t in targets])
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        min_dist = min(distances)
        max_dist = max(distances)
        print(f"  📏 Target spacing: {min_dist:.3f}m - {max_dist:.3f}m")
        
        if min_dist < 0.1:
            print("  ⚠️ Some targets very close together")
    
    print("  ✅ Target coordinates validated")
    return True


def validate_distance_calculations(trial_dir):
    """Validate distance calculation outputs."""
    print(f"\n📏 Validating distance calculations for {trial_dir}")
    
    csv_file = os.path.join(trial_dir, "processed_subject_result_table.csv")
    if not os.path.exists(csv_file):
        print("❌ No CSV results found")
        return False
        
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    # Check for target distance columns
    distance_cols = [col for col in df.columns if col.endswith('_r')]
    print(f"  📊 Found {len(distance_cols)} distance columns: {distance_cols}")
    
    if not distance_cols:
        print("  ❌ No distance columns found")
        return False
    
    # Check for valid distances
    for col in distance_cols:
        valid_count = df[col].notna().sum()
        total_count = len(df)
        mean_dist = df[col].mean()
        
        print(f"    {col}: {valid_count}/{total_count} valid, mean={mean_dist:.3f}m")
        
        if valid_count == 0:
            print(f"      ❌ No valid distances for {col}")
            return False
    
    # Check for target labeling method
    if 'target_labeling_method' in df.columns:
        methods = df['target_labeling_method'].value_counts()
        print(f"  🏷️ Target labeling methods: {dict(methods)}")
    
    # Check for closest target info
    if 'closest_target_distance' in df.columns:
        closest_mean = df['closest_target_distance'].mean()
        print(f"  🎯 Mean closest target distance: {closest_mean:.3f}m")
    
    print("  ✅ Distance calculations validated")
    return True


def test_coordinate_transformation(trial_dir):
    """Test coordinate transformation accuracy."""
    print(f"\n🔄 Testing coordinate transformations for {trial_dir}")
    
    # Load camera intrinsics
    metadata_files = [
        os.path.join(trial_dir, "rosbag_metadata.yaml"),
        os.path.join(os.path.dirname(trial_dir), "rosbag_metadata.yaml"),
        os.path.join(os.path.dirname(__file__), "config", "camera_config.yaml")
    ]
    
    intrinsics = None
    for metadata_file in metadata_files:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                data = yaml.safe_load(f)
            intrinsics = data['intrinsics']
            print(f"  📷 Loaded intrinsics from {metadata_file}")
            break
    
    if not intrinsics:
        print("  ❌ No camera intrinsics found")
        return False
    
    # Test forward and backward transformation
    test_points = [
        [640, 360, 2.0],  # Center point at 2m depth
        [100, 100, 1.5],  # Top-left
        [1180, 620, 3.0]  # Bottom-right
    ]
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['ppx'], intrinsics['ppy']
    
    print("  🧪 Testing pixel ↔ 3D transformations:")
    for u, v, depth in test_points:
        # Pixel to 3D
        x_3d = (u - cx) * depth / fx
        y_3d = (v - cy) * depth / fy
        z_3d = depth
        
        # 3D back to pixel
        u_back = (x_3d * fx / z_3d) + cx
        v_back = (y_3d * fy / z_3d) + cy
        
        error_u = abs(u - u_back)
        error_v = abs(v - v_back)
        
        print(f"    ({u}, {v}, {depth}) -> ({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}) -> ({u_back:.1f}, {v_back:.1f})")
        print(f"      Error: {error_u:.3f}, {error_v:.3f} pixels")
        
        if error_u > 0.1 or error_v > 0.1:
            print("      ❌ High transformation error")
            return False
    
    print("  ✅ Coordinate transformations accurate")
    return True


def validate_pipeline_outputs(trial_dir):
    """Validate expected pipeline outputs exist."""
    print(f"\n📁 Validating pipeline outputs for {trial_dir}")
    
    expected_files = [
        "interactive_points.json",
        "sam2_scale_metadata.json", 
        "masked_video.mp4",
        "processed_subject_result.json",
        "processed_subject_result_table.csv",
        "processed_subject_result_trace3d.png",
        "processed_subject_result_distance_comparison.png"
    ]
    
    missing_files = []
    for expected_file in expected_files:
        if not os.path.exists(os.path.join(trial_dir, expected_file)):
            missing_files.append(expected_file)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print("  ✅ All expected files present")
        return True


def run_comprehensive_validation(trial_dir):
    """Run all validation tests on a trial."""
    print(f"\n{'='*60}")
    print(f"🧪 COMPREHENSIVE VALIDATION: {os.path.basename(trial_dir)}")
    print(f"{'='*60}")
    
    tests = [
        ("Scale Factors", validate_scale_factors),
        ("Depth Alignment", validate_depth_alignment),
        ("Target Coordinates", validate_target_coordinates),
        ("Distance Calculations", validate_distance_calculations),
        ("Coordinate Transformations", test_coordinate_transformation),
        ("Pipeline Outputs", validate_pipeline_outputs)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func(trial_dir)
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status} {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ FAIL {test_name}: {str(e)}")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 All validations passed!")
    else:
        print("⚠️ Some validations failed. Check outputs above.")
        
    return passed == total


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Validation Tool")
    parser.add_argument("--trial_dir", type=str, help="Trial directory to validate")
    parser.add_argument("--base_dir", type=str, help="Base directory with multiple trials")
    
    args = parser.parse_args()
    
    if args.trial_dir:
        trial_dir = os.path.expanduser(args.trial_dir)
        if not os.path.exists(trial_dir):
            print(f"❌ Trial directory not found: {trial_dir}")
            return
        run_comprehensive_validation(trial_dir)
        
    elif args.base_dir:
        base_dir = os.path.expanduser(args.base_dir)
        if not os.path.exists(base_dir):
            print(f"❌ Base directory not found: {base_dir}")
            return
            
        # Find all trial directories
        trial_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                color_dir = os.path.join(item_path, "Color")
                if os.path.isdir(color_dir):
                    trial_dirs.append(item_path)
                else:
                    # Check subdirectories
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            color_dir = os.path.join(subitem_path, "Color")
                            if os.path.isdir(color_dir):
                                trial_dirs.append(subitem_path)
        
        print(f"Found {len(trial_dirs)} trial directories to validate")
        
        overall_results = []
        for trial_dir in trial_dirs:
            result = run_comprehensive_validation(trial_dir)
            overall_results.append((trial_dir, result))
        
        # Overall summary
        passed_trials = sum(1 for _, result in overall_results if result)
        total_trials = len(overall_results)
        
        print(f"\n🏆 OVERALL SUMMARY: {passed_trials}/{total_trials} trials passed validation")
        
        # List failed trials
        failed_trials = [trial for trial, result in overall_results if not result]
        if failed_trials:
            print("\n❌ Failed trials:")
            for trial in failed_trials:
                print(f"  - {trial}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()