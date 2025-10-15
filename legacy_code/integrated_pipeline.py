#!/usr/bin/env python3
"""
Integrated Computer Vision Pipeline with Scale Correction and Target Labeling

This script integrates all pipeline components with proper coordinate handling:
1. SAM2 segmentation with scale metadata
2. Target labeling (manual or static)
3. Pose detection with coordinate correction
4. Distance calculation with enhanced target support
5. Validation and quality checks
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add local modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dog_script'))

from dog_script.segmentation import SAM2VideoSegmenter
from dog_script.dog_pose_visualize import pose_visualize
from label_targets import label_single_trial
from validate_pipeline import run_comprehensive_validation


class IntegratedPipeline:
    """Main pipeline coordinator with scale correction and target labeling."""
    
    def __init__(self, base_dir, enable_validation=True):
        self.base_dir = Path(base_dir).expanduser()
        self.enable_validation = enable_validation
        self.processed_trials = []
        self.failed_trials = []
        
    def find_trial_directories(self):
        """Find all trial directories in the base directory."""
        trial_dirs = []
        
        for item in self.base_dir.iterdir():
            if not item.is_dir():
                continue
                
            # Check if this is directly a trial directory
            color_dir = item / "Color"
            depth_dir = item / "Depth"
            
            if color_dir.exists() and depth_dir.exists():
                trial_dirs.append(item)
            else:
                # Check subdirectories for trials
                for subitem in item.iterdir():
                    if not subitem.is_dir():
                        continue
                    sub_color_dir = subitem / "Color"
                    sub_depth_dir = subitem / "Depth"
                    
                    if sub_color_dir.exists() and sub_depth_dir.exists():
                        trial_dirs.append(subitem)
        
        return sorted(trial_dirs)
    
    def check_trial_completeness(self, trial_dir):
        """Check if trial has all required input data."""
        color_dir = trial_dir / "Color"
        depth_dir = trial_dir / "Depth"
        
        # Check for color frames
        color_files = list(color_dir.glob("*.png")) + list(color_dir.glob("*.jpg"))
        depth_files = list(depth_dir.glob("*.raw"))
        
        if not color_files:
            print(f"❌ No color frames found in {color_dir}")
            return False
            
        if not depth_files:
            print(f"❌ No depth frames found in {depth_dir}")
            return False
            
        if len(color_files) != len(depth_files):
            print(f"⚠️ Frame count mismatch: {len(color_files)} color, {len(depth_files)} depth")
            # Allow processing but warn
            
        return True
    
    def run_segmentation_step(self, trial_dir, force_reprocess=False):
        """Run SAM2 segmentation with scale metadata."""
        print(f"\n🔧 Step 1: SAM2 Segmentation - {trial_dir.name}")
        
        # Check if already processed
        scale_metadata_path = trial_dir / "sam2_scale_metadata.json"
        masked_video_path = trial_dir / "masked_video.mp4"
        
        if not force_reprocess and scale_metadata_path.exists() and masked_video_path.exists():
            print("✅ Segmentation already completed, skipping")
            return True
            
        try:
            color_dir = trial_dir / "Color"
            segmenter = SAM2VideoSegmenter(str(color_dir))
            segmenter.load_video_frames()
            segmenter.interactive_segmentation()
            segmenter.propagate_segmentation()
            segmenter.visualize_results()
            
            print("✅ Segmentation completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Segmentation failed: {str(e)}")
            return False
    
    def run_target_labeling_step(self, trial_dir, force_relabel=False):
        """Run target labeling (manual or static fallback)."""
        print(f"\n🎯 Step 2: Target Labeling - {trial_dir.name}")
        
        target_file = trial_dir / "target_coordinates.json"
        
        if not force_relabel and target_file.exists():
            with open(target_file, 'r') as f:
                data = json.load(f)
            target_count = len(data.get('targets', []))
            print(f"✅ Manual targets already exist ({target_count} targets), skipping")
            return True
        
        try:
            result = label_single_trial(str(trial_dir), force_relabel)
            if result:
                print("✅ Target labeling completed successfully")
                return True
            else:
                print("⚠️ Manual target labeling skipped, will use static targets")
                return True  # Still allow processing with static targets
                
        except Exception as e:
            print(f"❌ Target labeling failed: {str(e)}")
            print("⚠️ Will attempt to use static targets")
            return True  # Allow processing to continue
    
    def run_pose_processing_step(self, trial_dir, side_view=False, dog=True, force_reprocess=False):
        """Run pose detection and 3D reconstruction with coordinate correction."""
        print(f"\n🤖 Step 3: Pose Processing - {trial_dir.name}")
        
        # Check if already processed
        results_csv = trial_dir / "processed_subject_result_table.csv"
        
        if not force_reprocess and results_csv.exists():
            print("✅ Pose processing already completed, skipping")
            return True
        
        try:
            # Look for skeleton detection results
            skeleton_json = trial_dir / "masked_video_skeleton.json"
            if not skeleton_json.exists():
                print(f"❌ No skeleton detection results found: {skeleton_json}")
                print("   Please run skeleton detection first")
                return False
            
            # Run pose visualization with coordinate correction
            pose_visualize(str(skeleton_json), side_view=side_view, dog=dog)
            
            print("✅ Pose processing completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Pose processing failed: {str(e)}")
            return False
    
    def run_validation_step(self, trial_dir):
        """Run pipeline validation."""
        if not self.enable_validation:
            return True
            
        print(f"\n🧪 Step 4: Validation - {trial_dir.name}")
        
        try:
            result = run_comprehensive_validation(str(trial_dir))
            if result:
                print("✅ Validation passed")
            else:
                print("⚠️ Some validation checks failed")
            return result
            
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            return False
    
    def process_single_trial(self, trial_dir, force_reprocess=False, force_relabel=False, side_view=False, dog=True):
        """Process a single trial through the entire pipeline."""
        print(f"\n{'='*60}")
        print(f"🚀 PROCESSING TRIAL: {trial_dir.name}")
        print(f"{'='*60}")
        
        # Check trial completeness
        if not self.check_trial_completeness(trial_dir):
            print(f"❌ Trial {trial_dir.name} is incomplete, skipping")
            self.failed_trials.append(str(trial_dir))
            return False
        
        # Step 1: Segmentation
        if not self.run_segmentation_step(trial_dir, force_reprocess):
            self.failed_trials.append(str(trial_dir))
            return False
        
        # Step 2: Target Labeling
        if not self.run_target_labeling_step(trial_dir, force_relabel):
            self.failed_trials.append(str(trial_dir))
            return False
        
        # Step 3: Pose Processing
        if not self.run_pose_processing_step(trial_dir, side_view, dog, force_reprocess):
            self.failed_trials.append(str(trial_dir))
            return False
        
        # Step 4: Validation
        validation_passed = self.run_validation_step(trial_dir)
        
        if validation_passed:
            self.processed_trials.append(str(trial_dir))
            print(f"🎉 Trial {trial_dir.name} completed successfully!")
        else:
            self.failed_trials.append(str(trial_dir))
            print(f"⚠️ Trial {trial_dir.name} completed with validation warnings")
        
        return True
    
    def process_multiple_trials(self, force_reprocess=False, force_relabel=False, side_view=False, dog=True):
        """Process multiple trials."""
        trial_dirs = self.find_trial_directories()
        
        if not trial_dirs:
            print(f"❌ No trial directories found in {self.base_dir}")
            return
        
        print(f"📂 Found {len(trial_dirs)} trial directories")
        
        for trial_dir in trial_dirs:
            self.process_single_trial(trial_dir, force_reprocess, force_relabel, side_view, dog)
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print processing summary."""
        total = len(self.processed_trials) + len(self.failed_trials)
        
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {len(self.processed_trials)}/{total}")
        print(f"❌ Failed trials: {len(self.failed_trials)}/{total}")
        
        if self.failed_trials:
            print(f"\n❌ Failed trials:")
            for trial in self.failed_trials:
                print(f"  - {trial}")
        
        if self.processed_trials:
            print(f"\n✅ Successfully processed trials:")
            for trial in self.processed_trials:
                print(f"  - {trial}")


def main():
    parser = argparse.ArgumentParser(description="Integrated Computer Vision Pipeline")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing trials")
    parser.add_argument("--trial_dir", type=str, help="Process single trial directory")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing even if outputs exist")
    parser.add_argument("--force_relabel", action="store_true", help="Force re-labeling targets")
    parser.add_argument("--side_view", action="store_true", help="Process as side view camera")
    parser.add_argument("--human", action="store_true", help="Process human subjects instead of dogs")
    parser.add_argument("--no_validation", action="store_true", help="Skip validation steps")
    parser.add_argument("--steps", type=str, nargs='+', choices=['segmentation', 'targets', 'pose', 'validation'], 
                       help="Run only specific pipeline steps")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = IntegratedPipeline(args.base_dir, enable_validation=not args.no_validation)
    
    # Determine subject type
    dog = not args.human
    
    if args.trial_dir:
        # Process single trial
        trial_dir = Path(args.trial_dir).expanduser()
        if not trial_dir.exists():
            print(f"❌ Trial directory not found: {trial_dir}")
            return
        
        if args.steps:
            # Run specific steps only
            if 'segmentation' in args.steps:
                pipeline.run_segmentation_step(trial_dir, args.force_reprocess)
            if 'targets' in args.steps:
                pipeline.run_target_labeling_step(trial_dir, args.force_relabel)
            if 'pose' in args.steps:
                pipeline.run_pose_processing_step(trial_dir, args.side_view, dog, args.force_reprocess)
            if 'validation' in args.steps:
                pipeline.run_validation_step(trial_dir)
        else:
            # Run full pipeline
            pipeline.process_single_trial(trial_dir, args.force_reprocess, args.force_relabel, args.side_view, dog)
    else:
        # Process multiple trials
        pipeline.process_multiple_trials(args.force_reprocess, args.force_relabel, args.side_view, dog)


if __name__ == "__main__":
    main()