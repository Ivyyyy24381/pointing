#!/usr/bin/env python3
"""
Interactive Target Labeling Tool

This tool allows manual labeling of target positions in trials for accurate distance calculations.
It handles coordinate scaling between display and original resolutions.
"""

import os
import json
import argparse
import numpy as np
import cv2
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from types import SimpleNamespace


class TargetLabeler:
    """Interactive tool for labeling target positions in trials."""
    
    def __init__(self, trial_dir):
        self.trial_dir = trial_dir
        self.color_dir = os.path.join(trial_dir, "Color")
        self.depth_dir = os.path.join(trial_dir, "Depth")
        
        # Load camera intrinsics
        self.intrinsics = self.load_camera_intrinsics()
        
        # Target storage
        self.targets = []
        self.current_target_id = 1
        
        # Display settings
        self.display_size = 800  # Max display size for labeling
        self.scale_factor = 1.0
        
        # Representative frame selection
        self.representative_frame = None
        self.depth_frame = None
        
    def load_camera_intrinsics(self):
        """Load camera intrinsics from various possible sources."""
        # Try rosbag metadata first
        metadata_path = os.path.join(self.trial_dir, "rosbag_metadata.yaml")
        if os.path.exists(metadata_path):
            return self.load_intrinsics_from_yaml(metadata_path)
        
        # Try parent directory
        parent_dir = os.path.dirname(self.trial_dir)
        metadata_path = os.path.join(parent_dir, "rosbag_metadata.yaml")
        if os.path.exists(metadata_path):
            return self.load_intrinsics_from_yaml(metadata_path)
            
        # Fall back to config
        config_path = os.path.join(os.path.dirname(__file__), "config", "camera_config.yaml")
        if os.path.exists(config_path):
            return self.load_intrinsics_from_yaml(config_path)
            
        # Default intrinsics if nothing found
        print("⚠️ No camera intrinsics found, using defaults")
        return SimpleNamespace(
            width=640, height=480,
            fx=455.0, fy=455.0,
            ppx=320.0, ppy=240.0,
            coeffs=[0.0] * 5
        )
    
    def load_intrinsics_from_yaml(self, yaml_path):
        """Load intrinsics from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        intr_data = data['intrinsics']
        return SimpleNamespace(
            width=intr_data['width'],
            height=intr_data['height'],
            ppx=intr_data['ppx'],
            ppy=intr_data['ppy'],
            fx=intr_data['fx'],
            fy=intr_data['fy'],
            coeffs=intr_data['coeffs']
        )
    
    def select_representative_frame(self):
        """Choose a clear frame for target labeling."""
        color_files = sorted([f for f in os.listdir(self.color_dir) if f.endswith('.png')])
        depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.raw')])
        
        if not color_files or not depth_files:
            raise ValueError(f"No color or depth files found in {self.trial_dir}")
        
        # Select middle frame for stability
        mid_idx = len(color_files) // 2
        color_frame_path = os.path.join(self.color_dir, color_files[mid_idx])
        depth_frame_path = os.path.join(self.depth_dir, depth_files[mid_idx])
        
        # Load color frame
        self.representative_frame = cv2.imread(color_frame_path)
        self.representative_frame = cv2.cvtColor(self.representative_frame, cv2.COLOR_BGR2RGB)
        
        # Load depth frame
        with open(depth_frame_path, 'rb') as f:
            raw_depth = np.frombuffer(f.read(), dtype=np.uint16)
            h, w = self.representative_frame.shape[:2]
            self.depth_frame = raw_depth.reshape((h, w)) / 1000.0  # Convert to meters
        
        # Calculate scale factor for display
        orig_h, orig_w = self.representative_frame.shape[:2]
        if max(orig_h, orig_w) > self.display_size:
            self.scale_factor = self.display_size / max(orig_h, orig_w)
        else:
            self.scale_factor = 1.0
            
        print(f"📷 Selected frame: {color_files[mid_idx]}")
        print(f"🔍 Display scale factor: {self.scale_factor:.3f}")
        
        return color_files[mid_idx]
    
    def pixel_to_3d(self, pixel_coords, depth_frame):
        """Convert 2D clicks to 3D world coordinates."""
        x_px, y_px = pixel_coords
        
        # Get depth at clicked location with patch sampling for robustness
        patch_size = 5
        h, w = depth_frame.shape
        half = patch_size // 2
        
        y_start = max(0, int(y_px) - half)
        y_end = min(h, int(y_px) + half + 1)
        x_start = max(0, int(x_px) - half)
        x_end = min(w, int(x_px) + half + 1)
        
        patch = depth_frame[y_start:y_end, x_start:x_end]
        valid_depths = patch[patch > 0]
        
        if len(valid_depths) == 0:
            print(f"⚠️ No valid depth at ({x_px:.0f}, {y_px:.0f})")
            return None
            
        depth = np.median(valid_depths)
        
        # Convert to 3D using camera intrinsics
        x_3d = (x_px - self.intrinsics.ppx) * depth / self.intrinsics.fx
        y_3d = (y_px - self.intrinsics.ppy) * depth / self.intrinsics.fy
        z_3d = depth
        
        return [x_3d, y_3d, z_3d]
    
    def interactive_target_labeling(self):
        """GUI for clicking on 4 targets."""
        if self.representative_frame is None:
            self.select_representative_frame()
        
        # Resize frame for display
        display_h = int(self.representative_frame.shape[0] * self.scale_factor)
        display_w = int(self.representative_frame.shape[1] * self.scale_factor)
        display_frame = cv2.resize(self.representative_frame, (display_w, display_h))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_frame)
        ax.set_title("Click on 4 targets in the scene\n(Left click to place target, Right click to remove last)")
        
        # Track clicked points
        clicked_points = []
        target_circles = []
        
        def onclick(event):
            if event.inaxes != ax:
                return
                
            if event.button == 1:  # Left click - add target
                if len(clicked_points) >= 4:
                    print("⚠️ Maximum 4 targets allowed")
                    return
                    
                # Convert display coordinates back to original resolution
                orig_x = event.xdata / self.scale_factor
                orig_y = event.ydata / self.scale_factor
                
                # Normalize pixel coordinates to [0,1] range
                orig_h, orig_w = self.representative_frame.shape[:2]
                normalized_x = float(orig_x / orig_w)
                normalized_y = float(orig_y / orig_h)
                
                # Convert to 3D coordinates
                world_coords = self.pixel_to_3d([orig_x, orig_y], self.depth_frame)
                
                if world_coords is not None:
                    target_id = len(clicked_points) + 1
                    target_data = {
                        "id": target_id,
                        "label": f"target_{target_id}",
                        "pixel_coords": [float(orig_x), float(orig_y)],
                        "normalized_coords": [normalized_x, normalized_y],
                        "world_coords": world_coords,
                        "depth_m": float(world_coords[2])
                    }
                    
                    clicked_points.append(target_data)
                    
                    # Add visual marker
                    circle = Circle((event.xdata, event.ydata), 10, 
                                  color='red', fill=True, alpha=0.7)
                    ax.add_patch(circle)
                    ax.text(event.xdata + 15, event.ydata - 15, f"T{target_id}", 
                           color='white', fontsize=12, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))
                    target_circles.append(circle)
                    
                    print(f"✅ Target {target_id}: ({orig_x:.0f}, {orig_y:.0f}) -> normalized({normalized_x:.3f}, {normalized_y:.3f}) -> "
                          f"({world_coords[0]:.3f}, {world_coords[1]:.3f}, {world_coords[2]:.3f}) m")
                    
                    fig.canvas.draw()
                    
            elif event.button == 3:  # Right click - remove last target
                if clicked_points:
                    removed = clicked_points.pop()
                    if target_circles:
                        circle = target_circles.pop()
                        circle.remove()
                        # Remove text (more complex, just redraw)
                        ax.clear()
                        ax.imshow(display_frame)
                        ax.set_title("Click on 4 targets in the scene\n(Left click to place target, Right click to remove last)")
                        
                        # Redraw remaining targets
                        for i, target in enumerate(clicked_points):
                            display_x = target["pixel_coords"][0] * self.scale_factor
                            display_y = target["pixel_coords"][1] * self.scale_factor
                            circle = Circle((display_x, display_y), 10, 
                                          color='red', fill=True, alpha=0.7)
                            ax.add_patch(circle)
                            ax.text(display_x + 15, display_y - 15, f"T{i+1}", 
                                   color='white', fontsize=12, weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))
                            target_circles.append(circle)
                        
                        fig.canvas.draw()
                    print(f"🗑️ Removed target {removed['id']}")
        
        # Connect event handler
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.tight_layout()
        plt.show()
        
        self.targets = clicked_points
        return clicked_points
    
    def validate_targets(self):
        """Check target positions across multiple frames."""
        if len(self.targets) != 4:
            print(f"⚠️ Warning: Expected 4 targets, got {len(self.targets)}")
            return False
            
        # Check that targets are reasonably spaced
        positions = np.array([t["world_coords"] for t in self.targets])
        distances = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        min_dist = min(distances)
        max_dist = max(distances)
        
        print(f"📏 Target distances: min={min_dist:.3f}m, max={max_dist:.3f}m")
        
        if min_dist < 0.1:
            print("⚠️ Warning: Some targets are very close together")
            
        return True
    
    def save_target_coordinates(self):
        """Save labeled targets to JSON."""
        if not self.targets:
            print("❌ No targets to save")
            return None
            
        # Get the frame that was used for labeling
        frame_name = self.select_representative_frame()
        
        target_data = {
            "targets": self.targets,
            "labeling_frame": frame_name,
            "original_dimensions": {
                "width": self.intrinsics.width,
                "height": self.intrinsics.height
            },
            "camera_intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "ppx": self.intrinsics.ppx,
                "ppy": self.intrinsics.ppy,
                "coeffs": self.intrinsics.coeffs
            },
            "labeling_metadata": {
                "display_scale_factor": self.scale_factor,
                "tool_version": "1.0"
            }
        }
        
        output_path = os.path.join(self.trial_dir, "target_coordinates.json")
        with open(output_path, 'w') as f:
            json.dump(target_data, f, indent=2)
            
        print(f"💾 Saved target coordinates to {output_path}")
        return output_path
    
    def load_existing_targets(self):
        """Load existing target coordinates if available."""
        target_file = os.path.join(self.trial_dir, "target_coordinates.json")
        if os.path.exists(target_file):
            with open(target_file, 'r') as f:
                data = json.load(f)
            self.targets = data.get("targets", [])
            print(f"📥 Loaded {len(self.targets)} existing targets")
            return True
        return False


def label_single_trial(trial_dir, force_relabel=False):
    """Label targets for a single trial."""
    target_file = os.path.join(trial_dir, "target_coordinates.json")
    
    if os.path.exists(target_file) and not force_relabel:
        print(f"✅ Target coordinates already exist: {target_file}")
        return target_file
    
    print(f"🎯 Labeling targets for trial: {trial_dir}")
    
    labeler = TargetLabeler(trial_dir)
    
    # Try to load existing targets first
    if not force_relabel:
        labeler.load_existing_targets()
    
    # Perform interactive labeling
    targets = labeler.interactive_target_labeling()
    
    if targets:
        # Validate targets
        if labeler.validate_targets():
            # Save targets
            output_path = labeler.save_target_coordinates()
            return output_path
    
    print("❌ Target labeling incomplete or invalid")
    return None


def process_multiple_trials(base_dir, force_relabel=False):
    """Process multiple trials in a base directory."""
    processed_count = 0
    skipped_count = 0
    
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        # Check if this looks like a trial directory
        color_dir = os.path.join(item_path, "Color")
        depth_dir = os.path.join(item_path, "Depth")
        
        if os.path.isdir(color_dir) and os.path.isdir(depth_dir):
            # This is a trial directory
            result = label_single_trial(item_path, force_relabel)
            if result:
                processed_count += 1
            else:
                skipped_count += 1
        else:
            # This might be a subject directory with multiple trials
            for trial in sorted(os.listdir(item_path)):
                trial_path = os.path.join(item_path, trial)
                if not os.path.isdir(trial_path):
                    continue
                    
                color_dir = os.path.join(trial_path, "Color")
                depth_dir = os.path.join(trial_path, "Depth")
                
                if os.path.isdir(color_dir) and os.path.isdir(depth_dir):
                    result = label_single_trial(trial_path, force_relabel)
                    if result:
                        processed_count += 1
                    else:
                        skipped_count += 1
    
    print(f"\n📊 Summary: {processed_count} trials processed, {skipped_count} skipped")


def main():
    parser = argparse.ArgumentParser(description="Interactive Target Labeling Tool")
    parser.add_argument("--trial_dir", type=str, help="Single trial directory to label")
    parser.add_argument("--base_dir", type=str, help="Base directory containing multiple trials")
    parser.add_argument("--force", action="store_true", help="Force re-labeling even if targets exist")
    
    args = parser.parse_args()
    
    if args.trial_dir:
        # Label single trial
        trial_dir = os.path.expanduser(args.trial_dir)
        if not os.path.exists(trial_dir):
            print(f"❌ Trial directory not found: {trial_dir}")
            return
            
        result = label_single_trial(trial_dir, args.force)
        if result:
            print(f"✅ Successfully labeled trial: {trial_dir}")
        else:
            print(f"❌ Failed to label trial: {trial_dir}")
            
    elif args.base_dir:
        # Process multiple trials
        base_dir = os.path.expanduser(args.base_dir)
        if not os.path.exists(base_dir):
            print(f"❌ Base directory not found: {base_dir}")
            return
            
        process_multiple_trials(base_dir, args.force)
        
    else:
        print("❌ Please provide either --trial_dir or --base_dir")
        parser.print_help()


if __name__ == "__main__":
    main()