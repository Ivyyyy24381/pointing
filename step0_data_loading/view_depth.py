#!/usr/bin/env python3
"""
Interactive depth image viewer with hover to see depth values.

Usage:
    python view_depth.py <trial_name> <camera_id> <frame_number>
    python view_depth.py trial_1 cam1 100
    python view_depth.py 1 None 50  # Single camera trial
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from step0_data_loading.trial_input_manager import TrialInputManager


class DepthViewer:
    def __init__(self, color_img: np.ndarray, depth_img: np.ndarray, window_name: str = "Depth Viewer"):
        self.color_img = color_img
        self.depth_img = depth_img
        self.window_name = window_name

        # Create depth colormap for visualization
        self.depth_colormap = self.create_depth_colormap(depth_img)

        # Current mouse position
        self.mouse_x = -1
        self.mouse_y = -1

    def create_depth_colormap(self, depth_img: np.ndarray) -> np.ndarray:
        """Create a colored depth map for visualization"""
        # Normalize depth to 0-255 for visualization
        valid_depth = depth_img[depth_img > 0]
        if len(valid_depth) > 0:
            depth_min = valid_depth.min()
            depth_max = valid_depth.max()
        else:
            depth_min = 0
            depth_max = 1

        # Normalize
        depth_normalized = np.zeros_like(depth_img)
        mask = depth_img > 0
        depth_normalized[mask] = ((depth_img[mask] - depth_min) / (depth_max - depth_min) * 255)
        depth_normalized = depth_normalized.astype(np.uint8)

        # Apply colormap
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colormap[~mask] = 0  # Set invalid depths to black

        return depth_colormap

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y

    def draw_info(self, img: np.ndarray) -> np.ndarray:
        """Draw depth information on the image"""
        display = img.copy()

        # Get depth at cursor
        if 0 <= self.mouse_y < self.depth_img.shape[0] and 0 <= self.mouse_x < self.depth_img.shape[1]:
            depth_value = self.depth_img[self.mouse_y, self.mouse_x]

            # Draw crosshair
            cv2.line(display, (self.mouse_x - 10, self.mouse_y), (self.mouse_x + 10, self.mouse_y),
                    (0, 255, 0), 1)
            cv2.line(display, (self.mouse_x, self.mouse_y - 10), (self.mouse_x, self.mouse_y + 10),
                    (0, 255, 0), 1)

            # Draw depth value
            if depth_value > 0:
                text = f"({self.mouse_x}, {self.mouse_y}): {depth_value:.3f}m"
            else:
                text = f"({self.mouse_x}, {self.mouse_y}): No depth"

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (self.mouse_x + 15, self.mouse_y - text_h - 5),
                         (self.mouse_x + 15 + text_w, self.mouse_y + 5), (0, 0, 0), -1)

            # Text
            cv2.putText(display, text, (self.mouse_x + 15, self.mouse_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw instructions
        instructions = [
            "Hover to see depth values",
            "Press 'c' - toggle Color/Depth",
            "Press 'q' - quit"
        ]
        y_offset = 30
        for i, instruction in enumerate(instructions):
            cv2.putText(display, instruction, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def run(self):
        """Run the interactive viewer"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        show_color = False  # Start with depth view

        print("Interactive Depth Viewer")
        print("- Hover mouse to see depth values")
        print("- Press 'c' to toggle between color and depth view")
        print("- Press 'q' to quit")

        while True:
            # Choose which image to display
            if show_color:
                display = self.draw_info(self.color_img)
            else:
                display = self.draw_info(self.depth_colormap)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_color = not show_color
                print(f"Switched to {'Color' if show_color else 'Depth'} view")

        cv2.destroyAllWindows()


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

    # Run interactive viewer
    viewer = DepthViewer(color, depth, window_name=f"{trial_name} - Frame {frame_number}")
    viewer.run()


if __name__ == "__main__":
    main()
