#!/usr/bin/env python3
"""
Pointing Gesture Analysis - Main UI

Integrated interface for the complete pointing gesture analysis pipeline.

Pages:
- Page 1: Target Detection (Step 0) - Load data and detect targets
- Page 2: Skeleton Processing (Step 2) - Extract skeleton and visualize 3D

Usage:
    python main_ui.py
"""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from step0_data_loading.ui_data_loader import DataLoaderUI
from step2_skeleton_extraction.ui_skeleton_extractor import SkeletonExtractorUI


class FrameWrapper(tk.Frame):
    """Wrapper that makes a Frame behave like a Tk root for embedding UIs."""

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=tk.BOTH, expand=True)
        self._menu = None

    def title(self, text):
        """Ignore title calls since we're in a frame."""
        pass

    def geometry(self, size):
        """Ignore geometry calls since we're in a frame."""
        pass

    def config(self, **kwargs):
        """Override config to ignore menu setting."""
        if 'menu' in kwargs:
            self._menu = kwargs.pop('menu')  # Store but don't apply
        if kwargs:  # If there are other options, pass them to Frame
            super().config(**kwargs)

    def configure(self, **kwargs):
        """Alias for config."""
        self.config(**kwargs)


class MainUI:
    """Main UI with two pages: Target Detection and Skeleton Processing"""

    def __init__(self, root):
        self.root = root
        self.root.title("Pointing Gesture Analysis - Main UI")

        # Adaptive window sizing - fits any screen resolution
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)  # 80% of screen width
        window_height = int(screen_height * 0.8)  # 80% of screen height

        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        self.root.minsize(1024, 768)  # Minimum usable size

        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create two pages
        self.create_page1_target_detection()
        self.create_page2_skeleton_processing()

        # Bind tab change to share trial info
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready - Start with Page 1 to detect targets", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_page1_target_detection(self):
        """Create Page 1: Target Detection (Step 0)"""
        # Create frame for Page 1
        page1_frame = ttk.Frame(self.notebook)
        self.notebook.add(page1_frame, text="📍 Page 1: Target Detection")

        # Add header with instructions
        header_frame = ttk.Frame(page1_frame)
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(
            header_frame,
            text="📍 Page 1: Target Detection",
            font=("Arial", 16, "bold"),
            foreground="blue"
        ).pack(anchor=tk.W)

        ttk.Label(
            header_frame,
            text="1. Load trial data  →  2. Detect and save targets  →  3. Switch to Page 2 for skeleton processing",
            font=("Arial", 11),
            foreground="#4A4A4A"
        ).pack(anchor=tk.W, pady=2)

        ttk.Separator(page1_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Create a wrapper frame that mimics a Tk root
        wrapper_frame = FrameWrapper(page1_frame)

        # Embed DataLoaderUI
        self.page1_ui = DataLoaderUI(wrapper_frame)

    def create_page2_skeleton_processing(self):
        """Create Page 2: Skeleton Processing (Step 2)"""
        # Create frame for Page 2
        page2_frame = ttk.Frame(self.notebook)
        self.notebook.add(page2_frame, text="🦴 Page 2: Skeleton Processing")

        # Add header with instructions
        header_frame = ttk.Frame(page2_frame)
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(
            header_frame,
            text="🦴 Page 2: Skeleton Processing & 3D Visualization",
            font=("Arial", 16, "bold"),
            foreground="green"
        ).pack(anchor=tk.W)

        ttk.Label(
            header_frame,
            text="1. Load trial  →  2. Process all frames (auto-saves)  →  3. View 3D skeleton with targets",
            font=("Arial", 11),
            foreground="#4A4A4A"
        ).pack(anchor=tk.W, pady=2)

        ttk.Separator(page2_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Create a wrapper frame that mimics a Tk root
        wrapper_frame = FrameWrapper(page2_frame)

        # Embed SkeletonExtractorUI
        self.page2_ui = SkeletonExtractorUI(wrapper_frame)

    def on_tab_changed(self, event):
        """Handle tab changes - auto-load trial from Page 1 to Page 2"""
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:
            # Switched to Page 1
            self.status_bar.config(text="Page 1: Load trial and detect targets")
        elif current_tab == 1:
            # Switched to Page 2
            # Auto-load trial from Page 1 if available
            if hasattr(self.page1_ui, 'trial_input_path') and self.page1_ui.trial_input_path:
                trial_path = self.page1_ui.trial_input_path

                # Load the trial in Page 2 (this will reload even if already loaded)
                self.load_trial_in_page2(trial_path)

                self.status_bar.config(
                    text=f"Page 2: Auto-loaded '{trial_path.name}' from Page 1 - Ready to process"
                )
            else:
                self.status_bar.config(text="Page 2: Load trial to process skeleton and view 3D visualization")

    def load_trial_in_page2(self, trial_path):
        """Load a trial in Page 2 skeleton extractor."""
        try:
            # Set the trial path
            self.page2_ui.trial_path = trial_path

            # Load original path from metadata (for syncing results back)
            self.page2_ui.load_original_path_from_config(trial_path)

            # Check for color folder
            color_folder = trial_path / "color"
            if not color_folder.exists():
                self.status_bar.config(text=f"Error: No 'color' folder found in {trial_path.name}")
                return

            # Load trial
            self.page2_ui.color_images = sorted(color_folder.glob("frame_*.png"))
            self.page2_ui.total_frames = len(self.page2_ui.color_images)

            if self.page2_ui.total_frames == 0:
                self.status_bar.config(text="Error: No frame images found in color folder")
                return

            # Detect camera intrinsics
            import cv2
            sample_img = cv2.imread(str(self.page2_ui.color_images[0]))
            h, w = sample_img.shape[:2]

            if w == 640 and h == 480:
                self.page2_ui.fx = self.page2_ui.fy = 615.0
                self.page2_ui.cx = 320.0
                self.page2_ui.cy = 240.0
            elif w == 1280 and h == 720:
                self.page2_ui.fx = self.page2_ui.fy = 922.5
                self.page2_ui.cx = 640.0
                self.page2_ui.cy = 360.0
            elif w == 1920 and h == 1080:
                self.page2_ui.fx = self.page2_ui.fy = 1383.75
                self.page2_ui.cx = 960.0
                self.page2_ui.cy = 540.0
            else:
                self.page2_ui.fx = self.page2_ui.fy = w * 0.9
                self.page2_ui.cx = w / 2.0
                self.page2_ui.cy = h / 2.0

            # Update UI elements
            self.page2_ui.trial_label.config(text=f"{trial_path.name}", foreground="black")
            self.page2_ui.frame_scale.config(to=self.page2_ui.total_frames)
            self.page2_ui.current_frame = 1
            self.page2_ui.frame_var.set(1)

            # Reset all subject-specific result dictionaries
            self.page2_ui.human_results = {}
            self.page2_ui.dog_results = {}
            self.page2_ui.baby_results = {}

            # Load ground plane transform and targets
            self.page2_ui.load_ground_plane_and_targets()

            # Load first frame
            self.page2_ui.load_frame(1)

            self.page2_ui.status_label.config(
                text=f"Auto-loaded {self.page2_ui.total_frames} frames from Page 1",
                foreground="green"
            )

            # Update main status bar with success message
            self.status_bar.config(
                text=f"✓ Loaded {trial_path.name}: {self.page2_ui.total_frames} frames, "
                f"Ground plane: {'Yes' if self.page2_ui.ground_plane_transform else 'No'}, "
                f"Targets: {len(self.page2_ui.targets) if self.page2_ui.targets else 0}"
            )

        except Exception as e:
            error_msg = f"Error loading trial in Page 2: {str(e)}"
            self.status_bar.config(text=error_msg)
            self.page2_ui.status_label.config(text=error_msg, foreground="red")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = MainUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
