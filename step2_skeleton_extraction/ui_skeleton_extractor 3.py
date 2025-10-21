#!/usr/bin/env python3
"""
Step 2: Skeleton Extraction UI

Integrates all skeleton extraction components:
- MediaPipe human pose detection
- Real-time visualization
- Batch processing
- 3D landmark extraction from depth
- Arm vector computation
"""

import sys
import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for embedding in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector
from step2_skeleton_extraction.batch_processor import determine_pointing_hand_whole_trial
from step0_data_loading.load_trial_data_flexible import (
    load_color_flexible,
    load_depth_flexible,
    detect_folder_structure,
    detect_depth_shape
)


class SkeletonExtractorUI:
    """UI for skeleton extraction with MediaPipe."""

    def __init__(self, root, trial_path=None):
        self.root = root
        self.root.title("Step 2: Skeleton Extraction")

        # Make window size dynamic - fill 90% of screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)

        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # State variables
        self.trial_path = None  # Path to trial_input/ (standardized)
        self.original_trial_path = None  # Original user input path from Step 0
        self.camera_id = None
        self.current_frame = 1
        self.total_frames = 0
        self.color_images = []

        # Separate detectors for each subject type
        self.detector = None  # Human detector
        self.dog_detector = None  # Dog detector
        self.baby_detector = None  # Baby detector

        # Separate results dictionaries for each subject type
        self.human_results = {}
        self.dog_results = {}
        self.baby_results = {}

        self.playing = False

        # Camera intrinsics (auto-detected)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        # Current frame data
        self.current_color = None
        self.current_depth = None

        # Current results for each subject type
        self.current_human_result = None
        self.current_dog_result = None
        self.current_baby_result = None

        # 3D plot components
        self.plot_canvas = None
        self.plot_fig = None
        self.plot_ax = None

        # Ground plane transform (loaded once)
        self.ground_plane_transform = None
        self.targets = None

        self.setup_ui()
        self.initialize_detector()

        # Auto-load trial if provided
        if trial_path:
            self.root.after(100, lambda: self.load_trial_path(trial_path))

    def setup_ui(self):
        """Create the UI layout."""
        # Main container
        main_container = ttk.Frame(self.root, padding=10)
        main_container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Three-panel layout: Controls (left), 2D Image (middle), 3D Plot (right)
        left_panel = ttk.Frame(main_container, padding=10)
        left_panel.grid(row=0, column=0, sticky="nsew")

        middle_panel = ttk.Frame(main_container, padding=10)
        middle_panel.grid(row=0, column=1, sticky="nsew")

        right_panel = ttk.Frame(main_container, padding=10)
        right_panel.grid(row=0, column=2, sticky="nsew")

        main_container.columnconfigure(0, weight=0, minsize=350)  # Controls (fixed)
        main_container.columnconfigure(1, weight=1)               # 2D Image (expandable)
        main_container.columnconfigure(2, weight=1)               # 3D Plot (expandable)
        main_container.rowconfigure(0, weight=1)

        self.setup_controls(left_panel)
        self.setup_visualization(middle_panel)
        self.setup_3d_plot(right_panel)

    def setup_controls(self, parent):
        """Setup control panel."""
        # Trial loading
        load_frame = ttk.LabelFrame(parent, text="Trial Info", padding=10)
        load_frame.grid(row=0, column=0, sticky="ew", pady=5)

        self.trial_label = ttk.Label(load_frame, text="No trial loaded",
                                     foreground="gray")
        self.trial_label.pack(fill=tk.X, pady=2)

        # Detector settings
        settings_frame = ttk.LabelFrame(parent, text="Detector Settings", padding=10)
        settings_frame.grid(row=1, column=0, sticky="ew", pady=5)

        ttk.Label(settings_frame, text="Model Complexity:").grid(row=0, column=0, sticky="w")
        self.complexity_var = tk.IntVar(value=1)
        ttk.Radiobutton(settings_frame, text="0 (Fast)", variable=self.complexity_var,
                       value=0, command=self.reinitialize_detector).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(settings_frame, text="1 (Balanced)", variable=self.complexity_var,
                       value=1, command=self.reinitialize_detector).grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(settings_frame, text="2 (Accurate)", variable=self.complexity_var,
                       value=2, command=self.reinitialize_detector).grid(row=3, column=0, sticky="w")

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky="ew", pady=5)

        ttk.Label(settings_frame, text="Detection Confidence:").grid(row=5, column=0, sticky="w")
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0,
                              variable=self.conf_var, orient=tk.HORIZONTAL,
                              command=self.reinitialize_detector)
        conf_scale.grid(row=6, column=0, sticky="ew")
        self.conf_label = ttk.Label(settings_frame, text="0.50")
        self.conf_label.grid(row=7, column=0, sticky="w")

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=8, column=0, sticky="ew", pady=5)

        # Subject detection checkboxes
        ttk.Label(settings_frame, text="Detect Subjects:").grid(row=9, column=0, sticky="w")

        self.detect_human_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="👤 Human", variable=self.detect_human_var,
                       command=self.on_subject_selection_change).grid(row=10, column=0, sticky="w")

        self.detect_dog_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="🐶 Dog (lower half)", variable=self.detect_dog_var,
                       command=self.on_subject_selection_change).grid(row=11, column=0, sticky="w")

        self.detect_baby_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="👶 Baby (lower half)", variable=self.detect_baby_var,
                       command=self.on_subject_selection_change).grid(row=12, column=0, sticky="w")

        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=13, column=0, sticky="ew", pady=5)

        ttk.Label(settings_frame, text="Pointing Arm:").grid(row=14, column=0, sticky="w")
        self.arm_selection_var = tk.StringVar(value="auto")
        ttk.Radiobutton(settings_frame, text="Auto Detect", variable=self.arm_selection_var,
                       value="auto", command=self.on_arm_selection_change).grid(row=15, column=0, sticky="w")
        ttk.Radiobutton(settings_frame, text="Left Arm", variable=self.arm_selection_var,
                       value="left", command=self.on_arm_selection_change).grid(row=16, column=0, sticky="w")
        ttk.Radiobutton(settings_frame, text="Right Arm", variable=self.arm_selection_var,
                       value="right", command=self.on_arm_selection_change).grid(row=17, column=0, sticky="w")

        # Frame navigation
        nav_frame = ttk.LabelFrame(parent, text="Frame Navigation", padding=10)
        nav_frame.grid(row=2, column=0, sticky="ew", pady=5)

        button_frame = ttk.Frame(nav_frame)
        button_frame.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame, text="⏮", width=4,
                  command=self.first_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="◀", width=4,
                  command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(button_frame, text="▶", width=4,
                                      command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="▶▶", width=4,
                  command=self.next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏭", width=4,
                  command=self.last_frame).pack(side=tk.LEFT, padx=2)

        self.frame_var = tk.IntVar(value=1)
        self.frame_scale = ttk.Scale(nav_frame, from_=1, to=100,
                                     variable=self.frame_var, orient=tk.HORIZONTAL,
                                     command=self.on_frame_change)
        self.frame_scale.pack(fill=tk.X, pady=5)

        self.frame_label = ttk.Label(nav_frame, text="Frame: 1 / 0")
        self.frame_label.pack()

        # Processing
        process_frame = ttk.LabelFrame(parent, text="Processing", padding=10)
        process_frame.grid(row=3, column=0, sticky="ew", pady=5)

        ttk.Button(process_frame, text="🎯 Process Current Frame",
                  command=self.process_current_frame).pack(fill=tk.X, pady=2)
        ttk.Button(process_frame, text="🔄 Process All Frames (Auto-saves)",
                  command=self.process_all_frames).pack(fill=tk.X, pady=2)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(process_frame, variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(process_frame, text="Ready", foreground="gray")
        self.status_label.pack()

        # Results info
        info_frame = ttk.LabelFrame(parent, text="Detection Info", padding=10)
        info_frame.grid(row=4, column=0, sticky="nsew", pady=5)
        parent.rowconfigure(4, weight=1)

        self.info_text = tk.Text(info_frame, height=12, width=40, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)

    def setup_visualization(self, parent):
        """Setup visualization panel."""
        # Display options
        options_frame = ttk.Frame(parent)
        options_frame.pack(fill=tk.X, pady=5)

        self.show_skeleton_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Skeleton",
                       variable=self.show_skeleton_var,
                       command=self.update_display).pack(side=tk.LEFT, padx=5)

        self.show_vectors_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Arm Vectors",
                       variable=self.show_vectors_var,
                       command=self.update_display).pack(side=tk.LEFT, padx=5)

        self.show_depth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Show Depth",
                       variable=self.show_depth_var,
                       command=self.update_display).pack(side=tk.LEFT, padx=5)

        # Canvas for image display
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind resize event
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def setup_3d_plot(self, parent):
        """Setup 3D plot panel."""
        # Title
        title_label = ttk.Label(parent, text="3D Skeleton View", font=("Arial", 12, "bold"))
        title_label.pack(pady=5)

        # Create matplotlib figure for 3D plot
        self.plot_fig = plt.figure(figsize=(6, 6))
        self.plot_ax = self.plot_fig.add_subplot(111, projection='3d')
        self.plot_ax.set_xlabel('X (m)')
        self.plot_ax.set_ylabel('Y (m)')
        self.plot_ax.set_zlabel('Z (m)')
        self.plot_ax.set_title('3D Skeleton')

        # Embed matplotlib in Tkinter
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=parent)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def initialize_detector(self):
        """Initialize detector based on selected subject types."""
        try:
            complexity = self.complexity_var.get()
            confidence = self.conf_var.get()

            # Check which subject types are enabled
            detect_human = self.detect_human_var.get()
            detect_dog = self.detect_dog_var.get()
            detect_baby = self.detect_baby_var.get()

            # Initialize human detector if enabled
            if detect_human:
                self.detector = MediaPipeHumanDetector(
                    min_detection_confidence=confidence,
                    min_tracking_confidence=confidence,
                    model_complexity=complexity
                )
            else:
                self.detector = None

            # Initialize subject detectors if enabled
            if detect_dog:
                from step3_subject_extraction import SubjectDetector
                self.dog_detector = SubjectDetector(subject_type='dog', crop_ratio=0.6)
            else:
                self.dog_detector = None

            if detect_baby:
                from step3_subject_extraction import SubjectDetector
                self.baby_detector = SubjectDetector(subject_type='baby', crop_ratio=0.6)
            else:
                self.baby_detector = None

            # Build status message
            enabled = []
            if detect_human:
                enabled.append("Human")
            if detect_dog:
                enabled.append("Dog")
            if detect_baby:
                enabled.append("Baby")
            status_msg = f"Detectors ready: {', '.join(enabled)}" if enabled else "No detectors enabled"
            self.status_label.config(text=status_msg, foreground="green")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize detector:\n{e}")
            self.status_label.config(text="Detector error", foreground="red")
            import traceback
            traceback.print_exc()

    def reinitialize_detector(self, *args):
        """Reinitialize detector with new settings."""
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")
        self.initialize_detector()
        if self.current_color is not None:
            self.process_current_frame()

    def on_subject_selection_change(self):
        """Handle subject selection change - reinitialize detectors."""
        self.reinitialize_detector()

    def on_arm_selection_change(self):
        """Handle manual arm selection change - applies to ALL loaded results."""
        selected_arm = self.arm_selection_var.get()

        if selected_arm != "auto":
            # Update ALL loaded human results with the selected arm
            if self.human_results:
                for frame_key, result in self.human_results.items():
                    if result.landmarks_3d:
                        result.metadata['pointing_arm'] = selected_arm
                        result.arm_vectors = self.detector._compute_arm_vectors(
                            result.landmarks_3d,
                            selected_arm
                        )

            # Update current human result if available
            if self.current_human_result and self.current_human_result.landmarks_3d:
                self.current_human_result.metadata['pointing_arm'] = selected_arm
                self.current_human_result.arm_vectors = self.detector._compute_arm_vectors(
                    self.current_human_result.landmarks_3d,
                    selected_arm
                )

            # Update display
            self.update_display()
            self.update_info()
            self.status_label.config(text=f"Updated to {selected_arm} arm", foreground="blue")

        else:
            # Auto mode - reprocess current frame
            if self.current_color is not None:
                self.process_current_frame()

    def on_subject_type_change(self):
        """Handle subject type selection changes (checkboxes)."""
        # Get current selection states
        detect_human = self.detect_human_var.get()
        detect_dog = self.detect_dog_var.get()
        detect_baby = self.detect_baby_var.get()

        enabled_types = []
        if detect_human:
            enabled_types.append("Human")
        if detect_dog:
            enabled_types.append("Dog")
        if detect_baby:
            enabled_types.append("Baby")

        # Show info for newly enabled types (only show once per session)
        if detect_dog and not hasattr(self, '_dog_info_shown'):
            messagebox.showinfo("Dog Detection",
                              "🐶 Dog detection enabled!\n\n"
                              "⚠️  Note: Dog detection is not yet fully implemented.\n"
                              "Frame-by-frame processing is not available.\n\n"
                              "For now, please use Human or Baby detection.")
            self._dog_info_shown = True

        if detect_baby and not hasattr(self, '_baby_info_shown'):
            messagebox.showinfo("Baby Detection",
                              "👶 Baby detection enabled!\n\n"
                              "Uses MediaPipe Pose on lower half of image\n"
                              "to avoid detecting adults.\n\n"
                              "Processes frames automatically.")
            self._baby_info_shown = True

        # Reinitialize detector with new subject type
        self.reinitialize_detector()

        # Note: Human detection is always on (checkbox is disabled)

    def load_original_path_from_config(self, trial_path):
        """Load the original user input path from metadata.json saved by Step 0."""
        import json
        import glob

        # Try to find any metadata*.json file
        metadata_files = list(trial_path.glob("metadata*.json"))

        if metadata_files:
            metadata_file = metadata_files[0]  # Use the first one found
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                original_path_str = metadata.get("trial_path")
                if original_path_str:
                    self.original_trial_path = Path(original_path_str).resolve()

                    # Check if original path still exists
                    if not self.original_trial_path.exists():
                        self.original_trial_path = None
                    return

            except Exception:
                pass

        # Fallback to source_path.txt (if saved)
        source_path_file = trial_path / "source_path.txt"
        if source_path_file.exists():
            try:
                with open(source_path_file, 'r') as f:
                    original_path_str = f.read().strip()

                self.original_trial_path = Path(original_path_str).resolve()

                # Check if original path still exists
                if not self.original_trial_path.exists():
                    self.original_trial_path = None
                return

            except Exception:
                pass

        # No source info found
        self.original_trial_path = None

    def load_trial_path(self, trial_path):
        """Load trial from a given path."""
        # Convert to Path object if string
        if isinstance(trial_path, str):
            trial_path = Path(trial_path).resolve()
        else:
            trial_path = trial_path.resolve()

        # Check for color folder
        color_folder = trial_path / "color"
        if not color_folder.exists():
            messagebox.showerror("Error", "No 'color' folder found in selected directory")
            return

        # Try to load original trial path from config
        self.load_original_path_from_config(trial_path)

        # Load trial
        self.trial_path = trial_path
        self.color_images = sorted(color_folder.glob("frame_*.png"))
        self.total_frames = len(self.color_images)

        if self.total_frames == 0:
            messagebox.showerror("Error", "No frame images found in color folder")
            return

        # Detect camera intrinsics
        sample_img = cv2.imread(str(self.color_images[0]))
        h, w = sample_img.shape[:2]

        if w == 640 and h == 480:
            self.fx = self.fy = 615.0
            self.cx = 320.0
            self.cy = 240.0
        elif w == 1280 and h == 720:
            self.fx = self.fy = 922.5
            self.cx = 640.0
            self.cy = 360.0
        elif w == 1920 and h == 1080:
            self.fx = self.fy = 1383.75
            self.cx = 960.0
            self.cy = 540.0
        else:
            self.fx = self.fy = w * 0.9
            self.cx = w / 2.0
            self.cy = h / 2.0

        # Update UI
        self.trial_label.config(text=f"{trial_path.name}", foreground="black")
        self.frame_scale.config(to=self.total_frames)
        self.current_frame = 1
        self.frame_var.set(1)
        self.results = {}

        # Load ground plane transform and targets
        self.load_ground_plane_and_targets()

        # Load first frame
        self.load_frame(1)

        self.status_label.config(text=f"Loaded {self.total_frames} frames", foreground="blue")

    def load_ground_plane_and_targets(self):
        """Load ground plane transform and targets from trial folder."""
        if not self.trial_path:
            return

        # Try to find ground_plane_transform.json in multiple locations
        trial_input_base = Path("trial_input")

        # Extract trial name and camera from path
        # Path could be: trial_input/trial_1/cam1
        camera_name = self.trial_path.name  # cam1
        trial_name = self.trial_path.parent.name  # trial_1

        possible_transform_paths = [
            self.trial_path.parent / "ground_plane_transform.json",  # Same level as cam folders
            self.trial_path / "ground_plane_transform.json",  # In cam folder
            trial_input_base / trial_name / "ground_plane_transform.json",  # trial_input/trial_X/
        ]

        self.ground_plane_transform = None
        for path in possible_transform_paths:
            if path.exists():
                print(f"✅ Loaded ground plane transform: {path}")
                with open(path) as f:
                    transform_data = json.load(f)
                    self.ground_plane_transform = np.array(transform_data['rotation_matrix'])
                break

        # Try to load targets from multiple locations
        trial_output_base = Path("trial_output")
        possible_target_paths = [
            self.trial_path.parent / "target_detections_cam_frame.json",  # Same level as cam folders
            self.trial_path / "target_detections_cam_frame.json",  # In cam folder
            trial_input_base / trial_name / "target_detections_cam_frame.json",  # trial_input/trial_X/
            trial_input_base / trial_name / camera_name / "target_detections_cam_frame.json",  # trial_input/trial_X/cam1/
            trial_output_base / trial_name / camera_name / "target_detections_cam_frame.json",  # trial_output/trial_X/cam1/
        ]

        self.targets = None
        for path in possible_target_paths:
            if path.exists():
                with open(path) as f:
                    self.targets = json.load(f)
                break

    def load_frame(self, frame_num: int):
        """Load specific frame."""
        if frame_num < 1 or frame_num > self.total_frames:
            return

        self.current_frame = frame_num
        self.frame_var.set(frame_num)
        self.frame_label.config(text=f"Frame: {frame_num} / {self.total_frames}")

        # Load color
        color_path = self.color_images[frame_num - 1]
        self.current_color = cv2.imread(str(color_path))
        self.current_color = cv2.cvtColor(self.current_color, cv2.COLOR_BGR2RGB)

        # Extract actual frame number from filename (e.g., "frame_000227.png" → 227)
        actual_frame_num = int(color_path.stem.split('_')[1])

        # Load depth if available
        depth_folder = self.trial_path / "depth"
        if depth_folder.exists():
            try:
                depth_npy = depth_folder / f"frame_{actual_frame_num:06d}.npy"
                depth_raw = depth_folder / f"frame_{actual_frame_num:06d}.raw"

                if depth_npy.exists():
                    self.current_depth = np.load(str(depth_npy))
                    print(f"✅ Loaded depth from: {depth_npy.name}, shape: {self.current_depth.shape}")
                elif depth_raw.exists():
                    # Load using flexible loader
                    depth_data = np.fromfile(str(depth_raw), dtype=np.uint16)
                    # Detect shape
                    h, w = self.current_color.shape[:2]
                    self.current_depth = depth_data.reshape((h, w)).astype(np.float32) / 1000.0
                    print(f"✅ Loaded depth from: {depth_raw.name}, shape: {self.current_depth.shape}")
                else:
                    print(f"⚠️  No depth file found: {depth_npy.name} or {depth_raw.name}")
                    self.current_depth = None
            except Exception as e:
                print(f"❌ Could not load depth: {e}")
                import traceback
                traceback.print_exc()
                self.current_depth = None
        else:
            print(f"⚠️  Depth folder not found: {depth_folder}")
            self.current_depth = None

        # Check if already processed
        # Use actual frame number from filename to match what process_all_frames uses
        frame_key = f"frame_{actual_frame_num:06d}"

        # Load results for each subject type
        if frame_key in self.human_results:
            self.current_human_result = self.human_results[frame_key]
        else:
            self.current_human_result = None

        if frame_key in self.dog_results:
            self.current_dog_result = self.dog_results[frame_key]
        else:
            self.current_dog_result = None

        if frame_key in self.baby_results:
            self.current_baby_result = self.baby_results[frame_key]
        else:
            self.current_baby_result = None

        self.update_display()
        self.update_info()

    def process_current_frame(self):
        """Process current frame - detects all enabled subject types."""
        if self.current_color is None:
            return

        self.status_label.config(text="Processing...", foreground="orange")
        self.root.update()

        try:
            # Get actual frame number from filename
            color_path = self.color_images[self.current_frame - 1]
            actual_frame_num = int(color_path.stem.split('_')[1])
            frame_key = f"frame_{actual_frame_num:06d}"

            # Detect human if enabled
            if self.detector is not None:
                human_result = self.detector.detect_frame(
                    self.current_color,
                    actual_frame_num,
                    depth_image=self.current_depth,
                    fx=self.fx, fy=self.fy,
                    cx=self.cx, cy=self.cy
                )
                if human_result:
                    self.human_results[frame_key] = human_result
                    self.current_human_result = human_result
            else:
                self.current_human_result = None

            # Detect dog if enabled
            if self.dog_detector is not None:
                dog_result = self.dog_detector.detect_frame(
                    self.current_color,
                    actual_frame_num,
                    depth_image=self.current_depth,
                    fx=self.fx, fy=self.fy,
                    cx=self.cx, cy=self.cy
                )
                if dog_result:
                    self.dog_results[frame_key] = dog_result
                    self.current_dog_result = dog_result
            else:
                self.current_dog_result = None

            # Detect baby if enabled
            if self.baby_detector is not None:
                baby_result = self.baby_detector.detect_frame(
                    self.current_color,
                    actual_frame_num,
                    depth_image=self.current_depth,
                    fx=self.fx, fy=self.fy,
                    cx=self.cx, cy=self.cy
                )
                if baby_result:
                    self.baby_results[frame_key] = baby_result
                    self.current_baby_result = baby_result
            else:
                self.current_baby_result = None

            # Update status
            detected = []
            if self.current_human_result: detected.append("human")
            if self.current_dog_result: detected.append("dog")
            if self.current_baby_result: detected.append("baby")

            if detected:
                self.status_label.config(text=f"Detected: {', '.join(detected)}", foreground="green")
            else:
                self.status_label.config(text="No subjects detected", foreground="orange")

            self.update_display()
            self.update_info()

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")
            self.status_label.config(text="Processing error", foreground="red")
            import traceback
            traceback.print_exc()

    def process_all_frames(self):
        """Process all frames in batch."""
        if not self.color_images:
            messagebox.showwarning("Warning", "No trial loaded")
            return

        response = messagebox.askyesno(
            "Confirm Batch Processing",
            f"Process all {self.total_frames} frames?\nThis may take several minutes."
        )

        if not response:
            return

        # Clear previous results
        self.human_results = {}
        self.dog_results = {}
        self.baby_results = {}
        self.progress_var.set(0)

        skipped_frames = 0

        # Lists to store all frames and metadata for batch processing
        all_frames = []
        all_frame_numbers = []
        all_depth_images = []

        # Phase 1: Load all frames and process human/baby frame-by-frame
        print(f"\n{'='*60}")
        print(f"PHASE 1: Loading frames and processing human/baby")
        print(f"{'='*60}")

        for i, color_path in enumerate(self.color_images, 1):
            # Load frame
            frame_num = int(color_path.stem.split('_')[-1])
            color_img = cv2.imread(str(color_path))

            # Skip if frame couldn't be loaded
            if color_img is None:
                print(f"⚠️ Skipping frame {frame_num}: Could not read {color_path}")
                skipped_frames += 1
                continue

            color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

            # Load depth
            depth_img = None
            depth_folder = self.trial_path / "depth"
            if depth_folder.exists():
                try:
                    depth_npy = depth_folder / f"frame_{frame_num:06d}.npy"
                    depth_raw = depth_folder / f"frame_{frame_num:06d}.raw"

                    if depth_npy.exists():
                        depth_img = np.load(str(depth_npy))
                    elif depth_raw.exists():
                        h, w = color_rgb.shape[:2]
                        depth_data = np.fromfile(str(depth_raw), dtype=np.uint16)
                        depth_img = depth_data.reshape((h, w)).astype(np.float32) / 1000.0
                except:
                    pass

            # Store for batch processing
            all_frames.append(color_rgb)
            all_frame_numbers.append(frame_num)
            all_depth_images.append(depth_img)

            frame_key = f"frame_{frame_num:06d}"

            # Detect human if enabled (frame-by-frame)
            if self.detector is not None:
                human_result = self.detector.detect_frame(
                    color_rgb, frame_num,
                    depth_image=depth_img,
                    fx=self.fx, fy=self.fy,
                    cx=self.cx, cy=self.cy
                )
                if human_result:
                    self.human_results[frame_key] = human_result

            # Detect baby if enabled (frame-by-frame with MediaPipe)
            if self.baby_detector is not None:
                baby_result = self.baby_detector.detect_frame(
                    color_rgb, frame_num,
                    depth_image=depth_img,
                    fx=self.fx, fy=self.fy,
                    cx=self.cx, cy=self.cy
                )
                if baby_result:
                    self.baby_results[frame_key] = baby_result

            # Update visualization if this is the current frame being viewed
            if i == self.current_frame:
                self.current_human_result = self.human_results.get(frame_key)
                self.current_baby_result = self.baby_results.get(frame_key)
                self.current_color = color_rgb
                self.current_depth = depth_img

            # Update progress (Phase 1: 0-50%)
            progress = (i / self.total_frames) * 50
            self.progress_var.set(progress)
            self.status_label.config(
                text=f"Phase 1: Processing {i}/{self.total_frames} (Human/Baby)...",
                foreground="orange"
            )
            self.root.update()

        # Phase 2: Batch process dog detection with DeepLabCut
        if self.dog_detector is not None and len(all_frames) > 0:
            print(f"\n{'='*60}")
            print(f"PHASE 2: Batch processing dog detection with DeepLabCut")
            print(f"{'='*60}")

            self.status_label.config(
                text=f"Phase 2: Creating video for dog detection...",
                foreground="orange"
            )
            self.root.update()

            # Create temporary video file
            camera_name = self.trial_path.name
            trial_name = self.trial_path.parent.name
            if camera_name == trial_name:
                temp_video_path = Path("trial_output") / camera_name / "temp_dog_detection.mp4"
            else:
                temp_video_path = Path("trial_output") / trial_name / camera_name / "temp_dog_detection.mp4"

            temp_video_path.parent.mkdir(parents=True, exist_ok=True)

            # Run batch dog detection
            try:
                dog_results_dict = self.dog_detector.process_batch_video(
                    str(temp_video_path),
                    all_frames,
                    all_frame_numbers,
                    depth_images=all_depth_images,
                    fx=self.fx, fy=self.fy,
                    cx=self.cx, cy=self.cy
                )

                # Merge dog results
                self.dog_results.update(dog_results_dict)

                # Update current dog result if viewing current frame
                if self.current_frame <= len(all_frames):
                    frame_key = f"frame_{all_frame_numbers[self.current_frame - 1]:06d}"
                    self.current_dog_result = self.dog_results.get(frame_key)

                print(f"✅ Dog detection complete: {len(dog_results_dict)} frames processed")

            except Exception as e:
                print(f"❌ Dog batch processing failed: {e}")
                import traceback
                traceback.print_exc()

        # Complete progress
        self.progress_var.set(100)
        self.status_label.config(
            text=f"Processing complete",
            foreground="green"
        )
        self.root.update()

        # Post-process: determine pointing arm for human results only
        selected_arm = self.arm_selection_var.get()
        pointing_hand = None  # Initialize to None

        if self.human_results:
            if selected_arm == "auto":
                results_list = list(self.human_results.values())
                pointing_hand = determine_pointing_hand_whole_trial(results_list)
                print(f"\n{'='*60}")
                print(f"👆 AUTO-DETECTED POINTING HAND: {pointing_hand.upper()}")
                print(f"{'='*60}\n")
            else:
                pointing_hand = selected_arm
                print(f"\n{'='*60}")
                print(f"👆 USER-SELECTED POINTING HAND: {pointing_hand.upper()}")
                print(f"{'='*60}\n")

            # Update all human results with the determined pointing hand
            for result in self.human_results.values():
                result.metadata['pointing_hand_whole_trial'] = pointing_hand
                if pointing_hand in ['left', 'right'] and result.landmarks_3d:
                    result.arm_vectors = self.detector._compute_arm_vectors(
                        result.landmarks_3d,
                        pointing_hand
                    )
                    result.metadata['pointing_arm'] = pointing_hand

        # Show processing summary
        summary_msg = f"Processed: {len(self.human_results)} human, {len(self.dog_results)} dog, {len(self.baby_results)} baby"
        if skipped_frames > 0:
            summary_msg += f" ({skipped_frames} frames skipped)"
        self.status_label.config(text=summary_msg, foreground="green")

        if skipped_frames > 0:
            print(f"\n⚠️ Warning: {skipped_frames} frames were skipped due to read errors")

        # Compute pointing analysis for human results
        if self.human_results and self.targets:
            self.status_label.config(text="Computing pointing analysis...", foreground="orange")
            self.root.update()

            from step2_skeleton_extraction.pointing_analysis import analyze_pointing_frame
            from step2_skeleton_extraction.csv_exporter import export_pointing_analysis_to_csv

            analyses = {}
            for frame_key, result in self.human_results.items():
                if result.landmarks_3d:
                    analysis = analyze_pointing_frame(
                        result,
                        self.targets,
                        pointing_arm=result.metadata.get('pointing_arm', 'right')
                    )
                    if analysis:
                        analyses[frame_key] = analysis

            # Save to trial_output (temporary working directory)
            camera_name = self.trial_path.name
            trial_name = self.trial_path.parent.name

            # Create trial_output path
            if camera_name == trial_name:
                output_path = Path("trial_output") / camera_name
            else:
                output_path = Path("trial_output") / trial_name / camera_name

            print(f"\n💾 Saving results to trial_output: {output_path}")
            print(f"   Trial: {trial_name}")
            print(f"   Camera: {camera_name}")

            output_path.mkdir(parents=True, exist_ok=True)
            csv_path = output_path / "processed_gesture.csv"

            export_pointing_analysis_to_csv(
                self.human_results,
                analyses,
                csv_path,
                global_start_frame=0
            )

            print(f"\n✅ Pointing analysis complete: {len(analyses)} frames analyzed")

            # Generate 2D pointing trace plot
            from step2_skeleton_extraction.plot_pointing_trace import plot_2d_pointing_trace

            # Compute human center position (hip center average across all frames)
            human_positions = []
            for result in self.human_results.values():
                if result.landmarks_3d and len(result.landmarks_3d) > 24:
                    left_hip = np.array(result.landmarks_3d[23])
                    right_hip = np.array(result.landmarks_3d[24])
                    hip_center = (left_hip + right_hip) / 2.0
                    human_positions.append(hip_center)

            if human_positions:
                human_center = np.mean(human_positions, axis=0).tolist()
            else:
                human_center = [0, 0, 0]

            # Create plot
            plot_path = output_path / "2d_pointing_trace.png"
            plot_2d_pointing_trace(
                analyses,
                self.targets,
                human_center,
                plot_path,
                trial_name=f"{trial_name}_{camera_name}"
            )

        # Auto-save to trial_output (temporary)
        self.auto_save_results(pointing_hand)

        # Save 3D visualizations to trial_output/fig/
        self.save_3d_visualizations()

        # Copy all results from trial_output back to original input path
        self.sync_results_to_input()

        # Update display with already-processed results (don't reload/reprocess)
        self.update_display()
        self.update_info()
        self.update_3d_plot()

    def save_results(self):
        """Save skeleton extraction results."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save")
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir="trial_output"
        )

        if not output_dir:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert results to dict format
        output_data = {}
        for frame_key, result in self.results.items():
            output_data[frame_key] = result.to_dict()

        # Save JSON
        json_file = output_path / "skeleton_2d.json"
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Save summary
        summary_file = output_path / "skeleton_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Skeleton Extraction Summary\n")
            f.write(f"{'='*40}\n")
            f.write(f"Trial: {self.trial_path.name if self.trial_path else 'unknown'}\n")
            f.write(f"Total frames processed: {len(self.results)}\n")
            f.write(f"Detector: MediaPipe Pose\n")
            f.write(f"Model complexity: {self.complexity_var.get()}\n")
            f.write(f"Detection confidence: {self.conf_var.get():.2f}\n")
            f.write(f"\nPointing arm distribution:\n")

            # Count pointing arms
            pointing_arms = {}
            for result in self.results.values():
                arm = result.metadata.get('pointing_arm', 'unknown')
                pointing_arms[arm] = pointing_arms.get(arm, 0) + 1

            for arm, count in sorted(pointing_arms.items()):
                pct = (count / len(self.results)) * 100
                f.write(f"  {arm}: {count} frames ({pct:.1f}%)\n")

        messagebox.showinfo("Success",
                           f"Saved results to:\n{json_file}\n{summary_file}")
        self.status_label.config(text="Results saved", foreground="green")

    def auto_save_results(self, pointing_hand: str = None):
        """Automatically save results after batch processing."""
        if not self.trial_path:
            return

        # Save to trial_output (temporary working directory)
        camera_name = self.trial_path.name
        trial_name = self.trial_path.parent.name

        # Create trial_output path
        if camera_name == trial_name:
            output_path = Path("trial_output") / camera_name
        else:
            output_path = Path("trial_output") / trial_name / camera_name

        output_path.mkdir(parents=True, exist_ok=True)

        # Save human results (existing format: skeleton_2d.json)
        if self.human_results:
            output_data = {}
            for frame_key, result in self.human_results.items():
                output_data[frame_key] = result.to_dict()

            json_file = output_path / "skeleton_2d.json"
            with open(json_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Saved human results: {json_file}")

        # Save dog results (new: dog_detection_results.json)
        if self.dog_results:
            output_data = {}
            for frame_key, result in self.dog_results.items():
                output_data[frame_key] = result.to_dict()

            json_file = output_path / "dog_detection_results.json"
            with open(json_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Saved dog results: {json_file}")

            # Generate plots for dog
            try:
                from step2_skeleton_extraction.subject_plot_generator import generate_subject_plots
                generate_subject_plots(self.trial_path, subject_type='dog', subject_name='Dog',
                                     fps=30.0, output_path_override=output_path)
            except Exception as e:
                print(f"⚠️ Could not generate dog plots: {e}")
                import traceback
                traceback.print_exc()

        # Save baby results (new: baby_detection_results.json)
        if self.baby_results:
            output_data = {}
            for frame_key, result in self.baby_results.items():
                output_data[frame_key] = result.to_dict()

            json_file = output_path / "baby_detection_results.json"
            with open(json_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Saved baby results: {json_file}")

            # Generate plots for baby
            try:
                from step2_skeleton_extraction.subject_plot_generator import generate_subject_plots
                generate_subject_plots(self.trial_path, subject_type='baby', subject_name='Baby',
                                     fps=30.0, output_path_override=output_path)
            except Exception as e:
                print(f"⚠️ Could not generate baby plots: {e}")
                import traceback
                traceback.print_exc()

        # Save summary
        if self.human_results or self.dog_results or self.baby_results:
            summary_file = output_path / "detection_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Detection Summary\n")
                f.write(f"{'='*40}\n")
                f.write(f"Trial: {trial_name}\n")
                f.write(f"Camera: {self.trial_path.name}\n")
                f.write(f"\nDetection Results:\n")
                f.write(f"  Human frames: {len(self.human_results)}\n")
                f.write(f"  Dog frames: {len(self.dog_results)}\n")
                f.write(f"  Baby frames: {len(self.baby_results)}\n")
                f.write(f"\nDetector Settings:\n")
                f.write(f"  Model complexity: {self.complexity_var.get()}\n")
                f.write(f"  Detection confidence: {self.conf_var.get():.2f}\n")

                if pointing_hand and self.human_results:
                    f.write(f"\n{'='*40}\n")
                    f.write(f"POINTING HAND (whole trial): {pointing_hand.upper()}\n")
                    f.write(f"{'='*40}\n")
                    f.write(f"\nPer-frame pointing arm distribution:\n")

                    # Count pointing arms
                    pointing_arms = {}
                    for result in self.human_results.values():
                        arm = result.metadata.get('pointing_arm', 'unknown')
                        pointing_arms[arm] = pointing_arms.get(arm, 0) + 1

                    for arm, count in sorted(pointing_arms.items()):
                        pct = (count / len(self.human_results)) * 100
                        f.write(f"  {arm}: {count} frames ({pct:.1f}%)\n")

            print(f"✅ Saved summary: {summary_file}")

            # Save pointing hand to separate JSON if human results exist
            if pointing_hand and self.human_results:
                pointing_hand_file = output_path / "pointing_hand.json"
                with open(pointing_hand_file, 'w') as f:
                    json.dump({
                        "trial": trial_name,
                        "camera": self.trial_path.name,
                        "pointing_hand": pointing_hand,
                        "total_frames": len(self.human_results),
                        "frame_distribution": pointing_arms
                    }, f, indent=2)
                print(f"✅ Saved pointing hand: {pointing_hand_file}")

        print(f"\n✅ Auto-saved results to: {output_path}")

    def save_3d_visualizations(self):
        """Save 3D skeleton visualizations for all frames to /fig folder."""
        if not self.trial_path or not self.human_results:
            return

        # Save to trial_output
        camera_name = self.trial_path.name
        trial_name = self.trial_path.parent.name

        if camera_name == trial_name:
            output_path = Path("trial_output") / camera_name
        else:
            output_path = Path("trial_output") / trial_name / camera_name

        # Create fig directory in trial_output
        fig_dir = output_path / "fig"
        fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Saving 3D visualizations to {fig_dir}...")

        from step2_skeleton_extraction.visualize_skeleton_3d import plot_skeleton_3d
        from step2_skeleton_extraction.pointing_analysis import compute_head_orientation

        # Import matplotlib for saving
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        total_frames = len(self.human_results)
        saved_count = 0

        for idx, (frame_key, result) in enumerate(self.human_results.items(), 1):
            if result.landmarks_3d:
                # Create new figure for this frame
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Compute head orientation
                head_orientation = None
                try:
                    head_vec, head_origin = compute_head_orientation(result.landmarks_3d)
                    head_orientation = {
                        'head_orientation_vector': head_vec.tolist(),
                        'head_orientation_origin': head_origin.tolist()
                    }
                except Exception as e:
                    pass

                # Plot skeleton
                try:
                    plot_skeleton_3d(
                        result.landmarks_3d,
                        arm_vectors=result.arm_vectors,
                        frame_name=f"{frame_key}",
                        targets=self.targets,
                        show=False,
                        ax=ax,
                        head_orientation=head_orientation
                    )

                    # Save figure
                    output_file = fig_dir / f"{frame_key}.png"
                    plt.savefig(output_file, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    saved_count += 1

                    # Update progress every 10 frames
                    if idx % 10 == 0:
                        print(f"   Saved {idx}/{total_frames} frames...")

                except Exception as e:
                    print(f"   ⚠️ Failed to save {frame_key}: {e}")
                    plt.close(fig)

        print(f"✅ Saved {saved_count} 3D visualization frames to: {fig_dir}")

    def sync_results_to_input(self):
        """Copy all results from trial_output back to original input path."""
        if not self.original_trial_path:
            print(f"⚠️ No original path configured - skipping sync")
            print(f"   Results remain in trial_output/")
            return

        import shutil

        # Determine output_path in trial_output
        camera_name = self.trial_path.name
        trial_name = self.trial_path.parent.name

        if camera_name == trial_name:
            source_path = Path("trial_output") / camera_name
        else:
            source_path = Path("trial_output") / trial_name / camera_name

        destination_path = self.original_trial_path

        # List of files/folders to copy
        items_to_sync = [
            "processed_gesture.csv",
            "2d_pointing_trace.png",
            "skeleton_2d.json",
            "detection_summary.txt",
            "pointing_hand.json",
            "dog_detection_results.json",
            "baby_detection_results.json",
            "fig"  # entire folder
        ]

        print(f"\n📂 Syncing results from trial_output to original path...")
        print(f"   Source: {source_path}")
        print(f"   Destination: {destination_path}")

        copied_count = 0
        # Copy each item
        for item_name in items_to_sync:
            src = source_path / item_name
            dst = destination_path / item_name

            if src.exists():
                try:
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                    print(f"   ✓ Copied {item_name}")
                    copied_count += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to copy {item_name}: {e}")

        print(f"✅ Synced {copied_count} items to: {destination_path}")

    def update_display(self):
        """Update canvas display."""
        if self.current_color is None:
            return

        # Choose image to display
        if self.show_depth_var.get() and self.current_depth is not None:
            # Show depth as colored image
            depth_colored = self.colorize_depth(self.current_depth)
            display_img = depth_colored.copy()
        else:
            display_img = self.current_color.copy()

        # Draw human skeleton if available (green)
        if self.current_human_result and self.show_skeleton_var.get():
            display_img = self.draw_skeleton(display_img, self.current_human_result)

        # Draw human arm vectors if available
        if self.current_human_result and self.show_vectors_var.get():
            display_img = self.draw_arm_vectors(display_img, self.current_human_result)

        # Draw dog detection if available (blue)
        if self.current_dog_result and self.show_skeleton_var.get():
            display_img = self.draw_dog_detection(display_img, self.current_dog_result)

        # Draw baby skeleton if available (yellow)
        if self.current_baby_result and self.show_skeleton_var.get():
            display_img = self.draw_baby_skeleton(display_img, self.current_baby_result)

        # Display on canvas
        self.display_image(display_img)

        # Update 3D plot
        self.update_3d_plot()

    def update_3d_plot(self):
        """Update 3D skeleton plot."""
        if not self.plot_ax or not self.plot_canvas:
            return

        # Clear previous plot
        self.plot_ax.clear()
        self.plot_ax.set_xlabel('X (m)')
        self.plot_ax.set_ylabel('Y (m)')
        self.plot_ax.set_zlabel('Z (m)')
        self.plot_ax.set_title(f'3D Skeleton - Frame {self.current_frame}')

        # Check if we have any 3D data
        has_data = False

        # Plot human skeleton if available
        if self.current_human_result and self.current_human_result.landmarks_3d:
            has_data = True
            landmarks_3d = self.current_human_result.landmarks_3d
            arm_vectors = self.current_human_result.arm_vectors
            targets_to_plot = self.targets

            # Plot using existing visualization function
            from step2_skeleton_extraction.visualize_skeleton_3d import plot_skeleton_3d
            from step2_skeleton_extraction.pointing_analysis import compute_head_orientation

            # Compute head orientation for current frame
            head_orientation = None
            try:
                head_vec, head_origin = compute_head_orientation(landmarks_3d)
                head_orientation = {
                    'head_orientation_vector': head_vec.tolist(),
                    'head_orientation_origin': head_origin.tolist()
                }
            except Exception as e:
                print(f"Could not compute head orientation: {e}")

            try:
                plot_skeleton_3d(
                    landmarks_3d,
                    arm_vectors=arm_vectors,
                    frame_name=f"Frame {self.current_frame}",
                    targets=targets_to_plot,
                    show=False,
                    ax=self.plot_ax,
                    head_orientation=head_orientation
                )
            except Exception as e:
                print(f"\n❌ ERROR in 3D visualization:")
                print(f"   {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        # Plot dog as triangle with facing arrow (simplified visualization)
        if self.current_dog_result and self.current_dog_result.keypoints_3d:
            has_data = True
            self._plot_dog_simplified_3d(self.current_dog_result.keypoints_3d)

        # Plot baby simplified if available (yellow triangle + arrow)
        if self.current_baby_result and self.current_baby_result.keypoints_3d:
            has_data = True
            self._plot_baby_simplified_3d(self.current_baby_result.keypoints_3d)

        if not has_data:
            self.plot_ax.text(0, 0, 0, 'No 3D data', fontsize=14, ha='center')

        self.plot_canvas.draw()

    def _plot_subject_skeleton_3d(self, keypoints_3d, color='blue', label='Subject'):
        """Plot dog/baby skeleton in 3D."""
        # MediaPipe connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)  # Face
        ]

        # Draw connections
        for connection in connections:
            idx1, idx2 = connection
            if idx1 < len(keypoints_3d) and idx2 < len(keypoints_3d):
                kp1 = keypoints_3d[idx1]
                kp2 = keypoints_3d[idx2]

                # Check if both keypoints are valid (not [0,0,0])
                if (kp1[0] != 0 or kp1[1] != 0 or kp1[2] != 0) and \
                   (kp2[0] != 0 or kp2[1] != 0 or kp2[2] != 0):
                    self.plot_ax.plot(
                        [kp1[0], kp2[0]],
                        [kp1[1], kp2[1]],
                        [kp1[2], kp2[2]],
                        color=color, linewidth=2
                    )

        # Draw keypoints
        valid_kps = [kp for kp in keypoints_3d if (kp[0] != 0 or kp[1] != 0 or kp[2] != 0)]
        if valid_kps:
            xs = [kp[0] for kp in valid_kps]
            ys = [kp[1] for kp in valid_kps]
            zs = [kp[2] for kp in valid_kps]
            self.plot_ax.scatter(xs, ys, zs, c=color, marker='o', s=50, label=label)

    def _plot_dog_simplified_3d(self, keypoints_3d):
        """
        Plot dog as a triangle at centroid with directional arrow showing facing direction.

        Args:
            keypoints_3d: List of 33 3D keypoints [[x, y, z], ...]
        """
        # MediaPipe keypoint indices
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24

        # Get key points for computing centroid and direction
        nose = np.array(keypoints_3d[NOSE]) if NOSE < len(keypoints_3d) else None
        left_shoulder = np.array(keypoints_3d[LEFT_SHOULDER]) if LEFT_SHOULDER < len(keypoints_3d) else None
        right_shoulder = np.array(keypoints_3d[RIGHT_SHOULDER]) if RIGHT_SHOULDER < len(keypoints_3d) else None
        left_hip = np.array(keypoints_3d[LEFT_HIP]) if LEFT_HIP < len(keypoints_3d) else None
        right_hip = np.array(keypoints_3d[RIGHT_HIP]) if RIGHT_HIP < len(keypoints_3d) else None

        # Filter valid keypoints (not [0, 0, 0])
        def is_valid(kp):
            return kp is not None and not (kp[0] == 0 and kp[1] == 0 and kp[2] == 0)

        valid_points = []
        if is_valid(nose):
            valid_points.append(nose)
        if is_valid(left_shoulder):
            valid_points.append(left_shoulder)
        if is_valid(right_shoulder):
            valid_points.append(right_shoulder)
        if is_valid(left_hip):
            valid_points.append(left_hip)
        if is_valid(right_hip):
            valid_points.append(right_hip)

        if len(valid_points) < 2:
            # Not enough valid points
            return

        # Compute centroid (center of torso)
        centroid = np.mean(valid_points, axis=0)

        # Compute facing direction: from hip to shoulder (direction vector)
        if (is_valid(left_shoulder) or is_valid(right_shoulder)) and \
           (is_valid(left_hip) or is_valid(right_hip)):
            # Compute hip center
            hip_points = []
            if is_valid(left_hip):
                hip_points.append(left_hip)
            if is_valid(right_hip):
                hip_points.append(right_hip)
            hip_center = np.mean(hip_points, axis=0)

            # Compute shoulder center
            shoulder_points = []
            if is_valid(left_shoulder):
                shoulder_points.append(left_shoulder)
            if is_valid(right_shoulder):
                shoulder_points.append(right_shoulder)
            shoulder_center = np.mean(shoulder_points, axis=0)

            # Direction vector from hip to shoulder
            direction = shoulder_center - hip_center
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0.01:  # Only if significant distance
                direction = direction / direction_norm  # Normalize
            else:
                direction = None
        else:
            direction = None

        # Draw triangle at centroid (top view, like a marker)
        triangle_size = 0.15  # meters
        triangle_color = 'blue'

        # Create triangle vertices (in XY plane, pointing in Z direction if no head)
        if direction is not None:
            # Orient triangle towards facing direction
            # Project direction onto XY plane
            forward = np.array([direction[0], direction[1], 0])
            forward_norm = np.linalg.norm(forward)
            if forward_norm > 0.01:
                forward = forward / forward_norm
            else:
                forward = np.array([0, 0, 1])  # Default forward

            # Perpendicular vector
            right = np.array([-forward[1], forward[0], 0])

            # Triangle vertices: front tip, left back, right back
            tip = centroid + forward * triangle_size
            left_back = centroid - forward * triangle_size * 0.5 + right * triangle_size * 0.5
            right_back = centroid - forward * triangle_size * 0.5 - right * triangle_size * 0.5
        else:
            # Default triangle orientation
            tip = centroid + np.array([0, 0, triangle_size])
            left_back = centroid + np.array([-triangle_size * 0.5, 0, -triangle_size * 0.5])
            right_back = centroid + np.array([triangle_size * 0.5, 0, -triangle_size * 0.5])

        # Draw filled triangle
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        triangle_verts = [np.array([tip, left_back, right_back, tip])]
        triangle_poly = Poly3DCollection(triangle_verts, alpha=0.6, facecolor=triangle_color, edgecolor='darkblue', linewidths=2)
        self.plot_ax.add_collection3d(triangle_poly)

        # Draw centroid point (large square block for subject)
        self.plot_ax.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                           c='darkblue', marker='s', s=500, edgecolors='black', linewidths=3,
                           label='Dog (Subject)', zorder=10)

        # Draw facing arrow if direction is valid (from hip to shoulder)
        if direction is not None:
            # Compute actual hip and shoulder centers for arrow
            hip_points = []
            if is_valid(left_hip):
                hip_points.append(left_hip)
            if is_valid(right_hip):
                hip_points.append(right_hip)
            hip_center = np.mean(hip_points, axis=0) if hip_points else centroid

            shoulder_points = []
            if is_valid(left_shoulder):
                shoulder_points.append(left_shoulder)
            if is_valid(right_shoulder):
                shoulder_points.append(right_shoulder)
            shoulder_center = np.mean(shoulder_points, axis=0) if shoulder_points else centroid

            # Arrow length is the distance from hip to shoulder
            arrow_vec = shoulder_center - hip_center
            arrow_length = np.linalg.norm(arrow_vec)

            # Draw arrow from hip to shoulder (endpoint at shoulder)
            self.plot_ax.quiver(
                hip_center[0], hip_center[1], hip_center[2],
                arrow_vec[0], arrow_vec[1], arrow_vec[2],
                color='red', arrow_length_ratio=0.2, linewidth=2.5, label='Dog facing'
            )

            # Add text label at shoulder (endpoint)
            label_pos = shoulder_center + direction * 0.1  # Slightly beyond shoulder
            self.plot_ax.text(label_pos[0], label_pos[1], label_pos[2],
                            'HEAD', fontsize=9, color='red', weight='bold')

    def _plot_baby_simplified_3d(self, keypoints_3d):
        """
        Plot baby as a triangle at centroid with directional arrow showing facing direction.

        Args:
            keypoints_3d: List of 33 3D keypoints [[x, y, z], ...]
        """
        # MediaPipe keypoint indices
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24

        # Get key points for computing centroid and direction
        nose = np.array(keypoints_3d[NOSE]) if NOSE < len(keypoints_3d) else None
        left_shoulder = np.array(keypoints_3d[LEFT_SHOULDER]) if LEFT_SHOULDER < len(keypoints_3d) else None
        right_shoulder = np.array(keypoints_3d[RIGHT_SHOULDER]) if RIGHT_SHOULDER < len(keypoints_3d) else None
        left_hip = np.array(keypoints_3d[LEFT_HIP]) if LEFT_HIP < len(keypoints_3d) else None
        right_hip = np.array(keypoints_3d[RIGHT_HIP]) if RIGHT_HIP < len(keypoints_3d) else None

        # Filter valid keypoints (not [0, 0, 0])
        def is_valid(kp):
            return kp is not None and not (kp[0] == 0 and kp[1] == 0 and kp[2] == 0)

        valid_points = []
        if is_valid(nose):
            valid_points.append(nose)
        if is_valid(left_shoulder):
            valid_points.append(left_shoulder)
        if is_valid(right_shoulder):
            valid_points.append(right_shoulder)
        if is_valid(left_hip):
            valid_points.append(left_hip)
        if is_valid(right_hip):
            valid_points.append(right_hip)

        if len(valid_points) < 2:
            # Not enough valid points
            return

        # Compute centroid (center of torso)
        centroid = np.mean(valid_points, axis=0)

        # Compute facing direction: from hip to shoulder (direction vector)
        if (is_valid(left_shoulder) or is_valid(right_shoulder)) and \
           (is_valid(left_hip) or is_valid(right_hip)):
            # Compute hip center
            hip_points = []
            if is_valid(left_hip):
                hip_points.append(left_hip)
            if is_valid(right_hip):
                hip_points.append(right_hip)
            hip_center = np.mean(hip_points, axis=0)

            # Compute shoulder center
            shoulder_points = []
            if is_valid(left_shoulder):
                shoulder_points.append(left_shoulder)
            if is_valid(right_shoulder):
                shoulder_points.append(right_shoulder)
            shoulder_center = np.mean(shoulder_points, axis=0)

            # Direction vector from hip to shoulder
            direction = shoulder_center - hip_center
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0.01:  # Only if significant distance
                direction = direction / direction_norm  # Normalize
            else:
                direction = None
        else:
            direction = None

        # Draw triangle at centroid (top view, like a marker)
        triangle_size = 0.15  # meters
        triangle_color = 'yellow'

        # Create triangle vertices (in XY plane, pointing in Z direction if no head)
        if direction is not None:
            # Orient triangle towards facing direction
            # Project direction onto XY plane
            forward = np.array([direction[0], direction[1], 0])
            forward_norm = np.linalg.norm(forward)
            if forward_norm > 0.01:
                forward = forward / forward_norm
            else:
                forward = np.array([0, 0, 1])  # Default forward

            # Perpendicular vector
            right = np.array([-forward[1], forward[0], 0])

            # Triangle vertices: front tip, left back, right back
            tip = centroid + forward * triangle_size
            left_back = centroid - forward * triangle_size * 0.5 + right * triangle_size * 0.5
            right_back = centroid - forward * triangle_size * 0.5 - right * triangle_size * 0.5
        else:
            # Default triangle orientation
            tip = centroid + np.array([0, 0, triangle_size])
            left_back = centroid + np.array([-triangle_size * 0.5, 0, -triangle_size * 0.5])
            right_back = centroid + np.array([triangle_size * 0.5, 0, -triangle_size * 0.5])

        # Draw filled triangle
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        triangle_verts = [np.array([tip, left_back, right_back, tip])]
        triangle_poly = Poly3DCollection(triangle_verts, alpha=0.6, facecolor=triangle_color, edgecolor='orange', linewidths=2)
        self.plot_ax.add_collection3d(triangle_poly)

        # Draw centroid point (large square block for subject)
        self.plot_ax.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                           c='orange', marker='s', s=500, edgecolors='black', linewidths=3,
                           label='Baby (Subject)', zorder=10)

        # Draw facing arrow if direction is valid (from hip to shoulder)
        if direction is not None:
            # Compute actual hip and shoulder centers for arrow
            hip_points = []
            if is_valid(left_hip):
                hip_points.append(left_hip)
            if is_valid(right_hip):
                hip_points.append(right_hip)
            hip_center = np.mean(hip_points, axis=0) if hip_points else centroid

            shoulder_points = []
            if is_valid(left_shoulder):
                shoulder_points.append(left_shoulder)
            if is_valid(right_shoulder):
                shoulder_points.append(right_shoulder)
            shoulder_center = np.mean(shoulder_points, axis=0) if shoulder_points else centroid

            # Arrow length is the distance from hip to shoulder
            arrow_vec = shoulder_center - hip_center
            arrow_length = np.linalg.norm(arrow_vec)

            # Draw arrow from hip to shoulder (endpoint at shoulder)
            self.plot_ax.quiver(
                hip_center[0], hip_center[1], hip_center[2],
                arrow_vec[0], arrow_vec[1], arrow_vec[2],
                color='purple', arrow_length_ratio=0.2, linewidth=2.5, label='Baby facing'
            )

            # Add text label at shoulder (endpoint)
            label_pos = shoulder_center + direction * 0.1  # Slightly beyond shoulder
            self.plot_ax.text(label_pos[0], label_pos[1], label_pos[2],
                            'HEAD', fontsize=9, color='purple', weight='bold')

    def draw_skeleton(self, image, result):
        """Draw skeleton on image."""
        img = image.copy()
        h, w = img.shape[:2]

        # MediaPipe connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
        ]

        landmarks = result.landmarks_2d

        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                if start[2] > 0.5 and end[2] > 0.5:  # Check visibility
                    pt1 = (int(start[0]), int(start[1]))
                    pt2 = (int(end[0]), int(end[1]))
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks
        for x, y, visibility in landmarks:
            if visibility > 0.5:
                cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), -1)

        # Draw hip center if available
        if result.metadata and result.metadata.get('hip_center_2d'):
            hip_2d = result.metadata['hip_center_2d']
            hip_pt = (int(hip_2d[0]), int(hip_2d[1]))
            # Draw larger yellow circle for hip center
            cv2.circle(img, hip_pt, 8, (0, 255, 255), 2)  # Yellow outline
            cv2.circle(img, hip_pt, 3, (0, 255, 255), -1)  # Yellow center
            # Add label
            cv2.putText(img, "Hip Center", (hip_pt[0] + 10, hip_pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return img

    def draw_arm_vectors(self, image, result):
        """Draw arm vectors on image."""
        if not result.arm_vectors:
            return image

        img = image.copy()
        landmarks = result.landmarks_2d
        h, w = img.shape[:2]

        # Get landmark indices based on pointing arm
        # Use manual selection if not auto, otherwise use detection
        selected_arm = self.arm_selection_var.get()
        if selected_arm != "auto":
            pointing_arm = selected_arm
        else:
            pointing_arm = result.metadata.get('pointing_arm', 'right')

        if pointing_arm == 'left':
            shoulder_idx = 11  # Left shoulder
            elbow_idx = 13     # Left elbow
            wrist_idx = 15     # Left wrist
            eye_idx = 1        # Left eye
        else:  # right or auto
            shoulder_idx = 12  # Right shoulder
            elbow_idx = 14     # Right elbow
            wrist_idx = 16     # Right wrist
            eye_idx = 4        # Right eye

        nose_idx = 0

        # Get wrist position in 2D
        if wrist_idx < len(landmarks):
            wrist_2d = landmarks[wrist_idx]
            wrist_pt = (int(wrist_2d[0]), int(wrist_2d[1]))

            # Draw extended vectors FROM wrist
            vectors = result.arm_vectors
            vector_configs = [
                ('shoulder_to_wrist', (0, 255, 0), shoulder_idx),      # Green
                ('elbow_to_wrist', (255, 165, 0), elbow_idx),          # Orange
                ('eye_to_wrist', (255, 0, 255), eye_idx),              # Magenta
                ('nose_to_wrist', (0, 255, 255), nose_idx)             # Cyan
            ]

            for vec_name, color, origin_idx in vector_configs:
                if vec_name in vectors and vectors[vec_name] and origin_idx < len(landmarks):
                    # Get origin point (shoulder/elbow/eye/nose)
                    origin_2d = landmarks[origin_idx]
                    origin_pt = (int(origin_2d[0]), int(origin_2d[1]))

                    # 1. Draw dotted reference line from origin to wrist
                    self.draw_dotted_line(img, origin_pt, wrist_pt, color, thickness=2)

                    # 2. Calculate direction in 2D based on origin->wrist
                    dx = wrist_pt[0] - origin_pt[0]
                    dy = wrist_pt[1] - origin_pt[1]
                    length = np.sqrt(dx*dx + dy*dy)

                    if length > 0:
                        # Normalize direction
                        dx_norm = dx / length
                        dy_norm = dy / length

                        # Extended endpoint FROM wrist in pointing direction
                        scale = 500  # pixels - extend far to see intersection
                        end_pt = (
                            int(wrist_pt[0] + dx_norm * scale),
                            int(wrist_pt[1] + dy_norm * scale)
                        )

                        # Clamp to image bounds
                        end_pt = (
                            max(0, min(w-1, end_pt[0])),
                            max(0, min(h-1, end_pt[1]))
                        )

                        # 3. Draw extended vector arrow FROM wrist outward
                        cv2.arrowedLine(img, wrist_pt, end_pt, color, 3, tipLength=0.1)

        return img

    def draw_dog_detection(self, image, result):
        """Draw dog skeleton on image (blue)."""
        img = image.copy()

        # Draw skeleton keypoints if available
        if result.keypoints_2d:
            # MediaPipe connections (same as human/baby)
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                (11, 23), (12, 24), (23, 24),  # Torso
                (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
                (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)  # Face
            ]

            # Draw connections in blue
            for connection in connections:
                idx1, idx2 = connection
                if idx1 < len(result.keypoints_2d) and idx2 < len(result.keypoints_2d):
                    kp1 = result.keypoints_2d[idx1]
                    kp2 = result.keypoints_2d[idx2]

                    if len(kp1) >= 3 and len(kp2) >= 3:
                        x1, y1, conf1 = kp1[:3]
                        x2, y2, conf2 = kp2[:3]

                        if conf1 > 0.5 and conf2 > 0.5:
                            pt1 = (int(x1), int(y1))
                            pt2 = (int(x2), int(y2))
                            cv2.line(img, pt1, pt2, (255, 0, 0), 2)  # Blue lines

            # Draw keypoints in blue
            for kp in result.keypoints_2d:
                if len(kp) >= 3:
                    x, y, conf = kp[:3]
                    if conf > 0.5:
                        cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), -1)  # Blue dots

            # Draw label
            cv2.putText(img, "Dog", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw bounding box if available
        elif result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(img, "Dog", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return img

    def draw_baby_skeleton(self, image, result):
        """Draw baby skeleton on image (yellow)."""
        img = image.copy()

        if not result.keypoints_2d:
            return img

        # MediaPipe connections (same as human)
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]

        # Draw connections in yellow
        for connection in connections:
            idx1, idx2 = connection
            if idx1 < len(result.keypoints_2d) and idx2 < len(result.keypoints_2d):
                kp1 = result.keypoints_2d[idx1]
                kp2 = result.keypoints_2d[idx2]

                if len(kp1) >= 3 and len(kp2) >= 3:
                    x1, y1, conf1 = kp1[:3]
                    x2, y2, conf2 = kp2[:3]

                    if conf1 > 0.5 and conf2 > 0.5:
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                (0, 255, 255), 2)  # Yellow

        # Draw keypoints in yellow
        for kp in result.keypoints_2d:
            if len(kp) >= 3:
                x, y, conf = kp[:3]
                if conf > 0.5:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)  # Yellow

        # Draw label
        cv2.putText(img, "Baby", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return img

    def draw_dotted_line(self, img, pt1, pt2, color, thickness=2, gap=10):
        """Draw a dotted line between two points."""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        pts = []

        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int(pt1[0] * (1 - r) + pt2[0] * r)
            y = int(pt1[1] * (1 - r) + pt2[1] * r)
            pts.append((x, y))

        # Draw dots
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)

    def colorize_depth(self, depth):
        """Convert depth to colored visualization."""
        # Normalize depth
        valid_depth = depth[depth > 0]
        if len(valid_depth) == 0:
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        min_depth = valid_depth.min()
        max_depth = valid_depth.max()

        normalized = np.zeros_like(depth)
        mask = depth > 0
        normalized[mask] = (depth[mask] - min_depth) / (max_depth - min_depth)

        # Apply colormap
        colored = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(colored, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        colored[~mask] = 0

        return colored

    def display_image(self, image):
        """Display image on canvas."""
        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        img_h, img_w = image.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Convert to PhotoImage
        img_pil = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image=img_pil)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=photo, anchor=tk.CENTER)
        self.canvas.image = photo  # Keep reference

    def update_info(self):
        """Update info text."""
        self.info_text.delete(1.0, tk.END)

        info = f"Frame {self.current_frame}\n"
        info += f"{'='*30}\n\n"

        # Show human detection info
        if self.current_human_result:
            result = self.current_human_result
            info += f"👤 HUMAN DETECTED\n"
            info += f"Landmarks detected: {len(result.landmarks_2d)}\n"
            info += f"Pointing arm: {result.metadata.get('pointing_arm', 'unknown')}\n"
            info += f"Has depth: {result.metadata.get('has_depth', False)}\n"

            # Display wrist motion info if available
            if self.detector and hasattr(self.detector, 'landmark_history'):
                if len(self.detector.landmark_history) >= 2:
                    motion = self.detector._calculate_wrist_motion(result.landmarks_2d)
                    info += f"\n🏃 Wrist Motion:\n"
                    info += f"  Left:  {motion['left']:.2f} px/frame\n"
                    info += f"  Right: {motion['right']:.2f} px/frame\n"

                    # Show which is moving more
                    if motion['left'] > motion['right'] * 1.2:
                        info += f"  → Left wrist moving MORE 📍\n"
                    elif motion['right'] > motion['left'] * 1.2:
                        info += f"  → Right wrist moving MORE 📍\n"
                    else:
                        info += f"  → Similar motion\n"
                else:
                    info += f"\n🏃 Wrist Motion: Collecting...\n"

            # Display hip center info if available
            if result.metadata.get('hip_center_2d'):
                hip_2d = result.metadata['hip_center_2d']
                info += f"\n📍 Hip Center 2D:\n"
                info += f"  X: {hip_2d[0]:.1f} px\n"
                info += f"  Y: {hip_2d[1]:.1f} px\n"

            if result.metadata.get('hip_center_3d'):
                hip_3d = result.metadata['hip_center_3d']
                info += f"\n📍 Hip Center 3D:\n"
                info += f"  X: {hip_3d[0]:.3f} m\n"
                info += f"  Y: {hip_3d[1]:.3f} m\n"
                info += f"  Z: {hip_3d[2]:.3f} m\n"

            info += f"\n"

            if result.landmarks_3d:
                info += f"3D Landmarks: ✓\n"
            else:
                info += f"3D Landmarks: ✗\n"

            if result.arm_vectors:
                info += f"\nArm Vectors:\n"
                for vec_name, vec in result.arm_vectors.items():
                    if vec and vec_name != 'wrist_location':
                        info += f"  {vec_name}:\n"
                        info += f"    [{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]\n"

                if 'wrist_location' in result.arm_vectors and result.arm_vectors['wrist_location']:
                    loc = result.arm_vectors['wrist_location']
                    info += f"\nWrist 3D Location:\n"
                    info += f"  X: {loc[0]:.3f} m\n"
                    info += f"  Y: {loc[1]:.3f} m\n"
                    info += f"  Z: {loc[2]:.3f} m\n"

        # Show dog detection info
        if self.current_dog_result:
            info += f"\n🐶 DOG DETECTED\n"
            if self.current_dog_result.bbox:
                bbox = self.current_dog_result.bbox
                info += f"Bbox: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})\n"

        # Show baby detection info
        if self.current_baby_result:
            info += f"\n👶 BABY DETECTED\n"
            if self.current_baby_result.keypoints_2d:
                info += f"Keypoints: {len(self.current_baby_result.keypoints_2d)}\n"

        # If nothing detected
        if not self.current_human_result and not self.current_dog_result and not self.current_baby_result:
            info += "No detection for current frame\n\n"
            info += "Process this frame to detect subjects"

        self.info_text.insert(1.0, info)

    def on_frame_change(self, *args):
        """Handle frame slider change."""
        frame_num = self.frame_var.get()
        if frame_num != self.current_frame:
            self.load_frame(int(frame_num))

    def on_canvas_resize(self, event):
        """Handle canvas resize."""
        self.update_display()

    def first_frame(self):
        """Go to first frame."""
        self.load_frame(1)

    def prev_frame(self):
        """Go to previous frame."""
        self.load_frame(max(1, self.current_frame - 1))

    def next_frame(self):
        """Go to next frame."""
        self.load_frame(min(self.total_frames, self.current_frame + 1))

    def last_frame(self):
        """Go to last frame."""
        self.load_frame(self.total_frames)

    def toggle_play(self):
        """Toggle playback."""
        self.playing = not self.playing
        if self.playing:
            self.play_button.config(text="⏸")
            self.play_frames()
        else:
            self.play_button.config(text="▶")

    def play_frames(self):
        """Play frames automatically."""
        if not self.playing:
            return

        if self.current_frame < self.total_frames:
            self.next_frame()
            self.root.after(100, self.play_frames)  # 10 fps
        else:
            self.playing = False
            self.play_button.config(text="▶")


def main():
    root = tk.Tk()
    app = SkeletonExtractorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
