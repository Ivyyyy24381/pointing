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

    def __init__(self, root):
        self.root = root
        self.root.title("Step 2: Skeleton Extraction")
        self.root.geometry("1920x1000")  # Wider for 3-panel layout

        # State variables
        self.trial_path = None
        self.camera_id = None
        self.current_frame = 1
        self.total_frames = 0
        self.color_images = []
        self.detector = None
        self.results = {}
        self.playing = False

        # Camera intrinsics (auto-detected)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        # Current frame data
        self.current_color = None
        self.current_depth = None
        self.current_result = None

        # 3D plot components
        self.plot_canvas = None
        self.plot_fig = None
        self.plot_ax = None

        # Ground plane transform (loaded once)
        self.ground_plane_transform = None
        self.targets = None

        self.setup_ui()
        self.initialize_detector()

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
        load_frame = ttk.LabelFrame(parent, text="Load Trial", padding=10)
        load_frame.grid(row=0, column=0, sticky="ew", pady=5)

        ttk.Button(load_frame, text="📁 Select Trial Folder",
                  command=self.load_trial).pack(fill=tk.X, pady=2)

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

        ttk.Label(settings_frame, text="Subject Detection:").grid(row=9, column=0, sticky="w")
        # Human is always on by default, dog and baby are optional
        self.detect_human_var = tk.BooleanVar(value=True)
        self.detect_dog_var = tk.BooleanVar(value=False)
        self.detect_baby_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(settings_frame, text="👤 Human (default)",
                       variable=self.detect_human_var,
                       state="disabled").grid(row=10, column=0, sticky="w")
        ttk.Checkbutton(settings_frame, text="🐶 Dog (optional)",
                       variable=self.detect_dog_var,
                       command=self.on_subject_type_change).grid(row=11, column=0, sticky="w")
        ttk.Checkbutton(settings_frame, text="👶 Baby (optional)",
                       variable=self.detect_baby_var,
                       command=self.on_subject_type_change).grid(row=12, column=0, sticky="w")

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
        """Initialize MediaPipe detector."""
        try:
            complexity = self.complexity_var.get()
            confidence = self.conf_var.get()

            self.detector = MediaPipeHumanDetector(
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence,
                model_complexity=complexity
            )
            self.status_label.config(text="Detector initialized", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize detector:\n{e}")
            self.status_label.config(text="Detector error", foreground="red")

    def reinitialize_detector(self, *args):
        """Reinitialize detector with new settings."""
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")
        self.initialize_detector()
        if self.current_color is not None:
            self.process_current_frame()

    def on_arm_selection_change(self):
        """Handle manual arm selection change - applies to ALL loaded results."""
        selected_arm = self.arm_selection_var.get()
        print(f"\n🔄 Arm selection changed to: {selected_arm}")

        if selected_arm != "auto":
            # Update ALL loaded results with the selected arm
            if self.results:
                print(f"   Updating {len(self.results)} loaded frames with {selected_arm} arm...")
                for frame_key, result in self.results.items():
                    if result.landmarks_3d:
                        result.metadata['pointing_arm'] = selected_arm
                        result.arm_vectors = self.detector._compute_arm_vectors(
                            result.landmarks_3d,
                            selected_arm
                        )
                print(f"   ✓ All {len(self.results)} results updated")

            # Update current result if available
            if self.current_result and self.current_result.landmarks_3d:
                self.current_result.metadata['pointing_arm'] = selected_arm
                self.current_result.arm_vectors = self.detector._compute_arm_vectors(
                    self.current_result.landmarks_3d,
                    selected_arm
                )
                print(f"   ✓ Current frame updated")

            # Update display
            self.update_display()
            self.update_info()
            print(f"   ✓ Display updated\n")

        else:
            # Auto mode - reprocess current frame
            print(f"   Auto mode - reprocessing current frame...")
            if self.current_color is not None:
                self.process_current_frame()
            else:
                print(f"   ⚠️  No current frame to reprocess")

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

        print(f"\n🔄 Subject detection updated:")
        print(f"   Enabled: {', '.join(enabled_types) if enabled_types else 'None'}")

        # Enable/disable segmentation button based on dog/baby selection
        if detect_dog or detect_baby:
            self.segment_button.config(state="normal")
            print(f"   ✓ Segmentation available (required for dog/baby detection)")
        else:
            self.segment_button.config(state="disabled")

        # Show info for newly enabled types
        if detect_dog:
            messagebox.showinfo("Dog Detection",
                              "🐶 Dog detection enabled!\n\n"
                              "Requirements:\n"
                              "• Click 'Run Segmentation' to isolate the dog\n"
                              "• DeepLabCut with SuperAnimal quadruped model\n\n"
                              "Note: Segmentation focuses on lower half of image")
            print(f"   ✓ Dog detection enabled (requires segmentation)")

        if detect_baby:
            messagebox.showinfo("Baby Detection",
                              "👶 Baby detection enabled!\n\n"
                              "Requirements:\n"
                              "• Click 'Run Segmentation' to isolate the baby\n"
                              "• Uses MediaPipe Pose for skeleton detection\n\n"
                              "Note: Segmentation focuses on lower half of image")
            print(f"   ✓ Baby detection enabled (requires segmentation)")

        # Note: Human detection is always on (checkbox is disabled)

    def run_segmentation(self):
        """Run SAM2 video segmentation for dog/baby detection."""
        if not self.trial_path:
            messagebox.showerror("Error", "No trial loaded. Please load a trial first.")
            return

        # Check if we're already showing a segmented video
        masked_video_path = self.trial_path.parent.parent / "single_camera" / "masked_video.mp4"
        if masked_video_path.exists():
            response = messagebox.askyesno(
                "Segmentation Exists",
                "A masked video already exists. Run segmentation again?\n\n"
                "This will overwrite the existing segmented video."
            )
            if not response:
                return

        print("\n" + "="*60)
        print("🎭 STARTING SAM2 SEGMENTATION")
        print("="*60)

        try:
            # Import segmentation wrapper
            from step2_skeleton_extraction.segmenter_wrapper import run_segmentation_interactive

            # Disable UI during segmentation
            self.segment_button.config(state="disabled", text="⏳ Segmenting...")
            self.root.update()

            # Run segmentation (opens matplotlib window for point selection)
            def progress_callback(message, progress):
                """Update UI with progress."""
                self.segment_button.config(text=f"⏳ {message}")
                self.root.update()

            result_path = run_segmentation_interactive(
                str(self.trial_path),
                progress_callback=progress_callback
            )

            if result_path:
                messagebox.showinfo(
                    "Segmentation Complete",
                    f"✅ Segmentation successful!\n\n"
                    f"Masked video saved to:\n{result_path}\n\n"
                    f"You can now run skeleton detection on the segmented video."
                )
                print(f"✅ Segmentation complete: {result_path}")
            else:
                messagebox.showwarning(
                    "Segmentation Cancelled",
                    "Segmentation was cancelled or failed.\n"
                    "No masked video was created."
                )
                print("⚠️  Segmentation cancelled or failed")

        except Exception as e:
            messagebox.showerror(
                "Segmentation Error",
                f"Failed to run segmentation:\n\n{str(e)}\n\n"
                f"Check console for details."
            )
            print(f"❌ Segmentation error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Re-enable button
            detect_dog = self.detect_dog_var.get()
            detect_baby = self.detect_baby_var.get()
            if detect_dog or detect_baby:
                self.segment_button.config(state="normal", text="🎭 Run Segmentation")
            else:
                self.segment_button.config(state="disabled", text="🎭 Run Segmentation")

    def load_trial(self):
        """Load trial folder."""
        folder = filedialog.askdirectory(
            title="Select Trial Folder",
            initialdir="trial_input"
        )

        if not folder:
            return

        trial_path = Path(folder)

        # Check for color folder
        color_folder = trial_path / "color"
        if not color_folder.exists():
            messagebox.showerror("Error", "No 'color' folder found in selected directory")
            return

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
        print(f"🔍 Searching for targets in {len(possible_target_paths)} locations...")
        for i, path in enumerate(possible_target_paths, 1):
            print(f"   {i}. Checking: {path}")
            if path.exists():
                print(f"      ✅ Found!")
                with open(path) as f:
                    self.targets = json.load(f)
                print(f"✅ Loaded {len(self.targets)} targets from: {path}")
                break
            else:
                print(f"      ✗ Not found")

        if self.targets is None:
            print("⚠️  No target detections found. Run Page 1 to detect targets first.")

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
        if frame_key in self.results:
            self.current_result = self.results[frame_key]
        else:
            self.current_result = None

        self.update_display()
        self.update_info()

    def process_current_frame(self):
        """Process current frame."""
        if self.current_color is None:
            return

        self.status_label.config(text="Processing...", foreground="orange")
        self.root.update()

        try:
            # Get actual frame number from filename
            color_path = self.color_images[self.current_frame - 1]
            actual_frame_num = int(color_path.stem.split('_')[1])

            result = self.detector.detect_frame(
                self.current_color,
                actual_frame_num,  # Use actual frame number, not sequential index
                depth_image=self.current_depth,
                fx=self.fx, fy=self.fy,
                cx=self.cx, cy=self.cy
            )

            frame_key = f"frame_{actual_frame_num:06d}"  # Use actual frame number
            if result:
                self.results[frame_key] = result
                self.current_result = result
                self.status_label.config(text="Detection successful", foreground="green")
            else:
                self.current_result = None
                self.status_label.config(text="No pose detected", foreground="orange")

            self.update_display()
            self.update_info()

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")
            self.status_label.config(text="Processing error", foreground="red")

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

        self.results = {}
        self.progress_var.set(0)

        for i, color_path in enumerate(self.color_images, 1):
            # Load frame
            frame_num = int(color_path.stem.split('_')[-1])
            color_img = cv2.imread(str(color_path))
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

            # Process
            result = self.detector.detect_frame(
                color_rgb, frame_num,
                depth_image=depth_img,
                fx=self.fx, fy=self.fy,
                cx=self.cx, cy=self.cy
            )

            if result:
                frame_key = f"frame_{frame_num:06d}"
                self.results[frame_key] = result

                # Update visualization if this is the current frame being viewed
                if i == self.current_frame:
                    self.current_result = result
                    self.current_color = color_rgb
                    self.current_depth = depth_img
                    self.update_display()
                    self.update_info()

            # Update progress
            progress = (i / self.total_frames) * 100
            self.progress_var.set(progress)
            self.status_label.config(
                text=f"Processing {i}/{self.total_frames}...",
                foreground="orange"
            )
            self.root.update()

        # Determine pointing hand based on user selection or auto-detect
        selected_arm = self.arm_selection_var.get()

        if selected_arm == "auto":
            # Auto-detect from all results
            results_list = list(self.results.values())
            pointing_hand = determine_pointing_hand_whole_trial(results_list)
            print(f"\n{'='*60}")
            print(f"👆 AUTO-DETECTED POINTING HAND: {pointing_hand.upper()}")
            print(f"{'='*60}\n")
        else:
            # Use user-selected arm
            pointing_hand = selected_arm
            print(f"\n{'='*60}")
            print(f"👆 USER-SELECTED POINTING HAND: {pointing_hand.upper()}")
            print(f"{'='*60}\n")

        # Update all results with the determined/selected pointing hand
        for result in self.results.values():
            result.metadata['pointing_hand_whole_trial'] = pointing_hand
            # Recompute arm vectors for the selected/determined hand (only if 3D data available)
            if pointing_hand in ['left', 'right'] and result.landmarks_3d:
                result.arm_vectors = self.detector._compute_arm_vectors(
                    result.landmarks_3d,
                    pointing_hand
                )
                result.metadata['pointing_arm'] = pointing_hand

        self.status_label.config(
            text=f"Processed {len(self.results)}/{self.total_frames} frames",
            foreground="green"
        )

        # Auto-save results
        self.auto_save_results(pointing_hand)

        # Reload current frame to show results with updated arm vectors
        self.load_frame(self.current_frame)

        # Force update of 3D visualization with new arm selection
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

    def auto_save_results(self, pointing_hand: str):
        """Automatically save results after batch processing."""
        if not self.results or not self.trial_path:
            return

        # Determine output directory (consistent with load_ground_plane_and_targets)
        camera_name = self.trial_path.name  # e.g., "cam1" or "single_camera"
        trial_name = self.trial_path.parent.name  # e.g., "trial_1" or "1"

        # For single camera mode, if camera_name == trial_name, use parent's parent
        if camera_name == trial_name and self.trial_path.parent.parent.name == "trial_input":
            # Path is trial_input/single_camera, so just use single_camera once
            output_path = Path("trial_output") / camera_name
        else:
            # Normal case: trial_output/trial_name/camera_name
            output_path = Path("trial_output") / trial_name / camera_name

        output_path.mkdir(parents=True, exist_ok=True)

        # Convert results to dict format
        output_data = {}
        for frame_key, result in self.results.items():
            output_data[frame_key] = result.to_dict()

        # Save JSON
        json_file = output_path / "skeleton_2d.json"
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Save summary with whole-trial pointing hand
        summary_file = output_path / "skeleton_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Skeleton Extraction Summary\n")
            f.write(f"{'='*40}\n")
            f.write(f"Trial: {trial_name}\n")
            f.write(f"Camera: {self.trial_path.name}\n")
            f.write(f"Total frames processed: {len(self.results)}\n")
            f.write(f"Detector: MediaPipe Pose\n")
            f.write(f"Model complexity: {self.complexity_var.get()}\n")
            f.write(f"Detection confidence: {self.conf_var.get():.2f}\n")
            f.write(f"\n{'='*40}\n")
            f.write(f"POINTING HAND (whole trial): {pointing_hand.upper()}\n")
            f.write(f"{'='*40}\n")
            f.write(f"\nPer-frame pointing arm distribution:\n")

            # Count pointing arms
            pointing_arms = {}
            for result in self.results.values():
                arm = result.metadata.get('pointing_arm', 'unknown')
                pointing_arms[arm] = pointing_arms.get(arm, 0) + 1

            for arm, count in sorted(pointing_arms.items()):
                pct = (count / len(self.results)) * 100
                f.write(f"  {arm}: {count} frames ({pct:.1f}%)\n")

        # Save pointing hand to separate JSON
        pointing_hand_file = output_path / "pointing_hand.json"
        with open(pointing_hand_file, 'w') as f:
            json.dump({
                "trial": trial_name,
                "camera": self.trial_path.name,
                "pointing_hand": pointing_hand,
                "total_frames": len(self.results),
                "frame_distribution": pointing_arms
            }, f, indent=2)

        print(f"\n✅ Auto-saved results to: {output_path}")
        print(f"   - skeleton_2d.json")
        print(f"   - skeleton_summary.txt")
        print(f"   - pointing_hand.json")

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

        # Draw skeleton if available
        if self.current_result and self.show_skeleton_var.get():
            display_img = self.draw_skeleton(display_img, self.current_result)

        # Draw arm vectors if available
        if self.current_result and self.show_vectors_var.get():
            display_img = self.draw_arm_vectors(display_img, self.current_result)

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

        # Check if we have 3D data
        if not self.current_result or not self.current_result.landmarks_3d:
            self.plot_ax.text(0, 0, 0, 'No 3D data', fontsize=14, ha='center')
            self.plot_canvas.draw()
            return

        # Get skeleton data - use raw camera frame coordinates (no transformation)
        landmarks_3d = self.current_result.landmarks_3d
        arm_vectors = self.current_result.arm_vectors
        targets_to_plot = self.targets

        # Debug: Show we're NOT transforming
        print(f"\n   📊 3D PLOT DEBUG:")
        print(f"      Transformation: DISABLED (using raw camera coordinates)")
        if arm_vectors:
            pointing_arm = self.current_result.metadata.get('pointing_arm', 'unknown')
            print(f"      Arm: {pointing_arm}")

        # Debug target loading status
        print(f"      self.targets = {self.targets is not None}")
        if self.targets is not None:
            print(f"      Number of targets: {len(self.targets)}")

        # Show actual target positions
        if targets_to_plot:
            print(f"      Target positions:")
            for target in targets_to_plot:
                label = target.get('label', 'unknown')
                x, y, z = target.get('x', 0), target.get('y', 0), target.get('z', 0)
                print(f"         {label}: X={x:+.3f}, Y={y:+.3f}, Z={z:+.3f}")
        else:
            print(f"      ⚠️  NO TARGETS TO PLOT (targets_to_plot is None or empty)")
            print(f"      Check console for target loading messages above")

        # Plot using existing visualization function on our axis
        from step2_skeleton_extraction.visualize_skeleton_3d import plot_skeleton_3d

        try:
            plot_skeleton_3d(
                landmarks_3d,
                arm_vectors=arm_vectors,
                frame_name=f"Frame {self.current_frame}",
                targets=targets_to_plot,
                show=False,
                ax=self.plot_ax  # Pass our existing axis
            )

            # Redraw canvas
            self.plot_canvas.draw()
        except Exception as e:
            print(f"\n❌ ERROR in 3D visualization:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.plot_ax.text(0, 0, 0, f'Visualization Error:\n{str(e)}', fontsize=10, ha='center')
            self.plot_canvas.draw()

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

        if self.current_result:
            result = self.current_result
            info = f"Frame {self.current_frame}\n"
            info += f"{'='*30}\n\n"
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
        else:
            info = "No detection for current frame\n\n"
            info += "Process this frame to detect skeleton"

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
