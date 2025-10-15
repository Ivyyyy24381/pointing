"""
UI Part 1: Data Loading Interface

Simple GUI for browsing and loading trial data using DataManager.

Features:
- Browse folders containing multiple trials
- View trial metadata
- Select trial and camera
- Browse frames
- Display color and depth

Usage:
    python ui_data_loader.py [root_folder]

Example:
    python ui_data_loader.py sample_raw_data
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(__file__))
from data_manager import DataManager
from trial_input_manager import TrialInputManager
from target_detector import TargetDetector, get_default_model_path


class DataLoaderUI:
    """Simple GUI for loading trial data"""

    def __init__(self, root, initial_folder=None):
        self.root = root
        self.root.title("Data Loader - UI Part 1 (Auto-Standardizing)")
        self.root.geometry("1200x800")

        self.data_manager = None
        self.trial_input_manager = TrialInputManager("trial_input")
        self.current_trial = None
        self.current_camera = None
        self.current_frame = None
        self.available_frames = []
        self.trial_input_path = None  # Path to current trial in trial_input/

        # Frame trimming
        self.trim_start = None  # Start frame for trimming
        self.trim_end = None    # End frame for trimming

        # Target detection
        self.target_detector = None
        self.current_detections = []
        self.current_color = None
        self.current_depth = None
        self.detection_overlay = False

        # Create UI
        self.create_menu()
        self.create_control_panel()
        self.create_display_area()
        self.create_status_bar()

        # Load initial folder if provided
        if initial_folder and os.path.isdir(initial_folder):
            self.load_folder(initial_folder)

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder...", command=self.browse_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_command(label="Load Config", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

    def create_control_panel(self):
        """Create control panel"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Folder selection
        folder_frame = ttk.LabelFrame(control_frame, text="Data Folder", padding=10)
        folder_frame.pack(fill=tk.X, pady=5)

        self.folder_label = ttk.Label(folder_frame, text="No folder loaded", foreground="gray")
        self.folder_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(folder_frame, text="Browse...", command=self.browse_folder).pack(side=tk.RIGHT, padx=5)

        # Trial selection
        selection_frame = ttk.Frame(control_frame)
        selection_frame.pack(fill=tk.X, pady=5)

        # Trial dropdown
        trial_frame = ttk.Frame(selection_frame)
        trial_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        ttk.Label(trial_frame, text="Trial:").pack(side=tk.LEFT, padx=(0, 5))
        self.trial_var = tk.StringVar()
        self.trial_combo = ttk.Combobox(trial_frame, textvariable=self.trial_var, state="readonly", width=20)
        self.trial_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.trial_combo.bind("<<ComboboxSelected>>", self.on_trial_changed)

        # Camera dropdown
        camera_frame = ttk.Frame(selection_frame)
        camera_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        ttk.Label(camera_frame, text="Camera:").pack(side=tk.LEFT, padx=(0, 5))
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, state="readonly", width=15)
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_changed)

        # Frame slider
        frame_control_frame = ttk.LabelFrame(control_frame, text="Frame Selection", padding=10)
        frame_control_frame.pack(fill=tk.X, pady=5)

        self.frame_slider = tk.Scale(frame_control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     label="Frame", command=self.on_frame_changed, length=800)
        self.frame_slider.pack(fill=tk.X)

        # Navigation buttons
        nav_frame = ttk.Frame(frame_control_frame)
        nav_frame.pack(fill=tk.X, pady=5)

        ttk.Button(nav_frame, text="◀◀ -10", command=lambda: self.jump_frames(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="◀ Prev", command=lambda: self.jump_frames(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=lambda: self.jump_frames(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="+10 ▶▶", command=lambda: self.jump_frames(10)).pack(side=tk.LEFT, padx=2)

        # Load button
        ttk.Button(nav_frame, text="🔄 Reload Frame", command=self.load_current_frame).pack(side=tk.RIGHT, padx=10)

        # Frame trimming controls
        trim_frame = ttk.LabelFrame(control_frame, text="✂️ Frame Trimming", padding=10)
        trim_frame.pack(fill=tk.X, pady=5)

        trim_buttons = ttk.Frame(trim_frame)
        trim_buttons.pack(fill=tk.X, pady=2)

        ttk.Button(trim_buttons, text="📍 Set Start", command=self.set_trim_start).pack(side=tk.LEFT, padx=5)
        ttk.Button(trim_buttons, text="📍 Set End", command=self.set_trim_end).pack(side=tk.LEFT, padx=5)
        ttk.Button(trim_buttons, text="✂️ Apply Trim & Save", command=self.apply_trim).pack(side=tk.LEFT, padx=5)
        ttk.Button(trim_buttons, text="🔄 Reset", command=self.reset_trim).pack(side=tk.LEFT, padx=5)

        self.trim_label = ttk.Label(trim_frame, text="No trim range set", foreground="gray")
        self.trim_label.pack(pady=5)

        # Target detection controls
        detection_frame = ttk.LabelFrame(control_frame, text="Target Detection", padding=10)
        detection_frame.pack(fill=tk.X, pady=5)

        detect_btn_frame = ttk.Frame(detection_frame)
        detect_btn_frame.pack(fill=tk.X)

        ttk.Button(detect_btn_frame, text="🎯 Detect Targets", command=self.detect_targets).pack(side=tk.LEFT, padx=5)
        ttk.Button(detect_btn_frame, text="💾 Save Detections", command=self.save_detections).pack(side=tk.LEFT, padx=5)
        ttk.Button(detect_btn_frame, text="🗑️ Clear", command=self.clear_detections).pack(side=tk.LEFT, padx=5)

        self.detection_label = ttk.Label(detection_frame, text="No detections", foreground="gray")
        self.detection_label.pack(side=tk.LEFT, padx=10)

        # Info display
        info_frame = ttk.LabelFrame(control_frame, text="Frame Info", padding=10)
        info_frame.pack(fill=tk.X, pady=5)

        self.info_text = tk.Text(info_frame, height=4, state='disabled', wrap='word')
        self.info_text.pack(fill=tk.X)

    def create_display_area(self):
        """Create image display area"""
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Color canvas
        color_frame = ttk.LabelFrame(display_frame, text="Color Image", padding=5)
        color_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.color_canvas = tk.Canvas(color_frame, bg='gray20')
        self.color_canvas.pack(fill=tk.BOTH, expand=True)

        # Depth canvas
        depth_frame = ttk.LabelFrame(display_frame, text="Depth Image", padding=5)
        depth_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.depth_canvas = tk.Canvas(depth_frame, bg='gray20')
        self.depth_canvas.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ========== Data Loading ==========

    def browse_folder(self):
        """Browse for data folder"""
        folder = filedialog.askdirectory(title="Select Data Folder")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder_path):
        """Load data from folder"""
        try:
            self.status_bar.config(text=f"Loading folder: {folder_path}")
            self.root.update()

            # Create DataManager
            self.data_manager = DataManager(folder_path)

            # Update UI
            self.folder_label.config(text=folder_path, foreground="black")

            # Populate trial dropdown
            trials = self.data_manager.list_trials()
            self.trial_combo['values'] = trials

            if trials:
                self.trial_combo.current(0)
                self.on_trial_changed(None)

            self.status_bar.config(text=f"Loaded {len(trials)} trials from {folder_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load folder:\n{e}")
            self.status_bar.config(text="Error loading folder")

    def on_trial_changed(self, event):
        """Handle trial selection change"""
        trial_name = self.trial_var.get()
        if not trial_name or not self.data_manager:
            return

        self.current_trial = trial_name
        trial_info = self.data_manager.get_trial_info(trial_name)

        # Update camera dropdown
        cameras = trial_info.cameras
        if cameras:
            self.camera_combo['values'] = cameras
            self.camera_combo.current(0)
            self.current_camera = cameras[0]
        else:
            self.camera_combo['values'] = []
            self.current_camera = None

        self.on_camera_changed(None)

    def on_camera_changed(self, event):
        """Handle camera selection change"""
        if not self.current_trial or not self.data_manager:
            return

        camera = self.camera_var.get() if self.camera_var.get() else None
        self.current_camera = camera

        # Process trial to trial_input/ (auto-standardize)
        try:
            self.status_bar.config(text="Processing trial to trial_input/...")
            self.root.update()

            # Import process_trial
            from process_trial import process_trial, find_all_frames

            trial_info = self.data_manager.get_trial_info(self.current_trial)

            # Create standardized trial_input folder
            output_path = process_trial(
                trial_path=trial_info.trial_path,
                camera_id=camera,
                output_base="trial_input",
                frame_range=None  # Process all frames
            )

            # Save the trial_input path for sharing with other pages
            from pathlib import Path
            self.trial_input_path = Path(output_path)

            print(f"✅ Standardized trial saved to: {output_path}")

            # Now find frames in trial_input (standardized location)
            self.status_bar.config(text="Loading standardized data...")
            self.root.update()

            # Find frames from trial_input
            self.available_frames = find_all_frames(trial_info.trial_path, camera)

            if self.available_frames:
                # Update slider
                self.frame_slider.config(from_=0, to=len(self.available_frames) - 1)
                self.frame_slider.set(0)

                self.status_bar.config(text=f"Ready: {len(self.available_frames)} frames in trial_input/")

                # Load first frame
                self.load_current_frame()
            else:
                self.status_bar.config(text="No frames found")
                messagebox.showwarning("Warning", "No frames found for this trial/camera")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process trial:\n{e}")
            self.status_bar.config(text="Error processing trial")
            import traceback
            traceback.print_exc()

    def on_frame_changed(self, value):
        """Handle frame slider change"""
        self.load_current_frame()

    def jump_frames(self, delta):
        """Jump forward or backward by delta frames"""
        current = self.frame_slider.get()
        new_value = max(0, min(len(self.available_frames) - 1, current + delta))
        self.frame_slider.set(new_value)

    def set_trim_start(self):
        """Set trim start frame to current frame"""
        if not self.available_frames:
            messagebox.showwarning("Warning", "No frames loaded")
            return

        frame_idx = int(self.frame_slider.get())
        frame_num = self.available_frames[frame_idx]
        self.trim_start = frame_num

        self.update_trim_label()

    def set_trim_end(self):
        """Set trim end frame to current frame"""
        if not self.available_frames:
            messagebox.showwarning("Warning", "No frames loaded")
            return

        frame_idx = int(self.frame_slider.get())
        frame_num = self.available_frames[frame_idx]
        self.trim_end = frame_num

        self.update_trim_label()

    def reset_trim(self):
        """Reset trim range"""
        self.trim_start = None
        self.trim_end = None
        self.update_trim_label()

    def update_trim_label(self):
        """Update trim range label"""
        if self.trim_start is None and self.trim_end is None:
            self.trim_label.config(text="No trim range set", foreground="gray")
        elif self.trim_start is not None and self.trim_end is not None:
            num_frames = abs(self.trim_end - self.trim_start) + 1
            self.trim_label.config(
                text=f"Trim: Frame {self.trim_start} to {self.trim_end} ({num_frames} frames)",
                foreground="blue"
            )
        elif self.trim_start is not None:
            self.trim_label.config(text=f"Start: Frame {self.trim_start} (set end frame)", foreground="orange")
        elif self.trim_end is not None:
            self.trim_label.config(text=f"End: Frame {self.trim_end} (set start frame)", foreground="orange")

    def apply_trim(self):
        """Apply trimming and re-save to trial_input"""
        if self.trim_start is None or self.trim_end is None:
            messagebox.showwarning("Warning", "Please set both start and end frames")
            return

        if not self.current_trial or not self.current_camera:
            messagebox.showwarning("Warning", "No trial loaded")
            return

        # Ensure start < end
        start_frame = min(self.trim_start, self.trim_end)
        end_frame = max(self.trim_start, self.trim_end)

        num_frames = end_frame - start_frame + 1

        response = messagebox.askyesno(
            "Confirm Trim",
            f"Trim trial to frames {start_frame} - {end_frame}?\n"
            f"This will save {num_frames} frames to trial_input/\n\n"
            f"Original data will not be affected."
        )

        if not response:
            return

        try:
            self.status_bar.config(text=f"Trimming and saving {num_frames} frames...")
            self.root.update()

            # Clear existing trial_input directory first
            trial_info = self.data_manager.get_trial_info(self.current_trial)
            trial_name = os.path.basename(trial_info.trial_path)

            if self.current_camera:
                old_output_path = os.path.join("trial_input", trial_name, self.current_camera)
            else:
                old_output_path = os.path.join("trial_input", trial_name, "single_camera")

            # Remove old frames if directory exists
            if os.path.exists(old_output_path):
                import shutil
                print(f"🗑️ Clearing old frames from: {old_output_path}")
                shutil.rmtree(old_output_path)

            # Import process_trial
            from process_trial import process_trial, find_all_frames

            # Process trial with frame range
            output_path = process_trial(
                trial_path=trial_info.trial_path,
                camera_id=self.current_camera,
                output_base="trial_input",
                frame_range=(start_frame, end_frame)  # Apply trim range
            )

            # Save the trial_input path
            from pathlib import Path
            self.trial_input_path = Path(output_path)

            print(f"✅ Trimmed trial saved to: {output_path}")
            print(f"   Frames: {start_frame} to {end_frame} ({num_frames} frames)")

            # Reload frames from trial_input
            self.available_frames = find_all_frames(trial_info.trial_path, self.current_camera)

            if self.available_frames:
                # Update slider
                self.frame_slider.config(from_=0, to=len(self.available_frames) - 1)
                self.frame_slider.set(0)

                self.status_bar.config(text=f"✅ Trimmed: {len(self.available_frames)} frames saved to trial_input/")

                # Reset trim markers
                self.reset_trim()

                # Load first frame
                self.load_current_frame()

                messagebox.showinfo("Success", f"Trimmed {num_frames} frames and saved to trial_input/")
            else:
                messagebox.showerror("Error", "No frames found after trimming")

        except Exception as e:
            messagebox.showerror("Error", f"Trimming failed:\n{e}")
            import traceback
            traceback.print_exc()

    def load_current_frame(self):
        """Load and display current frame FROM trial_input/"""
        if not self.available_frames or not self.trial_input_manager:
            return

        frame_idx = int(self.frame_slider.get())
        frame_num = self.available_frames[frame_idx]

        try:
            self.status_bar.config(text=f"Loading frame {frame_num} from trial_input/...")
            self.root.update()

            # Load frame from trial_input/ (standardized location)
            color, depth = self.trial_input_manager.load_frame(
                trial_name=self.current_trial,
                camera_id=self.current_camera,
                frame_number=frame_num
            )

            # Store current frame
            self.current_color = color
            self.current_depth = depth
            self.current_frame = frame_num

            # Clear detections when changing frames
            self.current_detections = []
            self.detection_overlay = False

            # Display
            self.display_images(color, depth)

            # Update info
            self.update_info(frame_num, color, depth)

            self.status_bar.config(text=f"✅ Frame {frame_num} from trial_input/ ({frame_idx + 1}/{len(self.available_frames)})")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load frame {frame_num}:\n{e}")
            self.status_bar.config(text=f"Error loading frame {frame_num}")

    def display_images(self, color, depth):
        """Display color and depth images"""
        # Draw detections on color if enabled
        color_display = color
        if color is not None and self.detection_overlay and self.current_detections:
            if self.target_detector:
                color_display = self.target_detector.draw_detections(color, self.current_detections)

        # Display color
        if color_display is not None:
            self.display_image_on_canvas(color_display, self.color_canvas)
        else:
            self.clear_canvas(self.color_canvas)

        # Display depth
        if depth is not None:
            # Convert depth to colormap
            depth_viz = self.depth_to_colormap(depth)

            # Draw detections on depth if enabled
            if self.detection_overlay and self.current_detections:
                depth_viz = self.draw_detections_on_depth(depth, depth_viz, self.current_detections)

            self.display_image_on_canvas(depth_viz, self.depth_canvas)
        else:
            self.clear_canvas(self.depth_canvas)

    def depth_to_colormap(self, depth):
        """Convert depth to colored visualization"""
        depth_viz = depth.copy()
        valid_depth = depth_viz[depth_viz > 0]

        if len(valid_depth) > 0:
            depth_min = valid_depth.min()
            depth_max = np.percentile(valid_depth, 95)
            depth_viz = np.clip(depth_viz, depth_min, depth_max)
            depth_viz = (depth_viz - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_viz = depth_viz * 0

        depth_viz = depth_viz.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)

        return depth_colored

    def draw_detections_on_depth(self, depth, depth_viz, detections):
        """Draw bounding boxes and average depth on depth image"""
        img = depth_viz.copy()

        for det in detections:
            # Draw bounding box
            cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)

            # Compute average depth within bbox
            bbox_depth = depth[det.y1:det.y2, det.x1:det.x2]
            valid_depth = bbox_depth[bbox_depth > 0]

            if len(valid_depth) > 0:
                avg_depth = valid_depth.mean()
                median_depth = np.median(valid_depth)

                # Draw label with average and median depth
                label_text = f"{det.label} avg:{avg_depth:.3f}m"
                cv2.putText(img, label_text, (det.x1, det.y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                median_text = f"med:{median_depth:.3f}m"
                cv2.putText(img, median_text, (det.x1, det.y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center point
            center = det.center
            cv2.circle(img, center, 5, (0, 255, 0), -1)

        return img

    def display_image_on_canvas(self, img, canvas):
        """Display image on canvas"""
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        # Get canvas size
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Resize image to fit canvas
        img_h, img_w = img_rgb.shape[:2]
        scale = min(canvas_width / img_w, canvas_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        img_resized = cv2.resize(img_rgb, (new_w, new_h))

        # Convert to PIL and then to ImageTk
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Store reference
        canvas.image = img_tk

        # Display
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk, anchor=tk.CENTER)

    def clear_canvas(self, canvas):
        """Clear canvas"""
        canvas.delete("all")

    def update_info(self, frame_num, color, depth):
        """Update info display"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)

        info = f"📂 Source: trial_input/ (standardized)\n"
        info += f"Trial: {self.current_trial}\n"
        if self.current_camera:
            info += f"Camera: {self.current_camera}\n"
        info += f"Frame: {frame_num}\n"

        if color is not None:
            info += f"Color: {color.shape}, PNG\n"

        if depth is not None:
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                info += f"Depth: {depth.shape}, .npy, [{valid_depth.min():.3f}-{depth.max():.3f}]m"

        self.info_text.insert(1.0, info)
        self.info_text.config(state='disabled')

    # ========== Config Management ==========

    def save_config(self):
        """Save data configuration"""
        if not self.data_manager:
            messagebox.showwarning("Warning", "No data loaded")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.data_manager.save_config(file_path)
                messagebox.showinfo("Success", f"Configuration saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config:\n{e}")

    def load_config(self):
        """Load data configuration"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.data_manager = DataManager.from_config(file_path)
                self.folder_label.config(text=self.data_manager.root_folder, foreground="black")

                trials = self.data_manager.list_trials()
                self.trial_combo['values'] = trials

                if trials:
                    self.trial_combo.current(0)
                    self.on_trial_changed(None)

                messagebox.showinfo("Success", f"Loaded {len(trials)} trials")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config:\n{e}")

    # ========== Target Detection ==========

    def detect_targets(self):
        """Detect targets in current frame"""
        if self.current_color is None or self.current_depth is None:
            messagebox.showwarning("Warning", "Please load a frame first")
            return

        # Initialize detector if needed
        if self.target_detector is None:
            model_path = get_default_model_path()
            if model_path is None:
                messagebox.showerror("Error", "Model not found. Please check step1_calibration_process/target_detection/automatic_mode/best.pt")
                return

            self.status_bar.config(text="Loading YOLO model...")
            self.root.update()
            self.target_detector = TargetDetector(model_path, confidence_threshold=0.5)

        # Run detection
        self.status_bar.config(text="Detecting targets...")
        self.root.update()

        # Auto-detect camera intrinsics
        h, w = self.current_depth.shape
        if w == 640 and h == 480:
            fx = fy = 615.0
            cx, cy = 320.0, 240.0
        elif w == 1280 and h == 720:
            fx = fy = 922.5
            cx, cy = 640.0, 360.0
        else:
            fx = fy = w * 0.96
            cx, cy = w / 2, h / 2

        self.current_detections = self.target_detector.detect(
            self.current_color, self.current_depth,
            fx=fx, fy=fy, cx=cx, cy=cy
        )

        # Update display
        if self.current_detections:
            self.detection_overlay = True
            self.display_images(self.current_color, self.current_depth)
            self.detection_label.config(
                text=f"Found {len(self.current_detections)} target(s)",
                foreground="green"
            )
            self.status_bar.config(text=f"✅ Detected {len(self.current_detections)} target(s)")
        else:
            self.detection_label.config(text="No targets detected", foreground="orange")
            self.status_bar.config(text="No targets detected")

    def clear_detections(self):
        """Clear current detections"""
        self.current_detections = []
        self.detection_overlay = False
        self.detection_label.config(text="No detections", foreground="gray")
        if self.current_color is not None:
            self.display_images(self.current_color, self.current_depth)

    def save_detections(self):
        """Save detections to trial_output folder in legacy format"""
        if not self.current_detections:
            messagebox.showwarning("Warning", "No detections to save")
            return

        if not self.current_trial or self.current_frame is None:
            messagebox.showwarning("Warning", "No frame loaded")
            return

        # Create output directory
        output_dir = Path("trial_output") / self.current_trial
        if self.current_camera:
            output_dir = output_dir / self.current_camera
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort detections from right to left (by x-coordinate descending)
        sorted_detections = sorted(self.current_detections, key=lambda d: d.center[0], reverse=True)

        # Convert to legacy format
        detections_array = []
        for i, det in enumerate(sorted_detections, start=1):
            detection_dict = {
                "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                "center_px": [int(det.center[0]), int(det.center[1])],
                "avg_depth_m": float(det.avg_depth) if det.avg_depth is not None else 0.0,
                "x": float(det.center_3d[0]) if det.center_3d else 0.0,
                "y": float(det.center_3d[1]) if det.center_3d else 0.0,
                "z": float(det.center_3d[2]) if det.center_3d else 0.0,
                "label": f"target_{i}"
            }
            detections_array.append(detection_dict)

        # Save as JSON (array format, matching legacy format)
        output_file = output_dir / "target_detections_cam_frame.json"

        with open(output_file, 'w') as f:
            json.dump(detections_array, f, indent=2)

        # Compute and save ground plane transformation
        if len(detections_array) >= 3:
            try:
                # Ensure parent directory is in path for imports
                parent_dir = Path(__file__).parent.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))

                from step2_skeleton_extraction.ground_plane_correction import (
                    compute_ground_plane_transform,
                    fit_plane_to_points,
                    get_transform_info
                )

                R = compute_ground_plane_transform(detections_array)
            except ImportError as e:
                print(f"⚠️  Could not import ground_plane_correction: {e}")
                R = None
            except Exception as e:
                print(f"⚠️  Error computing ground plane transform: {e}")
                R = None

            if R is not None:
                # Get plane info
                target_positions = np.array([[d['x'], d['y'], d['z']] for d in detections_array])
                normal, centroid = fit_plane_to_points(target_positions)
                transform_info = get_transform_info(R, normal)

                # Save transformation
                transform_file = output_dir / "ground_plane_transform.json"
                with open(transform_file, 'w') as f:
                    json.dump({
                        'rotation_matrix': R.tolist(),
                        'info': transform_info,
                        'description': 'Rotation matrix to align ground plane to horizontal'
                    }, f, indent=2)

                print(f"✅ Computed ground plane correction: {transform_info['angle_deg']:.2f}° tilt")
                print(f"💾 Saved transformation to: {transform_file}")

        messagebox.showinfo("Success", f"Saved {len(self.current_detections)} detection(s) to:\n{output_file}")
        self.status_bar.config(text=f"💾 Saved detections to {output_file}")


def main():
    """Main entry point"""
    initial_folder = None

    if len(sys.argv) > 1:
        initial_folder = sys.argv[1]

    root = tk.Tk()
    app = DataLoaderUI(root, initial_folder)
    root.mainloop()


if __name__ == "__main__":
    main()
