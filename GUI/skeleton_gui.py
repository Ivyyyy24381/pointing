from tqdm import tqdm
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml
import os 
import pandas as pd
import sys
sys.path.append('./')  
sys.path.append('gesture')
from gesture_data_process import GestureDataProcessor
from batch_point_production import run_gesture_detection

class VideoTrimmerGUI:
    INTRINSICS_PATH = "config/camera_config.yaml"
    TARGETS_PATH = "config/targets.yaml"
    INTRINSICS = yaml.safe_load(open(INTRINSICS_PATH, 'r'))
    TARGETS = yaml.safe_load(open(TARGETS_PATH, 'r'))
    
    def __init__(self, root):
        self.root = root
        self.root.title("Video Trimmer with Pointing Arm Selection")

        self.video_path = None
        self.cap = None
        self.frame_pos = 0
        self.total_frames = 0
        self.start_frame = 0
        self.end_frame = 0
        self.video_queue = []
        self.progress = None
        self.forced_pointing_arm = None  # New attribute for forced pointing arm

        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Video canvas
        self.canvas = tk.Canvas(main_frame, width=640, height=480, highlightthickness=2)
        self.canvas.pack()

        # Control buttons frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(pady=5)

        # Load video button
        self.load_btn = tk.Button(control_frame, text="Load Video", command=self.load_video)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Frame slider
        self.frame_slider = tk.Scale(main_frame, from_=0, to=0, orient=tk.HORIZONTAL, 
                                   label="Frame", command=self.update_frame, length=800)
        self.frame_slider.pack(pady=5)

        # Frame marking buttons
        mark_frame = tk.Frame(main_frame)
        mark_frame.pack(pady=5)

        self.mark_start_btn = tk.Button(mark_frame, text="Mark Start", command=self.mark_start_frame)
        self.mark_start_btn.pack(side=tk.LEFT, padx=5)

        self.mark_end_btn = tk.Button(mark_frame, text="Mark End", command=self.mark_end_frame)
        self.mark_end_btn.pack(side=tk.LEFT, padx=5)

        # NEW: Trial selection frame (only show when batch processing)
        self.trial_selection_frame = tk.Frame(main_frame)
        # Don't pack initially - will be shown only for batch processing
        
        tk.Label(self.trial_selection_frame, text="Jump to Trial:").pack(side=tk.LEFT, padx=5)
        
        self.trial_var = tk.StringVar(value="Current")
        self.trial_combo = ttk.Combobox(self.trial_selection_frame, textvariable=self.trial_var, 
                                       state="readonly", width=15)
        self.trial_combo.pack(side=tk.LEFT, padx=5)
        self.trial_combo.bind("<<ComboboxSelected>>", self.on_trial_changed)
        
        # Navigation buttons for trials
        self.prev_trial_btn = tk.Button(self.trial_selection_frame, text="◀ Prev Trial", 
                                       command=self.prev_trial)
        self.prev_trial_btn.pack(side=tk.LEFT, padx=2)
        
        self.next_trial_btn = tk.Button(self.trial_selection_frame, text="Next Trial ▶", 
                                       command=self.next_trial)
        self.next_trial_btn.pack(side=tk.LEFT, padx=2)

        # NEW: Pointing arm selection frame
        pointing_arm_frame = tk.Frame(main_frame)
        pointing_arm_frame.pack(pady=10)

        tk.Label(pointing_arm_frame, text="Force Pointing Arm:").pack(side=tk.LEFT, padx=5)
        
        self.pointing_arm_var = tk.StringVar(value="Auto")  # Default to Auto
        pointing_arm_options = ["Auto", "Left", "Right"]
        
        self.pointing_arm_combo = ttk.Combobox(pointing_arm_frame, textvariable=self.pointing_arm_var, 
                                             values=pointing_arm_options, state="readonly", width=10)
        self.pointing_arm_combo.pack(side=tk.LEFT, padx=5)
        
        # Bind selection change event
        self.pointing_arm_combo.bind("<<ComboboxSelected>>", self.on_pointing_arm_changed)

        # Process button
        process_frame = tk.Frame(main_frame)
        process_frame.pack(pady=5)
        
        self.process_btn = tk.Button(process_frame, text="Process Selection", command=self.process_selection)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        # Add "Process All Remaining" button for batch mode
        self.process_all_btn = tk.Button(process_frame, text="Process All Remaining", 
                                        command=self.process_all_remaining)
        # Don't pack initially - will be shown only for batch processing

        # Keyboard bindings
        self.root.bind("<Right>", self.next_frame)
        self.root.bind("<Left>", self.prev_frame)
        self.root.bind("<Control-Right>", lambda e: self.next_trial())  # Ctrl+Right for next trial
        self.root.bind("<Control-Left>", lambda e: self.prev_trial())   # Ctrl+Left for prev trial
        self.root.bind("<Tab>", lambda e: self.next_trial())            # Tab for next trial
        self.root.bind("<Shift-Tab>", lambda e: self.prev_trial())      # Shift+Tab for prev trial

    def process_all_remaining(self):
        """Process all remaining trials starting from current position"""
        if not hasattr(self, "video_batch") or not hasattr(self, "batch_index"):
            messagebox.showwarning("Warning", "No batch processing mode active")
            return
            
        remaining_count = len(self.video_batch) - self.batch_index
        result = messagebox.askyesno("Confirm Batch Processing", 
                                   f"Process {remaining_count} remaining trials?\n"
                                   f"Current pointing arm setting: {self.pointing_arm_var.get()}")
        
        if not result:
            return
            
        print(f"🚀 Starting batch processing of {remaining_count} trials...")
        
        # Process current and all remaining trials
        while self.batch_index < len(self.video_batch):
            current_trial = os.path.basename(os.path.dirname(self.video_batch[self.batch_index]))
            print(f"\n📹 Processing trial {self.batch_index + 1}/{len(self.video_batch)}: {current_trial}")
            
            # Process current selection for this trial
            self.process_selection()
            
            # Move to next trial
            if self.batch_index < len(self.video_batch) - 1:
                self.batch_index += 1
                self.load_video_path(self.video_batch[self.batch_index])
            else:
                break
                
        print("🎉 Batch processing completed!")
        messagebox.showinfo("Batch Processing Complete", 
                           f"Successfully processed {remaining_count} trials.\n"
                           f"Global CSV has been updated.")

    def on_trial_changed(self, event=None):
        """Handle trial selection change"""
        selected_trial = self.trial_var.get()
        if selected_trial != "Current" and hasattr(self, "video_batch"):
            # Remove status indicators to get clean trial name
            clean_trial_name = selected_trial.replace("✅ ", "").replace("⏳ ", "")
            
            # Find the trial in video_batch
            for i, video_path in enumerate(self.video_batch):
                trial_name = os.path.basename(os.path.dirname(video_path))
                if trial_name == clean_trial_name:
                    print(f"🎯 Jumping to trial: {clean_trial_name}")
                    self.batch_index = i
                    self.load_video_path(video_path)
                    break

    def prev_trial(self):
        """Go to previous trial"""
        if hasattr(self, "video_batch") and hasattr(self, "batch_index"):
            if self.batch_index > 0:
                self.batch_index -= 1
                self.load_video_path(self.video_batch[self.batch_index])
                self.update_trial_selector()

    def next_trial(self):
        """Go to next trial"""
        if hasattr(self, "video_batch") and hasattr(self, "batch_index"):
            if self.batch_index < len(self.video_batch) - 1:
                self.batch_index += 1
                self.load_video_path(self.video_batch[self.batch_index])
                self.update_trial_selector()

    def update_trial_selector(self):
        """Update the trial selector dropdown and current selection"""
        if hasattr(self, "video_batch"):
            # Extract trial names from video paths and check processing status
            trial_names = []
            for video_path in self.video_batch:
                trial_name = os.path.basename(os.path.dirname(video_path))
                
                # Check if this trial has been processed
                trial_root = os.path.dirname(video_path)
                processed_csv = os.path.join(trial_root, "processed_gesture_data.csv")
                
                if os.path.exists(processed_csv):
                    trial_display = f"✅ {trial_name}"  # Checkmark for processed trials
                else:
                    trial_display = f"⏳ {trial_name}"  # Hourglass for unprocessed trials
                    
                trial_names.append(trial_display)
            
            # Update dropdown options
            self.trial_combo['values'] = trial_names
            
            # Set current selection
            if hasattr(self, "batch_index") and self.batch_index < len(trial_names):
                current_trial = trial_names[self.batch_index]
                self.trial_var.set(current_trial)
                
                # Update button states
                self.prev_trial_btn.config(state="normal" if self.batch_index > 0 else "disabled")
                self.next_trial_btn.config(state="normal" if self.batch_index < len(self.video_batch) - 1 else "disabled")

    def on_pointing_arm_changed(self, event=None):
        """Handle pointing arm selection change"""
        selected = self.pointing_arm_var.get()
        if selected == "Auto":
            self.forced_pointing_arm = None
        else:
            self.forced_pointing_arm = selected
        print(f"Pointing arm selection: {selected}")

    def process_selection(self):
        print(f"Processing video: {self.video_path}")
        print(f"Start frame: {self.start_frame}")
        print(f"End frame: {self.end_frame}")
        print(f"Forced pointing arm: {self.forced_pointing_arm}")

        import shutil
        from tqdm import tqdm
        import os

        root_path = self.video_path.rsplit('/', 1)[0]
        color_video_path = os.path.join(root_path, 'Color.mp4')
        data_path = os.path.join(root_path, "gesture_data.csv")
        
        # Generate gesture data if it doesn't exist
        if not os.path.exists(data_path):
            base_path, subject_name, trial_no, _ = self.video_path.rsplit('/', 3)
            run_gesture_detection(base_path, subject_folder=subject_name, trial_id=trial_no)
            
        # Initialize gesture data processor
        gesture_data_processor = GestureDataProcessor(data_path)
        trimmed_data = gesture_data_processor.trim_data(gesture_data_processor, 
                                                       start_frame=self.start_frame, 
                                                       end_frame=self.end_frame)

        # Remove existing processed_gesture_data.csv before processing
        processed_csv_path = os.path.join(root_path, "processed_gesture_data.csv")
        if os.path.exists(processed_csv_path):
            os.remove(processed_csv_path)

        # Process data with optional forced pointing arm
        gesture_data_processor.process_data(trimmed_data, forced_pointing_arm=self.forced_pointing_arm)

        tqdm.write(f"✅ Processed trimmed data saved to {root_path}/processed_gesture_data.csv")

        # NEW: Always update global CSV after processing any trial
        self.update_global_csv()

        # Reset video and proceed to next
        if hasattr(self, "video_batch") and hasattr(self, "batch_index"):
            self.cap.release()
            self.canvas.delete("all")
            # Don't auto-advance in manual trial selection mode
            # User can manually navigate using trial selector
            print(f"✅ Trial {os.path.basename(root_path)} processed. Use trial selector to navigate.")
        else:
            if self.progress:
                self.progress.update(1)
            self.next_video()

    def update_global_csv(self):
        """Update the global CSV file after processing all videos with enhanced guards"""
        try:
            if len(sys.argv) > 1:
                from urllib.parse import unquote
                root_folder = unquote(" ".join(sys.argv[1:]).strip('"'))
                if os.path.isdir(root_folder):
                    all_data = []
                    processed_folders = []
                    total_subfolders = 0
                    
                    # First pass: count total subfolders and check for processed data
                    for subdir, _, files in os.walk(root_folder):
                        # Skip the root folder itself
                        if subdir == root_folder:
                            continue
                            
                        # Check if this is a trial subfolder (contains Color.mp4)
                        if 'Color.mp4' in files:
                            total_subfolders += 1
                            csv_path = os.path.join(subdir, "processed_gesture_data.csv")
                            
                            if os.path.exists(csv_path):
                                try:
                                    df = pd.read_csv(csv_path)
                                    
                                    # Guard: Check if CSV has actual data (not just headers)
                                    if len(df) > 0:
                                        path_parts = os.path.normpath(subdir).split(os.sep)
                                        if len(path_parts) >= 2:
                                            df["dog"] = path_parts[-2]
                                            df["trial"] = path_parts[-1]
                                        else:
                                            df["dog"] = "unknown"
                                            df["trial"] = os.path.basename(subdir)
                                        
                                        all_data.append(df)
                                        processed_folders.append(os.path.basename(subdir))
                                        print(f"📁 Added data from trial: {os.path.basename(subdir)} ({len(df)} frames)")
                                    else:
                                        print(f"⚠️ Empty CSV found in: {os.path.basename(subdir)}")
                                        
                                except Exception as e:
                                    print(f"❌ Failed to read {csv_path}: {e}")
                            else:
                                print(f"⚠️ No processed CSV found in: {os.path.basename(subdir)}")
                    
                    # Guard: Only create global CSV if we have meaningful data
                    if all_data and len(all_data) > 0:
                        print(f"📊 Combining data from {len(all_data)}/{total_subfolders} trials")
                        
                        combined_df = pd.concat(all_data, ignore_index=True)
                        
                        # Guard: Check combined data is not empty
                        if len(combined_df) == 0:
                            print("⚠️ Combined dataset is empty, skipping global CSV creation")
                            return
                        
                        # Reorder columns to put 'dog' and 'trial' first
                        cols = combined_df.columns.tolist()
                        if 'dog' in cols and 'trial' in cols:
                            cols.remove('dog')
                            cols.remove('trial')
                            combined_df = combined_df[['dog', 'trial'] + cols]
                        
                        # Sort by 'trial' column (converted to numeric if possible)
                        try:
                            combined_df['trial_numeric'] = pd.to_numeric(combined_df['trial'], errors='coerce')
                            combined_df = combined_df.sort_values(by=['dog', 'trial_numeric', 'frame']).drop(columns=['trial_numeric'])
                        except Exception as e:
                            print(f"⚠️ Failed to sort by trial: {e}")
                            # Fallback: sort by trial as string
                            combined_df = combined_df.sort_values(by=['dog', 'trial', 'frame'])
                        
                        # Generate appropriate filename based on folder structure
                        path_parts = os.path.normpath(root_folder).split(os.sep)
                        root_folder_name = path_parts[-1]
                        
                        # Check if folder name contains "front" pattern and create appropriate filename
                        if "front" in root_folder_name.lower():
                            filename = f"{root_folder_name}_gesture_data.csv"
                        else:
                            # If not explicitly "front", still use the folder name
                            filename = f"{root_folder_name}_gesture_data.csv"
                        
                        save_path = os.path.join(root_folder, filename)
                        
                        # Guard: Check if we can write to the location
                        try:
                            combined_df.to_csv(save_path, index=False)
                            print(f"✅ Successfully saved combined data to: {filename}")
                            print(f"📈 Total frames: {len(combined_df)}")
                            print(f"🎯 Trials included: {', '.join(sorted(processed_folders))}")
                            
                            # Additional statistics
                            if 'pointing_arm' in combined_df.columns:
                                left_count = (combined_df['pointing_arm'] == 'Left').sum()
                                right_count = (combined_df['pointing_arm'] == 'Right').sum()
                                print(f"🫲 Left arm frames: {left_count}")
                                print(f"🫱 Right arm frames: {right_count}")
                                
                        except PermissionError:
                            print(f"❌ Permission denied: Cannot write to {save_path}")
                        except Exception as e:
                            print(f"❌ Error saving combined CSV: {e}")
                            
                    else:
                        if total_subfolders == 0:
                            print("⚠️ No trial subfolders found (folders with Color.mp4)")
                        else:
                            print(f"⚠️ No valid processed data found in {total_subfolders} trial folders")
                            print("💡 Make sure to process videos first before updating global CSV")
                            
                else:
                    print(f"❌ Invalid directory: {root_folder}")
            else:
                print("⚠️ No root folder specified for global CSV update")
                
        except Exception as e:
            print(f"❌ Error updating global CSV: {e}")
            import traceback
            print(f"🔍 Debug info: {traceback.format_exc()}")

    def set_video_queue(self, video_list):
        self.video_queue = video_list
        if self.video_queue:
            self.progress = tqdm(total=len(self.video_queue), desc="Processing Videos")
            self.load_video_path(self.video_queue.pop(0))

    def next_video(self):
        if self.video_queue:
            next_path = self.video_queue.pop(0)
            self.load_video_path(next_path)
        else:
            if self.progress:
                self.progress.close()
            # Update global CSV when processing queue is finished
            self.update_global_csv()
            self.root.quit()
    
    def load_video(self, video_path=None):
        if video_path:
            self.video_path = video_path
        else:
            self.video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if not self.video_path:
            return
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_pos = 0
        self.start_frame = 0
        self.end_frame = self.total_frames - 1
        self.frame_slider.config(to=self.total_frames - 1)
        self.frame_slider.set(0)
        self.show_frame()

    def show_frame(self):
        if self.cap:
            self.frame_pos = self.frame_slider.get()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_pos)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                self.tk_image = self.cv_to_tk(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
                
                # Enhanced title with trial information
                title = f"Video Trimmer - Frame {self.frame_pos}/{self.total_frames}"
                if hasattr(self, "video_batch") and hasattr(self, "batch_index"):
                    current_trial = os.path.basename(os.path.dirname(self.video_path))
                    title += f" - Trial: {current_trial} ({self.batch_index + 1}/{len(self.video_batch)})"
                
                self.root.title(title)
                
                if self.start_frame <= self.frame_pos <= self.end_frame:
                    self.canvas.config(highlightbackground="green")
                else:
                    self.canvas.config(highlightbackground="red")

    def cv_to_tk(self, frame):
        import PIL.Image, PIL.ImageTk
        img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return PIL.ImageTk.PhotoImage(image=img)

    def next_frame(self, event=None):
        if self.frame_pos < self.total_frames - 1:
            self.frame_pos += 1
            self.frame_slider.set(self.frame_pos)
            self.show_frame()

    def prev_frame(self, event=None):
        if self.frame_pos > 0:
            self.frame_pos -= 1
            self.frame_slider.set(self.frame_pos)
            self.show_frame()

    def update_frame(self, val):
        self.frame_pos = int(val)
        self.show_frame()

    def mark_start_frame(self):
        self.start_frame = self.frame_pos
        self.mark_start_btn.config(text=f"Mark Start (Frame {self.start_frame})")

    def mark_end_frame(self):
        self.end_frame = self.frame_pos
        self.mark_end_btn.config(text=f"Mark End (Frame {self.end_frame})")

    def load_video_path(self, path):
        """Call this externally to load a video from a given path."""
        self.load_video(video_path=path)
        
        # Update trial selector if in batch mode
        if hasattr(self, "video_batch"):
            self.update_trial_selector()
            # Show trial selection frame and process all button for batch processing
            if not self.trial_selection_frame.winfo_viewable():
                self.trial_selection_frame.pack(pady=5, before=self.process_btn.master)
                self.process_all_btn.pack(side=tk.LEFT, padx=5)
        
if __name__ == "__main__":
    import sys
    import glob
    import os
    from urllib.parse import unquote

    root = tk.Tk()
    app = VideoTrimmerGUI(root)

    if len(sys.argv) > 1:
        # Join all arguments into a single path string
        raw_input = " ".join(sys.argv[1:])
        clean_input = unquote(raw_input.strip('"'))

        if os.path.isdir(clean_input):
            video_files = sorted(glob.glob(os.path.join(clean_input, "*", "Color.mp4")))
            print(f"🎞️ Found {len(video_files)} videos.")
            
            if video_files:
                # Set video_batch and batch_index, and load the first video
                app.video_batch = sorted(video_files)
                app.batch_index = 0
                
                # Extract trial names for user feedback
                trial_names = [os.path.basename(os.path.dirname(vf)) for vf in video_files]
                print(f"📁 Available trials: {', '.join(trial_names)}")
                
                app.load_video_path(app.video_batch[0])
            else:
                print("❌ No Color.mp4 files found in subdirectories")
        elif os.path.isfile(clean_input):
            app.load_video_path(clean_input)

    root.mainloop()