#!/usr/bin/env python3
"""
Pointing Gesture Analysis - Main UI
====================================

Integrated tabbed interface for the complete pointing gesture analysis pipeline,
combining target detection and skeleton extraction into a seamless workflow.

Architecture Overview
---------------------
The UI consists of two sequential pages organized as tabs:
  1. Page 1: Target Detection (Step 0) - Load trial data and detect pointing targets
  2. Page 2: Skeleton Processing (Step 2) - Extract skeletons and visualize in 3D

Workflow
--------
Typical usage pattern:
  1. Launch application
  2. Use Page 1 to load raw trial data and detect targets (cups/objects)
  3. Switch to Page 2 - trial auto-loads with detected targets
  4. Process frames to extract human/dog/baby skeletons
  5. Review results in 3D visualization
  6. Results auto-save to both trial_output/ and original data folder

Data Flow
---------
  Raw Data → Page 1 (detect targets) → trial_input/ (standardized) →
  Page 2 (extract skeletons) → trial_output/ (temp) → Original Folder (synced)

Key Features
------------
  - Automatic trial sharing between pages (no re-loading needed)
  - Standardized data format conversion (automatic)
  - Real-time 2D and 3D visualization
  - Multi-subject detection (human, dog, baby)
  - Batch processing with progress tracking
  - Automatic result synchronization
  - Status bar with contextual help

Screen Requirements
-------------------
  - Minimum Resolution: 1280x720 (HD)
  - Recommended Resolution: 1920x1080 (Full HD)
  - Optimal Resolution: 2560x1440 (2K) or higher
  - Aspect Ratio: 16:9 or wider
  - Display: Single monitor (dual monitor for advanced users)

The UI auto-sizes to 1920x1080 but will adapt to available screen space.
On smaller screens, scrollbars will appear automatically.

Usage
-----
Basic:
    python main_ui.py

With initial folder (bypass Browse step):
    python main_ui.py /path/to/data/folder

Dependencies
------------
  - tkinter: UI framework (built into Python)
  - step0_data_loading.ui_data_loader: Page 1 implementation
  - step2_skeleton_extraction.ui_skeleton_extractor: Page 2 implementation

See Also
--------
  - MAIN_UI_GUIDE.md: Complete workflow guide with examples
  - USER_GUIDE.md: Installation and quick start
  - TROUBLESHOOTING.md: Common issues and solutions

Author: Pointing Gesture Analysis Team
License: MIT
Version: 2.0
"""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from step0_data_loading.ui_data_loader import DataLoaderUI
from step2_skeleton_extraction.ui_skeleton_extractor import SkeletonExtractorUI


class ToolTip:
    """
    Create a tooltip for a given widget.

    Tooltips appear on mouse hover and provide helpful context about UI elements.
    They use a light yellow background (standard tooltip appearance) and
    automatically position near the cursor.

    Usage:
        button = tk.Button(root, text="Click me")
        ToolTip(button, "This button does something useful")

    Parameters:
        widget: The Tkinter widget to attach tooltip to
        text: The tooltip text to display
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        # Bind mouse events - show on enter, hide on leave
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        """Display tooltip near mouse cursor."""
        if self.tooltip_window or not self.text:
            return

        # Get widget position
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # Create tooltip window (no decorations)
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # No window border/title bar
        tw.wm_geometry(f"+{x}+{y}")

        # Create label with text (light yellow background is standard for tooltips)
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",  # Light yellow
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", 9)
        )
        label.pack(ipadx=5, ipady=3)

    def hide_tooltip(self, event=None):
        """Remove tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class FrameWrapper(tk.Frame):
    """
    Wrapper that makes a Frame behave like a Tk root for embedding UIs.

    Purpose:
    --------
    The Page 1 and Page 2 UIs (DataLoaderUI and SkeletonExtractorUI) were
    originally designed as standalone windows expecting a Tk root object.
    This wrapper allows them to be embedded within notebook tabs by:
      1. Accepting method calls intended for Tk root (title, geometry, config)
      2. Ignoring window-specific operations (they don't apply to frames)
      3. Providing a compatible interface without modifying the child UIs

    This enables code reuse - the same UI classes work both standalone
    and embedded within the main UI.

    Design Pattern:
    ---------------
    This is an Adapter pattern - adapting the Frame interface to match
    what the embedded UIs expect from a Tk root window.
    """

    def __init__(self, parent):
        """
        Initialize wrapper frame and pack it to fill parent container.

        Args:
            parent: The parent widget (notebook tab frame)
        """
        super().__init__(parent)
        # Fill entire tab area - BOTH ensures it fills width and height
        # expand=True allows it to grow if parent resizes
        self.pack(fill=tk.BOTH, expand=True)
        self._menu = None  # Store menu bar reference (not applied to frames)

    def title(self, text):
        """
        Ignore title calls - tab labels are set by notebook.add().

        The embedded UIs call root.title() to set window title,
        but in a notebook, tab text is set differently. We just
        ignore these calls.
        """
        pass

    def geometry(self, size):
        """
        Ignore geometry calls - frame size controlled by parent notebook.

        The embedded UIs call root.geometry() to set window size,
        but frames don't have independent geometry. They're sized
        by their parent. We ignore these calls.
        """
        pass

    def config(self, **kwargs):
        """
        Override config to intercept menu setting.

        Menus are application-level in Tk, not frame-level. We store the
        menu reference but don't apply it to avoid conflicts with the main
        window's menu bar.

        Args:
            **kwargs: Configuration options for the frame
        """
        if 'menu' in kwargs:
            # Store menu but don't apply it (frames can't have menus)
            self._menu = kwargs.pop('menu')
        if kwargs:  # If there are other valid frame options
            super().config(**kwargs)

    def configure(self, **kwargs):
        """Alias for config() - Tkinter convention supports both names."""
        self.config(**kwargs)


class MainUI:
    """
    Main UI controller with two-page tabbed interface.

    Responsibilities:
    -----------------
      1. Create and manage notebook (tab container)
      2. Initialize Page 1 (Target Detection) and Page 2 (Skeleton Processing)
      3. Handle tab switching and data flow between pages
      4. Display status messages to guide user workflow
      5. Coordinate automatic trial loading when switching pages

    Data Sharing Architecture:
    --------------------------
    When user switches from Page 1 to Page 2:
      1. on_tab_changed() event fires
      2. Check if Page 1 has a loaded trial (self.page1_ui.trial_input_path)
      3. If yes, automatically load that trial in Page 2
      4. This avoids manual re-loading and ensures consistency

    The trial_input_path is the key shared state variable:
      - Set by Page 1 when trial is loaded
      - Read by MainUI when switching to Page 2
      - Used to auto-populate Page 2 with same trial

    Benefits:
    ---------
      - Seamless workflow (no re-browsing folders)
      - Guaranteed consistency (same trial in both pages)
      - Picks up changes (e.g., if user trims frames in Page 1)
      - User-friendly (auto-load feels natural)
    """

    def __init__(self, root):
        """
        Initialize main UI window and create page structure.

        Args:
            root: Tk root window instance

        Setup Sequence:
        ---------------
          1. Configure main window (title, size)
          2. Create notebook widget (tab container)
          3. Build Page 1 (Target Detection)
          4. Build Page 2 (Skeleton Processing)
          5. Bind tab change event handler
          6. Create status bar for user feedback
        """
        self.root = root
        self.root.title("Pointing Gesture Analysis - Main UI")

        # Fixed size optimized for Full HD displays
        # Users with smaller screens will see scrollbars
        # Users with larger screens will have extra space
        # TODO: Consider making this adaptive (see MAIN_UI_DOCUMENTATION_IMPROVEMENTS.md)
        self.root.geometry("1920x1080")

        # Create main notebook (tab container)
        # fill=BOTH ensures it fills window in both directions
        # expand=True allows it to grow if window resizes
        # padx/pady add 5px margin around notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create and populate the two pages
        # Order matters: Page 1 must exist before Page 2 for data flow
        self.create_page1_target_detection()
        self.create_page2_skeleton_processing()

        # Bind tab change event to enable automatic trial loading
        # <<NotebookTabChanged>> fires whenever user clicks a different tab
        # This is the critical mechanism that makes the workflow seamless
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Status bar provides contextual help and operation feedback
        # SUNKEN relief gives it a recessed appearance (standard for status bars)
        # anchor=W means text is left-aligned within the label
        self.status_bar = ttk.Label(
            self.root,
            text="Ready - Start with Page 1 to detect targets",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_page1_target_detection(self):
        """
        Create Page 1: Target Detection (Step 0).

        Purpose:
        --------
        This page allows users to:
          1. Load trial data from various folder structures
          2. Browse frames to find good target visibility
          3. Detect targets (cups/objects) using YOLO
          4. Save detected target positions for later analysis

        UI Structure:
        -------------
          ┌─────────────────────────────────┐
          │ Header with workflow instructions│
          │ ─────────────────────────────────│
          │ [Embedded DataLoaderUI]         │
          │   - Browse/load controls        │
          │   - Image displays              │
          │   - Detection buttons           │
          └─────────────────────────────────┘

        Implementation Notes:
        ---------------------
          - Uses FrameWrapper to embed standalone UI
          - Header added outside embedded UI for consistency
          - Emoji in tab label for visual identification (📍)
          - Instructions guide user through 3-step workflow
        """
        # Create frame container for this page
        page1_frame = ttk.Frame(self.notebook)

        # Add to notebook with emoji label for visual identification
        # 📍 = pin/location (represents target detection)
        self.notebook.add(page1_frame, text="📍 Page 1: Target Detection")

        # Add header section with page title and workflow instructions
        header_frame = ttk.Frame(page1_frame)
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        # Page title with visual emphasis (larger font, blue color)
        ttk.Label(
            header_frame,
            text="📍 Page 1: Target Detection",
            font=("Arial", 14, "bold"),
            foreground="blue"
        ).pack(anchor=tk.W)

        # Workflow instructions - shows user the 3 steps
        # Uses arrow (→) to indicate sequence
        # Gray color = helper text (less prominent than title)
        ttk.Label(
            header_frame,
            text="1. Load trial data  →  2. Detect and save targets  →  3. Switch to Page 2 for skeleton processing",
            font=("Arial", 10),
            foreground="gray"
        ).pack(anchor=tk.W, pady=2)

        # Visual separator between header and content
        # HORIZONTAL line spanning full width
        ttk.Separator(page1_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Create wrapper frame that mimics a Tk root window
        # This allows DataLoaderUI (designed for standalone use) to be embedded
        wrapper_frame = FrameWrapper(page1_frame)

        # Embed the actual DataLoaderUI
        # It thinks wrapper_frame is a root window due to FrameWrapper's interface
        # This is where all the Page 1 functionality lives
        self.page1_ui = DataLoaderUI(wrapper_frame)

    def create_page2_skeleton_processing(self):
        """
        Create Page 2: Skeleton Processing (Step 2).

        Purpose:
        --------
        This page allows users to:
          1. Load trial (auto-loaded from Page 1 or manual)
          2. Configure detection settings (model, confidence, subjects)
          3. Process frames to extract skeletons
          4. Visualize results in 2D and 3D
          5. Review detection info and navigate results

        UI Structure:
        -------------
          ┌──────────────────────────────────┐
          │ Header with workflow instructions │
          │ ──────────────────────────────────│
          │ [Embedded SkeletonExtractorUI]   │
          │   ├─ Left: Controls & 2D view    │
          │   └─ Right: 3D visualization     │
          └──────────────────────────────────┘

        Implementation Notes:
        ---------------------
          - Similar structure to Page 1 for consistency
          - Auto-loads trial from Page 1 (see on_tab_changed)
          - Green color theme (vs blue for Page 1)
          - 🦴 emoji represents skeleton/bones
        """
        # Create frame container for this page
        page2_frame = ttk.Frame(self.notebook)

        # Add to notebook with emoji label
        # 🦴 = bone (represents skeleton detection)
        self.notebook.add(page2_frame, text="🦴 Page 2: Skeleton Processing")

        # Add header section matching Page 1 style
        header_frame = ttk.Frame(page2_frame)
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        # Page title (green to differentiate from Page 1's blue)
        ttk.Label(
            header_frame,
            text="🦴 Page 2: Skeleton Processing & 3D Visualization",
            font=("Arial", 14, "bold"),
            foreground="green"
        ).pack(anchor=tk.W)

        # Workflow instructions for this page
        # Note: Step 1 says "Load trial" but it's actually auto-loaded
        # This is intentional - manual loading is still possible
        ttk.Label(
            header_frame,
            text="1. Load trial  →  2. Process all frames (auto-saves)  →  3. View 3D skeleton with targets",
            font=("Arial", 10),
            foreground="gray"
        ).pack(anchor=tk.W, pady=2)

        # Visual separator
        ttk.Separator(page2_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Create wrapper frame for embedding
        wrapper_frame = FrameWrapper(page2_frame)

        # Embed SkeletonExtractorUI
        # This is where all the Page 2 functionality lives
        self.page2_ui = SkeletonExtractorUI(wrapper_frame)

    def on_tab_changed(self, event):
        """
        Handle tab switching events - implements auto-loading workflow.

        This is the critical function that makes the two-page workflow seamless.

        Event Flow:
        -----------
          1. User clicks on a different tab
          2. Tkinter fires <<NotebookTabChanged>> event
          3. This handler checks which tab is now active
          4. If Page 2 is now active AND Page 1 has a loaded trial:
             → Automatically load that trial in Page 2
          5. Update status bar with contextual message

        Why Auto-Loading?
        -----------------
          Without: User loads trial in Page 1, switches to Page 2,
                   has to browse and re-load the same trial
          With:    User loads trial in Page 1, switches to Page 2,
                   trial appears automatically - ready to process

        Implementation Details:
        -----------------------
          - Uses notebook.index() and notebook.select() to identify current tab
          - Tab 0 = Page 1 (Target Detection)
          - Tab 1 = Page 2 (Skeleton Processing)
          - Checks hasattr() before accessing trial_input_path (defensive)
          - Always reloads even if already loaded (picks up changes like trimming)
          - Debug prints help troubleshoot auto-loading issues

        Args:
            event: Tkinter event object (automatically passed by bind())
        """
        # Get index of currently selected tab (0 = Page 1, 1 = Page 2)
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:
            # Switched to Page 1 (Target Detection)
            # Update status bar with helpful hint about what to do
            self.status_bar.config(text="Page 1: Load trial and detect targets")

        elif current_tab == 1:
            # Switched to Page 2 (Skeleton Processing)

            # Debug output to console (helps troubleshoot auto-loading)
            # Useful when users report "trial didn't auto-load"
            print(f"\n🔍 Debug: Checking for trial from Page 1...")
            print(f"   hasattr trial_input_path: {hasattr(self.page1_ui, 'trial_input_path')}")
            if hasattr(self.page1_ui, 'trial_input_path'):
                print(f"   trial_input_path value: {self.page1_ui.trial_input_path}")

            # Check if Page 1 has a loaded trial
            # trial_input_path is set by DataLoaderUI when trial is successfully loaded
            if hasattr(self.page1_ui, 'trial_input_path') and self.page1_ui.trial_input_path:
                trial_path = self.page1_ui.trial_input_path

                # Always reload to pick up any changes from Page 1
                # Example: User might have trimmed frames, we want latest state
                print(f"\n{'='*60}")
                print(f"🔄 Auto-loading trial from Page 1: {trial_path}")
                print(f"{'='*60}\n")

                # Perform the actual loading (see load_trial_in_page2)
                self.load_trial_in_page2(trial_path)

                # Update status bar to confirm auto-load
                # Shows trial name so user knows which one was loaded
                self.status_bar.config(
                    text=f"Page 2: Auto-loaded '{trial_path.name}' from Page 1 - Ready to process"
                )
            else:
                # No trial loaded in Page 1 (user went directly to Page 2)
                # Provide helpful instruction
                self.status_bar.config(
                    text="Page 2: Load trial to process skeleton and view 3D visualization"
                )

    def load_trial_in_page2(self, trial_path):
        """
        Load a trial into Page 2's skeleton extractor.

        This function performs all necessary initialization to prepare Page 2
        for skeleton processing. It's called by on_tab_changed() during auto-loading.

        Loading Sequence:
        -----------------
          1. Set trial path in Page 2 UI
          2. Load original path mapping (for result syncing)
          3. Verify color folder exists
          4. Load all color image paths
          5. Auto-detect camera intrinsics from image resolution
          6. Update UI controls (slider, labels, etc.)
          7. Load ground plane transform (if available)
          8. Load target detections (if available from Page 1)
          9. Display first frame
         10. Update status label
         11. Print confirmation to console

        Camera Intrinsic Auto-Detection:
        --------------------------------
        We detect fx, fy, cx, cy based on image resolution:
          - 640×480:   fx=fy=615.0,    cx=320,  cy=240   (VGA)
          - 1280×720:  fx=fy=922.5,    cx=640,  cy=360   (HD)
          - 1920×1080: fx=fy=1383.75,  cx=960,  cy=540   (Full HD)
          - Other:     fx=fy=w*0.9,    cx=w/2,  cy=h/2   (estimate)

        These are standard values for typical depth cameras (e.g., RealSense).

        Error Handling:
        ---------------
        If loading fails (e.g., missing folder, corrupt images):
          - Exception is caught and logged
          - Full traceback printed to console
          - UI remains in previous state
          - User can try loading different trial

        Args:
            trial_path: Path object pointing to trial_input/<trial>/<camera>/
        """
        try:
            # Step 1: Set the trial path in Page 2 UI state
            self.page2_ui.trial_path = trial_path

            # Step 2: Load original path from metadata
            # This enables syncing results back to the original data folder
            # (not just trial_output/)
            self.page2_ui.load_original_path_from_config(trial_path)

            # Step 3: Check for color folder
            # All trials must have a color/ subdirectory with frame images
            color_folder = trial_path / "color"
            if not color_folder.exists():
                print(f"⚠️  No 'color' folder found in {trial_path}")
                return

            # Step 4: Load all color image paths
            # Images are named frame_NNNNNN.png (6-digit zero-padded)
            # sorted() ensures frame order (frame_000001, frame_000002, ...)
            self.page2_ui.color_images = sorted(color_folder.glob("frame_*.png"))
            self.page2_ui.total_frames = len(self.page2_ui.color_images)

            if self.page2_ui.total_frames == 0:
                print(f"⚠️  No frame images found in color folder")
                return

            # Step 5: Auto-detect camera intrinsics from first frame's resolution
            import cv2
            sample_img = cv2.imread(str(self.page2_ui.color_images[0]))
            h, w = sample_img.shape[:2]

            # Match common resolutions to standard intrinsics
            if w == 640 and h == 480:
                # VGA resolution (standard for older depth cameras)
                self.page2_ui.fx = self.page2_ui.fy = 615.0
                self.page2_ui.cx = 320.0
                self.page2_ui.cy = 240.0
            elif w == 1280 and h == 720:
                # HD resolution (720p)
                self.page2_ui.fx = self.page2_ui.fy = 922.5
                self.page2_ui.cx = 640.0
                self.page2_ui.cy = 360.0
            elif w == 1920 and h == 1080:
                # Full HD resolution (1080p) - most common
                self.page2_ui.fx = self.page2_ui.fy = 1383.75
                self.page2_ui.cx = 960.0
                self.page2_ui.cy = 540.0
            else:
                # Unknown resolution - use heuristic estimate
                # Focal length ≈ 90% of image width is reasonable for typical cameras
                self.page2_ui.fx = self.page2_ui.fy = w * 0.9
                self.page2_ui.cx = w / 2.0
                self.page2_ui.cy = h / 2.0

            # Step 6: Update UI controls
            # Trial label shows which trial is loaded (black = success)
            self.page2_ui.trial_label.config(
                text=f"{trial_path.name}",
                foreground="black"
            )

            # Configure frame slider to match frame count
            # Range: 1 to total_frames
            self.page2_ui.frame_scale.config(to=self.page2_ui.total_frames)

            # Reset to first frame
            self.page2_ui.current_frame = 1
            self.page2_ui.frame_var.set(1)

            # Clear any previous results
            # Prevents confusion from mixing results of different trials
            self.page2_ui.results = {}

            # Step 7 & 8: Load ground plane and targets (if available)
            # Ground plane: Needed for 3D visualization coordinate system
            # Targets: Detected in Page 1, used for pointing analysis
            self.page2_ui.load_ground_plane_and_targets()

            # Step 9: Load and display first frame
            self.page2_ui.load_frame(1)

            # Step 10: Update status label
            self.page2_ui.status_label.config(
                text=f"Auto-loaded {self.page2_ui.total_frames} frames from Page 1",
                foreground="green"
            )

            # Step 11: Print confirmation to console (helpful for debugging)
            print(f"✅ Successfully loaded trial: {trial_path.name}")
            print(f"   - {self.page2_ui.total_frames} frames")
            print(f"   - Ground plane: {'✓' if self.page2_ui.ground_plane_transform is not None else '✗'}")
            print(f"   - Targets: {'✓' if self.page2_ui.targets is not None else '✗'}")

        except Exception as e:
            # Something went wrong - log error and show traceback
            print(f"❌ Error loading trial in Page 2: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main entry point for the application.

    Creates the Tkinter root window, initializes the MainUI, and starts
    the event loop.

    Usage:
        python main_ui.py

    The window will remain open until user closes it or presses Ctrl+C.
    """
    root = tk.Tk()
    app = MainUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
