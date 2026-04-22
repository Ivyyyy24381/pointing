#!/usr/bin/env python3
"""
Label the baby's bounding box in stuck CCD comprehension trials.

Shows a frame from each stuck trial — draw a box around the baby.
Saves all labels to stuck_trial_labels.json for reprocessing with box prompts.

Usage:
    python label_stuck_trials.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle

OUTPUT_DIR = Path("/home/tigerli/Documents/pointing_data/point_comprehension_CCD_output")
SSD_DIR = Path("/media/tigerli/Extreme SSD/pointing_data/point_comprehension_CCD")
LABELS_FILE = Path(__file__).parent / "stuck_trial_labels.json"

# 13 stuck trials (CCD0495/trial_4 excluded — raw data not on SSD)
STUCK_TRIALS = [
    ("CCD0425_PVPT_009E_side", "trial_3"),
    ("CCD0430_PVPT_014E_side", "trial_7"),
    ("CCD0430_PVPT_014E_side", "trial_8"),
    ("CCD0430_PVPT_014E_side", "trial_9"),
    ("CCD0431_PVPT_0013E_side", "trial_3"),
    ("CCD0431_PVPT_0013E_side", "trial_5"),
    ("CCD0431_PVPT_0013E_side", "trial_6"),
    ("CCD0431_PVPT_0013E_side", "trial_8"),
    ("CCD0444_PVPT_015E_side", "trial_1"),
    ("CCD0444_PVPT_015E_side", "trial_6"),
    ("CCD0444_PVPT_015E_side", "trial_7"),
    ("CCD0444_PVPT_015E_side", "trial_9"),
    ("CCD0540_PVPTC_001", "trial_3"),
]


def find_color_frames(subject, trial):
    """Find color frames for a trial, checking SSD source directories."""
    ssd_subj = SSD_DIR / subject
    if not ssd_subj.is_dir():
        return []

    trial_num = trial.replace("trial_", "")

    candidates = [
        ssd_subj / trial_num / "Color",
        ssd_subj / trial / "Color",
        ssd_subj / trial / "cam1" / "color",
        ssd_subj / trial_num / "cam1" / "color",
    ]

    for inner in ssd_subj.iterdir():
        if inner.is_dir() and not inner.name.startswith("."):
            candidates.extend([
                inner / trial_num / "Color",
                inner / trial / "Color",
                inner / trial / "cam1" / "color",
                inner / trial_num / "cam1" / "color",
            ])

    for color_dir in candidates:
        if color_dir.is_dir():
            pngs = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png" and not f.name.startswith("._")])
            if pngs:
                return pngs

    return []


def label_trial(subject, trial, color_files):
    """Show frame with matplotlib and get user bounding box around baby.
    Returns (bbox_dict, frame_idx) or (None, -1) for skip or (None, -2) for quit.
    bbox_dict has keys: x, y, w, h (pixel coordinates).
    """
    frame_idx = len(color_files) // 3
    img = cv2.imread(str(color_files[frame_idx]))
    if img is None:
        print(f"  Cannot read frame {color_files[frame_idx]}")
        return None, -1

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    state = {
        "frame_idx": frame_idx,
        "result": None,
        "press_pos": None,   # mouse press position
        "bbox": None,        # (x, y, w, h) in pixels
        "rect_patch": None,
        "text_ann": None,
        "dragging": False,
    }

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.15)
    im_handle = ax.imshow(img_rgb)
    ax.set_axis_off()
    title_text = ax.set_title(
        f"{subject} / {trial}  (frame {frame_idx+1}/{len(color_files)})\n"
        f"Click and drag to draw a box around the BABY. Use Prev/Next to browse.",
        fontsize=11
    )

    # Buttons
    ax_prev = plt.axes([0.15, 0.03, 0.12, 0.05])
    ax_next = plt.axes([0.30, 0.03, 0.12, 0.05])
    ax_confirm = plt.axes([0.50, 0.03, 0.12, 0.05])
    ax_skip = plt.axes([0.65, 0.03, 0.12, 0.05])
    ax_quit = plt.axes([0.80, 0.03, 0.12, 0.05])

    btn_prev = Button(ax_prev, "< Prev (A)")
    btn_next = Button(ax_next, "Next (D) >")
    btn_confirm = Button(ax_confirm, "Confirm")
    btn_skip = Button(ax_skip, "Skip (N)")
    btn_quit = Button(ax_quit, "Quit (Q)")

    def _clear_box():
        if state["rect_patch"]:
            state["rect_patch"].remove()
            state["rect_patch"] = None
        if state["text_ann"]:
            state["text_ann"].remove()
            state["text_ann"] = None
        state["bbox"] = None

    def _load_frame(idx):
        idx = max(0, min(len(color_files) - 1, idx))
        state["frame_idx"] = idx
        new_img = cv2.imread(str(color_files[idx]))
        if new_img is not None:
            new_rgb = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            im_handle.set_data(new_rgb)
        title_text.set_text(
            f"{subject} / {trial}  (frame {idx+1}/{len(color_files)})\n"
            f"Click and drag to draw a box around the BABY. Use Prev/Next to browse."
        )
        _clear_box()
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return
        state["press_pos"] = (event.xdata, event.ydata)
        state["dragging"] = True
        _clear_box()

    def on_motion(event):
        if not state["dragging"] or event.inaxes != ax or state["press_pos"] is None:
            return
        x0, y0 = state["press_pos"]
        x1, y1 = event.xdata, event.ydata
        bx = min(x0, x1)
        by = min(y0, y1)
        bw = abs(x1 - x0)
        bh = abs(y1 - y0)

        if state["rect_patch"]:
            state["rect_patch"].remove()
        state["rect_patch"] = ax.add_patch(
            Rectangle((bx, by), bw, bh,
                       linewidth=2, edgecolor='red', facecolor='none')
        )
        fig.canvas.draw_idle()

    def on_release(event):
        if not state["dragging"] or state["press_pos"] is None:
            return
        state["dragging"] = False

        if event.inaxes != ax:
            return

        x0, y0 = state["press_pos"]
        x1, y1 = event.xdata, event.ydata

        bx = int(round(min(x0, x1)))
        by = int(round(min(y0, y1)))
        bw = int(round(abs(x1 - x0)))
        bh = int(round(abs(y1 - y0)))

        # Minimum box size
        if bw < 10 or bh < 10:
            _clear_box()
            fig.canvas.draw_idle()
            return

        # Clamp to image
        bx = max(0, bx)
        by = max(0, by)
        bw = min(bw, img_w - bx)
        bh = min(bh, img_h - by)

        state["bbox"] = (bx, by, bw, bh)

        # Draw final rectangle
        if state["rect_patch"]:
            state["rect_patch"].remove()
        state["rect_patch"] = ax.add_patch(
            Rectangle((bx, by), bw, bh,
                       linewidth=2, edgecolor='red', facecolor=(1, 0, 0, 0.1))
        )

        if state["text_ann"]:
            state["text_ann"].remove()
        state["text_ann"] = ax.annotate(
            f"({bx},{by}) {bw}x{bh}", (bx, by),
            textcoords="offset points", xytext=(0, -14),
            color='red', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
        )
        fig.canvas.draw_idle()

    def on_prev(event):
        _load_frame(state["frame_idx"] - 5)

    def on_next(event):
        _load_frame(state["frame_idx"] + 5)

    def on_confirm(event):
        if state["bbox"]:
            bx, by, bw, bh = state["bbox"]
            bbox_dict = {"x": bx, "y": by, "w": bw, "h": bh}
            state["result"] = (bbox_dict, state["frame_idx"])
            plt.close(fig)

    def on_skip(event):
        state["result"] = (None, -1)
        plt.close(fig)

    def on_quit(event):
        state["result"] = (None, -2)
        plt.close(fig)

    def on_key(event):
        if event.key == 'a':
            on_prev(event)
        elif event.key == 'd':
            on_next(event)
        elif event.key == 'n':
            on_skip(event)
        elif event.key == 'q':
            on_quit(event)
        elif event.key == 'enter':
            on_confirm(event)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_confirm.on_clicked(on_confirm)
    btn_skip.on_clicked(on_skip)
    btn_quit.on_clicked(on_quit)

    plt.show(block=True)

    if state["result"] is not None:
        return state["result"]
    return None, -1


def main():
    # Load existing labels if any
    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} existing labels from {LABELS_FILE.name}")

    print(f"\nLabeling {len(STUCK_TRIALS)} stuck trials")
    print(f"Draw a bounding box around the baby, then press Confirm (or Enter).")
    print(f"Use Prev/Next (or A/D) to browse frames, Skip (N) to skip, Quit (Q) to quit.\n")

    labeled = 0
    skipped = 0

    for subject, trial in STUCK_TRIALS:
        key = f"{subject}/{trial}"
        if key in labels:
            print(f"  {key}: already labeled, skipping")
            labeled += 1
            continue

        color_files = find_color_frames(subject, trial)
        if not color_files:
            print(f"  {key}: NO COLOR FRAMES FOUND, skipping")
            skipped += 1
            continue

        print(f"  {key}: {len(color_files)} frames")
        bbox_dict, frame_idx = label_trial(subject, trial, color_files)

        if frame_idx == -2:  # quit
            print("\nQuitting early.")
            break

        if bbox_dict is None:
            print(f"    Skipped")
            skipped += 1
            continue

        # Read image dimensions
        img = cv2.imread(str(color_files[0]))
        img_h, img_w = img.shape[:2]

        labels[key] = {
            "box_x": bbox_dict["x"],
            "box_y": bbox_dict["y"],
            "box_w": bbox_dict["w"],
            "box_h": bbox_dict["h"],
            "frame_idx": frame_idx,
            "n_frames": len(color_files),
            "img_w": img_w,
            "img_h": img_h,
        }
        labeled += 1
        print(f"    Labeled box ({bbox_dict['x']},{bbox_dict['y']}) "
              f"{bbox_dict['w']}x{bbox_dict['h']} on frame {frame_idx}")

        # Save after each label (in case of crash)
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f, indent=2)

    print(f"\nDone: {labeled} labeled, {skipped} skipped")
    print(f"Labels saved to: {LABELS_FILE}")
    print(f"\nNext: run  python reprocess_stuck_trials.py  to reprocess with box prompts")


if __name__ == "__main__":
    main()
