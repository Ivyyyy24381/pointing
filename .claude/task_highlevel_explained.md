# 👆 Pointing Task — High-Level Overview

## 🧠 Study Goal

This project supports a study on **collaborative search between humans and different subjects** (e.g., dogs, babies). In each scenario, a **human provides pointing gestures**, and we analyze how the **subject responds and collaborates** in searching for objects in a shared environment.

Two main scenarios:
- 🐶 Human points for a **dog**
- 👶 Human points for a **baby**

The primary research question is:  
> **How do human pointing behaviors and subject responses interact during collaborative search?**

---

## 🧪 Data Collection Setup

We use **Intel RealSense** cameras to collect **RGB and depth streams** for each trial.

### Folder Structure

- Each **subject** (dog or baby) has its own top-level folder.
- Each **trial** corresponds to one session where a human points for that subject.
- Multiple **cameras** can be used per trial.
- Both **color** and **depth** frames are stored in `.png` format with aligned filenames.

---

## 🧭 Task Overview

The goal of this codebase is to provide a **lightweight web interface** that allows a researcher to:

1. **Navigate trials** easily across subjects and cameras  
2. **Visualize color and depth streams** side by side  
3. **Extract structured information** (e.g., human pointing pose, subject position) for future analysis  
4. **Mark key frames or intervals** corresponding to pointing events and subject responses  
5. Export this extracted data in a structured format (e.g., CSV or JSON) for downstream modeling and statistical analysis.

---

## 🧰 System Components

| Component                | Description |
|---------------------------|-------------|
| **Frontend (Web UI)**     | Simple browser-based interface for browsing trials, playing back frames, annotating, and extracting relevant information. |
| **Backend (Flask)**       | Serves the UI, reads trial metadata, streams frames, and stores annotations. |
| **Processing Scripts**    | Extract human skeletons (e.g., via MediaPipe), detect subject position, and compute relevant geometric features (e.g., gaze, pointing direction, distance). |
| **Data Export**           | Saves extracted annotations and analysis-ready features to structured files for later use (POMDP, gesture analysis, behavioral analysis, etc.). |

---

## 🌐 Example UI Features

- **Trial Selector**: Dropdown menu to pick subject → trial → camera.  
- **Frame Slider**: Scrub through frames; visualize color and depth together.  
- **Annotation Tools**: Mark human pointing frames, subject response intervals.  
- **Extraction Buttons**: Run MediaPipe on selected frames to extract human skeleton; run subject detection.  
- **Export Panel**: Save results as `trial_annotations.json` or `.csv`.

---

## 🚀 Planned Workflow

1. Collect data from RealSense cameras and organize into the standardized folder structure.  
2. Use the **UI tool** to browse each trial and mark relevant events.  
3. Run automatic extraction pipelines for pointing gestures and subject locations.  
4. Export structured data for gesture analysis, subject behavior modeling, or integration into larger planning frameworks (e.g., POMDP).

---

## 📌 Next Steps

- [ ] Build minimal Flask backend to serve trial list and images  
- [ ] Implement frontend with React/Vue or vanilla JS to display synchronized color and depth  
- [ ] Integrate MediaPipe-based human skeleton extraction  
- [ ] Add subject detection module (dog/baby keypoints or bounding boxes)  
- [ ] Design JSON schema for saving annotations  
- [ ] Add CSV export for easy statistical analysis

---