import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import threading
import queue
from datetime import datetime

CAMERA_SERIALS = {
    "cam1": "246322303830",
    "cam2": "829212071682",
    "cam3": "341222300776"
}

OUTPUT_DIR = "/home/h2r/Documents/"
preferred_directory = "/media/h2r/TOSHIBA EXT/"
if os.path.exists(preferred_directory):
    OUTPUT_DIR = preferred_directory
RESOLUTION = (640, 480)
FPS = 15
SAVE_EVERY_N = 1

recording = False
trial_id = 0
frame_id = 0
current_trial_dir = None
frame_id_lock = threading.Lock()
frame_buffers = {}
frame_locks = {}
record_queues = {}

# Barrier for frame sync across threads
camera_barrier = threading.Barrier(len(CAMERA_SERIALS))


def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
    return pipeline, config


def ensure_dirs(base_dir, camera_ids):
    for cid in camera_ids:
        os.makedirs(os.path.join(base_dir, cid, "color"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, cid, "depth"), exist_ok=True)


def saver_worker(q, cam_id):
    while True:
        item = q.get()
        if item is None:
            break
        fid, color_img, depth_img, trial_dir = item
        color_path = os.path.join(trial_dir, cam_id, "color", f"frame_{fid:06d}.png")
        depth_path = os.path.join(trial_dir, cam_id, "depth", f"frame_{fid:06d}.npy")
        cv2.imwrite(color_path, color_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        np.save(depth_path, depth_img)
        q.task_done()


def cam_worker(cam_id, serial):
    global recording, frame_id, current_trial_dir
    pipeline, config = create_pipeline(serial)
    align = rs.align(rs.stream.color)
    q = queue.Queue(maxsize=30)
    record_queues[cam_id] = q
    frame_locks[cam_id] = threading.Lock()
    frame_buffers[cam_id] = (None, None)
    pipeline.start(config)
    saver_thread = threading.Thread(target=saver_worker, args=(q, cam_id))
    saver_thread.start()
    print(f"📷 {cam_id} thread started.")

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms = 50000)
            if not frames:
                continue
            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue

            color_np = np.asanyarray(color.get_data())
            depth_np = np.asanyarray(depth.get_data())

            with frame_locks[cam_id]:
                frame_buffers[cam_id] = (color_np.copy(), depth_np.copy())

            if recording:
                camera_barrier.wait()  # sync point
                with frame_id_lock:
                    fid = frame_id
                if not q.full():
                    q.put((fid, color_np.copy(), depth_np.copy(), current_trial_dir))
                camera_barrier.wait()  # sync again before frame_id advances
            time.sleep(1 / FPS)
    finally:
        pipeline.stop()
        q.put(None)
        saver_thread.join()


def draw_ui(frame, is_recording, label_text):
    label = "Recording..." if is_recording else "Idle"
    color = (0, 0, 255) if is_recording else (0, 255, 0)
    return cv2.putText(frame, f"{label_text} - {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def verify_all_cameras_connected():
    ctx = rs.context()
    connected = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    print(f"🧩 Connected devices: {connected}")
    missing = [sid for sid in CAMERA_SERIALS.values() if sid not in connected]
    if missing:
        print(f"❌ Missing RealSense cameras: {missing}")
        return False
    return True


def main():
    global recording, frame_id, trial_id, current_trial_dir

    if not verify_all_cameras_connected():
        return

    print("🔧 Initializing...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"rs_session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    cam_threads = []
    for cam_id, serial in CAMERA_SERIALS.items():
        t = threading.Thread(target=cam_worker, args=(cam_id, serial), daemon=True)
        t.start()
        cam_threads.append(t)

    print("🟢 Press 's' to start, 'e' to stop, 'q' to quit.")

    while len(frame_buffers) < len(CAMERA_SERIALS):
        print("⏳ Waiting for camera threads to initialize...")
        time.sleep(0.5)

    try:
        while True:
            rgb_row, depth_row = [], []
            for cam_id in CAMERA_SERIALS:
                with frame_locks[cam_id]:
                    color_np, depth_np = frame_buffers.get(cam_id, (None, None))
                if color_np is None or depth_np is None:
                    continue
                rgb_small = cv2.resize(color_np, (RESOLUTION[0] // 2, RESOLUTION[1] // 2))
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=0.03), cv2.COLORMAP_JET)
                depth_small = cv2.resize(depth_colormap, (RESOLUTION[0] // 2, RESOLUTION[1] // 2))

                labeled_rgb = draw_ui(rgb_small, recording, cam_id)
                labeled_depth = draw_ui(depth_small, recording, cam_id)

                rgb_row.append(labeled_rgb)
                depth_row.append(labeled_depth)

            if rgb_row and depth_row:
                top = np.hstack(rgb_row)
                bottom = np.hstack(depth_row)
                full_view = np.vstack((top, bottom))
                cv2.imshow("Top: RGB | Bottom: Depth", full_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                trial_id += 1
                current_trial_dir = os.path.join(session_dir, f"trial_{trial_id}")
                ensure_dirs(current_trial_dir, CAMERA_SERIALS.keys())
                recording = True
                print(f"▶️ Recording started... Trial {trial_id}")
            elif key == ord('e'):
                recording = False
                print("⏹️ Recording stopped.")

            if recording:
                with frame_id_lock:
                    frame_id += 1

            time.sleep(1 / FPS)

    finally:
        print("🛑 Shutting down...")
        for cam_id in CAMERA_SERIALS:
            record_queues[cam_id].put(None)
        cv2.destroyAllWindows()
        print("✅ All data saved.")


if __name__ == "__main__":
    main()