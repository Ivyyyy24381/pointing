#!/usr/bin/env python3
import os, cv2, time, threading, queue
import numpy as np
import pyrealsense2 as rs
from datetime import datetime

CAMERA_SERIALS = {
    "cam1": "246322303830",
    "cam2": "829212071682",
    "cam3": "341222300776"
}
RESOLUTION = (640, 480)
FPS = 15

OUTPUT_DIR = "/home/h2r/Documents/"
PREFERRED = "/media/h2r/TOSHIBA EXT/"
if os.path.exists(PREFERRED):
    OUTPUT_DIR = PREFERRED

# ---- globals ----
recording = False
trial_id = 0
frame_id = 0
start_time = None
current_trial_dir = None

frame_buffers, frame_locks, record_queues = {}, {}, {}
frame_id_lock = threading.Lock()
camera_barrier = threading.Barrier(len(CAMERA_SERIALS))

# video writers per cam (created on start, released on stop)
video_writers_color = {}   # cam_id -> cv2.VideoWriter
video_writers_depth = {}   # cam_id -> cv2.VideoWriter
writers_lock = threading.Lock()

# ---- helpers ----
def fmt_elapsed(start_t):
    if not start_t:
        return "00:00"
    e = time.time() - start_t
    return f"{int(e//60):02d}:{int(e%60):02d}"

def draw_overlay(frame, cam_label, rec, trial_id, frame_id, elapsed_txt):
    state_color = (0,0,255) if rec else (0,255,0)
    cv2.putText(frame, f"{cam_label} - {'REC' if rec else 'IDLE'}",
                (10,26), cv2.FONT_HERSHEY_SIMPLEX,0.8,state_color,2)
    cv2.putText(frame, f"T{trial_id} F{frame_id:06d} {elapsed_txt}",
                (10,52), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    return frame

def ensure_dirs(base, cams):
    for cid in cams:
        os.makedirs(os.path.join(base,cid,"color"),exist_ok=True)
        os.makedirs(os.path.join(base,cid,"depth"),exist_ok=True)

def create_pipeline(serial):
    p, c = rs.pipeline(), rs.config()
    c.enable_device(serial)
    c.enable_stream(rs.stream.depth,*RESOLUTION,rs.format.z16,FPS)
    c.enable_stream(rs.stream.color,*RESOLUTION,rs.format.bgr8,FPS)
    return p,c

def open_writers_for_trial(trial_dir, cam_id):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    color_path = os.path.join(trial_dir, cam_id, f"{cam_id}_color.mp4")
    depth_path = os.path.join(trial_dir, cam_id, f"{cam_id}_depth.mp4")
    vw_color = cv2.VideoWriter(color_path, fourcc, FPS, RESOLUTION)
    vw_depth = cv2.VideoWriter(depth_path, fourcc, FPS, RESOLUTION)
    return vw_color, vw_depth

def close_writers(cam_id):
    with writers_lock:
        if cam_id in video_writers_color and video_writers_color[cam_id]:
            video_writers_color[cam_id].release()
        if cam_id in video_writers_depth and video_writers_depth[cam_id]:
            video_writers_depth[cam_id].release()
        video_writers_color[cam_id] = None
        video_writers_depth[cam_id] = None

# ---- workers ----
def saver_worker(q, cam_id):
    while True:
        item = q.get()
        if item is None: break
        fid, color_img, depth_img, trial_dir = item
        cv2.imwrite(os.path.join(trial_dir,cam_id,"color",f"frame_{fid:06d}.png"), color_img)
        np.save(os.path.join(trial_dir,cam_id,"depth",f"frame_{fid:06d}.npy"), depth_img)

        # also append to videos if writers are open
        with writers_lock:
            vw_c = video_writers_color.get(cam_id)
            vw_d = video_writers_depth.get(cam_id)
        if vw_c is not None:
            vw_c.write(color_img)  # BGR
        if vw_d is not None:
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            vw_d.write(depth_vis)

        q.task_done()

def cam_worker(cam_id, serial):
    global recording, frame_id, current_trial_dir
    pipe, cfg = create_pipeline(serial)
    align = rs.align(rs.stream.color)
    q = queue.Queue(maxsize=120)
    record_queues[cam_id]=q; frame_locks[cam_id]=threading.Lock(); frame_buffers[cam_id]=(None,None)
    pipe.start(cfg)
    threading.Thread(target=saver_worker,args=(q,cam_id),daemon=True).start()

    last_rec = False
    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            d,c = aligned.get_depth_frame(),aligned.get_color_frame()
            if not d or not c: continue
            color_np = np.asanyarray(c.get_data()); depth_np = np.asanyarray(d.get_data())

            with frame_locks[cam_id]:
                frame_buffers[cam_id] = (color_np.copy(), depth_np.copy())

            # detect start/stop edges to open/close writers
            if recording and not last_rec:
                # open writers for this cam
                with writers_lock:
                    vw_c, vw_d = open_writers_for_trial(current_trial_dir, cam_id)
                    video_writers_color[cam_id] = vw_c
                    video_writers_depth[cam_id] = vw_d
            elif not recording and last_rec:
                close_writers(cam_id)
            last_rec = recording

            if recording:
                camera_barrier.wait()
                with frame_id_lock: fid = frame_id
                if not q.full():
                    q.put((fid, color_np.copy(), depth_np.copy(), current_trial_dir))
                camera_barrier.wait()
            time.sleep(1/FPS)
    finally:
        pipe.stop()
        q.put(None)
        close_writers(cam_id)

# ---- main ----
def main():
    global recording, frame_id, trial_id, start_time, current_trial_dir
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir=os.path.join(OUTPUT_DIR,f"rs_session_{timestamp}")
    os.makedirs(session_dir,exist_ok=True)

    for cid,ser in CAMERA_SERIALS.items():
        threading.Thread(target=cam_worker,args=(cid,ser),daemon=True).start()
    while len(frame_buffers)<len(CAMERA_SERIALS): time.sleep(0.5)

    # nicer window
    win_name = "Top: RGB | Bottom: Depth"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1600, 900)

    print("Controls: s=start, e=stop, q=quit")
    try:
        while True:
            elapsed_txt=fmt_elapsed(start_time) if recording else "00:00"
            rgb_row, depth_row=[],[]
            for cid in CAMERA_SERIALS:
                with frame_locks[cid]: c_np,d_np=frame_buffers.get(cid,(None,None))
                if c_np is None: continue
                rgb=cv2.resize(c_np,(RESOLUTION[0]//2,RESOLUTION[1]//2))
                depth_vis=cv2.applyColorMap(cv2.convertScaleAbs(d_np,alpha=0.03),cv2.COLORMAP_JET)
                depth_vis=cv2.resize(depth_vis,(RESOLUTION[0]//2,RESOLUTION[1]//2))
                with frame_id_lock: fid=frame_id
                rgb_row.append(draw_overlay(rgb,cid,recording,trial_id,fid,elapsed_txt))
                depth_row.append(draw_overlay(depth_vis,cid,recording,trial_id,fid,elapsed_txt))
            if rgb_row and depth_row:
                mosaic = np.vstack((np.hstack(rgb_row), np.hstack(depth_row)))
                cv2.imshow(win_name, mosaic)

            k=cv2.waitKey(1)&0xFF
            if k==ord('q') or k==27: break
            elif k==ord('s') and not recording:
                current_trial_dir=os.path.join(session_dir,f"trial_{trial_id}")
                ensure_dirs(current_trial_dir,CAMERA_SERIALS)
                recording=True; start_time=time.time()
                with frame_id_lock: frame_id=0
                print(f"▶️ Trial {trial_id} started")
            elif k==ord('e') and recording:
                recording=False; trial_id+=1
                print("⏹️ Recording stopped")
            if recording:
                with frame_id_lock: frame_id+=1
            time.sleep(1/FPS)
    finally:
        for q in record_queues.values(): q.put(None)
        # ensure all writers closed
        for cid in CAMERA_SERIALS: close_writers(cid)
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
