import pyrealsense2 as rs

CAMERA_SERIALS = {
    "cam1": "246322303830",
    "cam2": "829212071682",
    "cam3": "341222300776"
}

def get_intrinsics(serial):
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)
        stream = profile.get_stream(rs.stream.color)
        video_profile = stream.as_video_stream_profile()
        intr = video_profile.get_intrinsics()
        pipeline.stop()
        return intr
    except Exception as e:
        return None

if __name__ == "__main__":
    for name, serial in CAMERA_SERIALS.items():
        print(f"\n=== {name} ({serial}) ===")
        intr = get_intrinsics(serial)
        if intr is None:
            print("[Error] Could not get intrinsics. Is the camera busy or disconnected?")
            continue

        print(f"Resolution : {intr.width}x{intr.height}")
        print(f"fx         : {intr.fx}")
        print(f"fy         : {intr.fy}")
        print(f"ppx        : {intr.ppx}")
        print(f"ppy        : {intr.ppy}")
        print(f"model      : {intr.model}")
        print(f"coeffs     : {intr.coeffs}")
