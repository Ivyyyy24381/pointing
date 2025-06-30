import subprocess
import time
import pyrealsense2 as rs

# List all connected RealSense devices
def get_realsense_serials():
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    return serials

# Generate roslaunch command for each camera with unique namespace
def launch_camera(serial, namespace):
    return [
        'roslaunch', 'realsense2_camera', 'rs_camera.launch',
        f'serial_no:={serial}',
        f'unite_imu_method:=""',
        f'initial_reset:=true',
        f'camera:=/{namespace}'
    ]

# Generate rosbag record command
def generate_rosbag_command(serials):
    topics = []
    for idx, _ in enumerate(serials):
        ns = f"/cam{idx+1}"
        topics.extend([
            f"{ns}/color/image_raw",
            f"{ns}/depth/image_raw",
            f"{ns}/aligned_depth_to_color/image_raw",
        ])
    return ['rosbag', 'record', '-o', 'multi_cam.bag'] + topics

if __name__ == "__main__":
    # Step 1: Detect Cameras
    serials = get_realsense_serials()
    if len(serials) < 3:
        print("❌ Fewer than 3 RealSense devices detected.")
        exit(1)

    print("✔️ Detected RealSense cameras:", serials)

    # Step 2: Launch each camera in a subprocess
    launch_processes = []
    for i, serial in enumerate(serials[:3]):
        ns = f"cam{i+1}"
        print(f"🚀 Launching camera {i+1} with serial {serial}...")
        proc = subprocess.Popen(launch_camera(serial, ns))
        launch_processes.append(proc)
        time.sleep(3)  # Give each camera some time to initialize

    # Step 3: Start rosbag recording
    print("📹 Starting rosbag recording...")
    rosbag_proc = subprocess.Popen(generate_rosbag_command(serials[:3]))

    try:
        print("🎥 Recording... Press Ctrl+C to stop.")
        rosbag_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all processes...")
        rosbag_proc.terminate()
        for proc in launch_processes:
            proc.terminate()
        print("✅ Done.")
        
        

    """
    <launch>
  <!-- Camera 1 -->
  <group ns="cam1">
    <param name="serial_no" type="string" value="123456789" />
    <include file="$(find realsense2_camera)/launch/rs_camera.launch" />
  </group>

  <!-- Camera 2 -->
  <group ns="cam2">
    <param name="serial_no" type="string" value="987654321" />
    <include file="$(find realsense2_camera)/launch/rs_camera.launch" />
  </group>

  <!-- Camera 3 -->
  <group ns="cam3">
    <param name="serial_no" type="string" value="1122334455" />
    <include file="$(find realsense2_camera)/launch/rs_camera.launch" />
  </group>
</launch>
rosbag record -o multi_cam.bag /cam1/color/image_raw /cam1/depth/image_raw /cam1/aligned_depth_to_color/image_raw \
                                  /cam2/color/image_raw /cam2/depth/image_raw /cam2/aligned_depth_to_color/image_raw \
                                  /cam3/color/image_raw /cam3/depth/image_raw /cam3/aligned_depth_to_color/image_raw
    """