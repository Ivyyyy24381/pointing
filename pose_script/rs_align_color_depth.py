import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load aligned frame from bag
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("/home/xhe71/Desktop/dog_data/BDL244_Hannah/BDL244_Hannah_PVP_019.bag")
profile = pipeline.start(config)

align = rs.align(rs.stream.color)
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
pipeline.stop()

depth_frame = aligned_frames.get_depth_frame()
color_frame = aligned_frames.get_color_frame()

depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

depth_meters = depth_image.astype(np.float32) / 1000.0
depth_meters[depth_image == 0] = np.nan

# Plot setup
fig, (ax_color, ax_depth) = plt.subplots(1, 2, figsize=(14, 6))
im_color = ax_color.imshow(color_image)
im_depth = ax_depth.imshow(depth_meters, cmap='gray')
plt.colorbar(im_depth, ax=ax_depth, fraction=0.046, pad=0.04)

ax_color.set_title("Click on Aligned Color Image")
ax_depth.set_title("Corresponding Location on Depth")
for ax in (ax_color, ax_depth):
    ax.axis('off')

# Rectangle that will be drawn on both images
box_color = patches.Rectangle((0, 0), 20, 20, linewidth=2, edgecolor='lime', facecolor='none')
box_depth = patches.Rectangle((0, 0), 20, 20, linewidth=2, edgecolor='red', facecolor='none')
ax_color.add_patch(box_color)
ax_depth.add_patch(box_depth)

# On click callback
def onclick(event):
    if event.inaxes == ax_color:
        x, y = int(event.xdata), int(event.ydata)
        box_color.set_xy((x - 10, y - 10))
        box_depth.set_xy((x - 10, y - 10))
        fig.canvas.draw_idle()
        
        # Print depth at that point
        if 0 <= x < depth_meters.shape[1] and 0 <= y < depth_meters.shape[0]:
            z = depth_meters[y, x]
            if np.isnan(z):
                print(f"Clicked ({x}, {y}) → Depth: INVALID")
            else:
                print(f"Clicked ({x}, {y}) → Depth: {z:.3f} meters")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()
