import cv2
import os
import numpy as np
import argparse

# Function to create a white blank image with the same width and height as needed
def create_blank_image(width, height):
    return 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)

# Resize the plot image to fit inside the frame image dimensions while maintaining the aspect ratio
def resize_and_pad_image(image, target_width, target_height):
    height, width, _ = image.shape

    # Calculate the aspect ratio of the plot image
    aspect_ratio = width / height

    # Determine whether to scale based on width or height
    if target_width / aspect_ratio <= target_height:
        # Scale based on width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Scale based on height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image to the new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a blank white image of target dimensions
    padded_image = create_blank_image(target_width, target_height)

    # Center the resized image on the blank canvas
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image

def merge_images_to_video(image_dir, output_video_path):
    # Get list of all frame (f) images and plot images
    frame_images = sorted([img for img in os.listdir(image_dir) if img.startswith('f') and img.endswith('.png')])
    plot_images = sorted([img for img in os.listdir(image_dir) if img.startswith('plot_p') and img.endswith('.png')])

    # Check if there are no frame images
    if not frame_images:
        print("No frame images found in the directory.")
        return

    # Load first frame to get the dimensions for video creation
    first_frame_image = cv2.imread(os.path.join(image_dir, frame_images[0]))
    frame_height, frame_width, _ = first_frame_image.shape

    # Set the video writer with resolution based on frame image dimensions
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, 2 * frame_height))

    for frame_file in frame_images:
        frame_image = cv2.imread(os.path.join(image_dir, frame_file))

        # Find corresponding plot image
        frame_number = int(frame_file[1:-4])  # Extract the frame number from 'f<number>.png'
        plot_file = f'plot_p{frame_number}.png'
        plot_image_path = os.path.join(image_dir, plot_file)

        if os.path.exists(plot_image_path):
            plot_image = cv2.imread(plot_image_path)
            # Resize the plot image to fit inside the frame dimensions (based on the smaller dimension)
            plot_image = resize_and_pad_image(plot_image, frame_width, frame_height)
        else:
            # Create a blank white image if the plot is not found
            plot_image = create_blank_image(frame_width, frame_height)

        # Vertically stack the frame and the plot image
        stitched_image = np.vstack((frame_image, plot_image))

        # Write the stitched image to the video
        video_writer.write(stitched_image)

    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge frame and plot images into a video.")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing frame and plot images")
    parser.add_argument('--output_video', type=str, required=True, help="Path to save the output video")
    # parser.add_argument('--fps', type=int, default=30, help="Frames per second for the output video")

    args = parser.parse_args()

    merge_images_to_video(args.image_dir, args.output_video)