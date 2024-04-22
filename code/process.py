# for single image that was extracted from rosbag using ____, with stacked color and depth stream. 
import os
# import mediapipe as mp
import cv2
import sys
import math
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import mediapipe as mp
import copy
# import pyrealsense2
# input




# Mediapipe & Rosbag
# 1. Split into color and depth image
def split_image(image, front):
     # Read the stacked image
    stacked_image = cv2.imread(image)
    if stacked_image is None:
       print("error reading %s" % image)
       return
    # Get the dimensions of the image
    height, width = (1440, 1280)

    # Calculate the height of each slice
    slice_height = height // 2
    # image in bgr form, need to convert to rgb for mediapipe
    color_image = stacked_image[0:slice_height, :]
    depth_image =  stacked_image[slice_height:, :]

    # scale to align depth and color image
    if front == 1:
       
      scale_factor = 1.4
      v_shift = 15
      h_shift = 30
    else:
      scale_factor = 1.001
      v_shift = 0
      h_shift = -70

    # Calculate the new dimensions based on the scale factor
    new_height = int(slice_height * scale_factor)
    new_width = int(width * scale_factor)

    # Resize the image
    scaled_image = cv2.resize(depth_image, (new_width, new_height))
    # Calculate the cropping boundaries
    crop_top = (new_height - slice_height -v_shift) // 2
    crop_bottom = new_height - crop_top -v_shift
    crop_left = (new_width - width - h_shift) // 2
    crop_right = new_width - crop_left - h_shift
    # Crop the scaled image from the center
    cropped_image = scaled_image[crop_top:crop_bottom, crop_left:crop_right]
    cropped_depth_image =  cv2.resize(cropped_image, (width, slice_height))
    
    return color_image, cropped_depth_image, width, slice_height



   


# 2. mediapipe get 2D landmark (pixel) of corresponding joints in color image
# in: color_image, depth_image
# out: landmark 2D coordinate
def detect_2d_landmark(color_image):
    # start mediapipe detection
   # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(image_rgb[:500, :])

    color_image_new = copy.deepcopy(color_image)
    depth_image_new =  copy.deepcopy(depth_image)
    # Draw the pose landmarks on the image
    mp_pose = mp.solutions.pose
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            color_image_new[:500, :], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(
            depth_image_new[:500, :], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
   
    # Display the image
   # Display the original and scaled images side by side 
    combined_image = np.vstack((color_image_new, depth_image_new))
    cv2.imshow("Original and Scaled Images", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 2D pixel convert back to pixel
    return results.pose_landmarks.landmark



def color_to_depth(b, g, r):
   # math is defined here: https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras 
   d_min = 1.5
   d_max = 3.9
   d = 0
   if b + g + r < 255:
        return 0
   elif r >= g and r >= b:
        if g >= b:
            d =  g - b
        else:
            d =  (g - b) + 1529
   elif g >= r and g >= b:
        d =  b - r + 510
   elif b >= g and b >= r:
        d =  r - g + 1020
   # inverse
   depth = d_min + (d_max - d_min) * d / 1529

   # depth = 1529 / (1529 * d_min + (d_max - d_min) * d)
   return depth

# 3. find corresponding pixels in depths image
def get_depth(dimage, xpixel, ypixel):
   dimage = np.array(dimage, dtype=np.float32) * 2
   depth_color = dimage[ypixel][xpixel]
   # perform a gaussian blur to get an average of the surrounding? 
   
   depth = color_to_depth(depth_color[0], depth_color[1], depth_color[2])
   # get color of the corresponding depth. 
   return depth

def filter_landmarks(results, arm, width, height):
   NOSE = 0
   LEFT_EYE = 2
   RIGHT_EYE = 5
   LEFT_SHOULDER = 11
   RIGHT_SHOULDER = 12
   LEFT_ELBOW = 13
   RIGHT_ELBOW = 14
   LEFT_WRIST = 15
   RIGHT_WRIST = 16

   LEFT_HEEL = 29
   RIGHT_HEEL = 30
   LEFT_FOOT_INDEX = 31

   filtered_landmarks = {}
   if arm == "l" or arm == "left":
      eye = results[LEFT_EYE]
      shoulder = results[LEFT_SHOULDER]
      elbow = results[LEFT_ELBOW]
      wrist = results[LEFT_WRIST]
   else:
      eye = results[RIGHT_EYE]
      shoulder = results[RIGHT_SHOULDER]
      elbow = results[RIGHT_ELBOW]
      wrist = results[RIGHT_WRIST]
   g_res = {}   
   g1 =  results[LEFT_HEEL]
   g2 =  results[RIGHT_HEEL]
   g3 =  results[LEFT_FOOT_INDEX]
   ground_points = [g1, g2, g3]
   ground = ["g1", "g2", "g3"]

   nose = results[NOSE]
   points = [nose, eye, shoulder, elbow, wrist]
   names = ["nose", "eye", "shoulder", "elbow", "wrist"]
   for i in range(len(names)):
      name = names[i]
      point = points[i]
      filtered_landmarks[name] = {"x": int(np.round(point.x*width)), "y": int(np.round(point.y*500))}
   for j in range(len(ground)):
      name = ground[j]
      point = ground_points[j]
      g_res[name] = {"x": int(np.round(point.x*width)), "y": int(np.round(point.y*500))}
   
   return filtered_landmarks, g_res


def to_3d_coordinate_camera_center(dimage, x, y):
   fx = 318.211578
   fy = 318.211578
   PPx = 319.3939
   PPy = 178.9961
   normalized_x = (x - PPx) / fx
   normalized_y = (y - PPy) / fy
   d = get_depth(dimage, x, y)
   world_x = - normalized_x * d
   world_y =  normalized_y * d / 1.6
   world_z =  d

   return world_x, world_y, world_z

def find_arm_coordinates(dimage, results, ground_res):
   # center at nose x, y, ground
   coordinates_3d = {}
   for key, val in  results.items():
      x = val["x"]
      y=  val["y"]
      loc = to_3d_coordinate_camera_center(dimage, x, y)

      coordinates_3d[key] = loc

   offset = np.array([coordinates_3d["nose"][0], 0, coordinates_3d["nose"][2]])
   s = 0
   for key, val in  ground_res.items():
      x = val["x"]
      y=  val["y"]
      grd_loc = to_3d_coordinate_camera_center(dimage, x, y)
      s +=  grd_loc[1]

   offset = - np.array(coordinates_3d["nose"]) - np.array([0, s/3, 0])

   res = {}
   for k, v in coordinates_3d.items():
      res[k] = - np.array(v) - offset

   return res, coordinates_3d["nose"]

def plane_line_intersection(la, lb, p0, p1, p2):

    # the line passing through la and lb is la + lab*t, where t is a scalar parameter
    la = np.array(la)
    lb = np.array(lb)
    lab = lb-la # vector from point 1 to point 2

    # the plane passing through p0, p1, p2 is p0 + p01*u + p02*v, where u and v are scalar parameters
    # ground plane (y-0)

   #  p0 = np.array([0,0-y_offset,0]) # point 0 on plane
   #  p1 = np.array([0,0-y_offset,1]) # point 1 on plane
   #  p2 = np.array([1,0-y_offset,0]) # point 2 on plane

    p01 = np.array(p1)-np.array(p0) # vector from point 0 to point 1
    p02 = np.array(p2)-np.array(p0) # vector from point 0 to point 2

    # setting this up as a system of linear equations and solving for t,u,v
    A = np.array([-lab, p01, p02]).T # the matrix of coefficients
    b = np.array([la-p0]).T# the vector of constants
    try:
      tuv = np.matmul(np.linalg.inv(A),b) # solve the system of linear equations
      intersection =  la+lab*tuv[0] # the solution is the point of intersection
      # calculate the angle between the vector and plane
      n = np.cross(p01, p02)
      angle = math.pi/2 - np.arccos(abs(np.dot(n, lab)) / np.linalg.norm(n) * np.linalg.norm(lab))
      return [angle, intersection]
    except:
      return [None]
  
def calculate_vector(a, b):
  if a and b:
    distance = [b[0]-a[0], b[1]-a[1], b[2]-a[2]]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2 + distance[2] ** 2)
    return [distance[0] / norm, distance[1] / norm, distance[2] / norm]
  else:
    return None
  
  #result[0]: right, result[1]: down, result[2]: forward
  # return result[2], -result[0], -result[1]
# 4. color to depth conversion 
# 5. 2D point to 3D point conversion (https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a )

# April Tag
# 1. Target pixel detection (locate center)
def get_tag_pixel():
   return
# 2. target 2D to 3D point conversion
def get_target_position():
   return
# 3. Human Center Detection
def get_human_position():
   return


# 4. human 2D to 3D conversion
def get_ground(p1, p2, p3):
   LEFT_HEEL = 29
   RIGHT_HEEL = 30
   LEFT_FOOT_INDEX = 31
   RIGHT_FOOT_INDEX = 32
   
   return




## NOTE: ALL PREVIOUS POINTS ARE CAMERA CENTRIC
# 1. Convert to human centric (vector + target locations)
# 2. coordinate indicator: right is positive , front is positive, above + 
# 3. find vector ground intersection for each vector and each plane

# SAVE in JSON format

# Other
# plot_2d
# plot_3d

# main


IMG_PATH = "/Users/ivyhe/Desktop/p1.png"
IMG_PATH = "/Users/ivyhe/Downloads/thumb0116.jpg"
folder_path = "/Users/ivyhe/Desktop/pointing/side_cam/left"
for filename in os.listdir(folder_path):
   IMG_PATH = os.path.join(folder_path, filename)
   if os.path.isfile(IMG_PATH) and (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):

      pointing_arm = "l"
      pointing_target = 1
      print("--------------")
      print(filename)

      # CONSTANTS
      # image size
      ox = 70
      oy = 50

      human_loc = [690+ox, 250+oy]
      t1c = [411+ox, 523+oy]
      t2c = [597+ox, 486+oy]
      t3c = [783+ox, 480+oy]
      t4c = [975+ox, 517+oy]
      # front camera: 1, side camera: 2
      color_image, depth_image, w, h = split_image(IMG_PATH, 2)

      raw_results = detect_2d_landmark(color_image)
      results, ground_points = filter_landmarks(raw_results, pointing_arm, w, h)

      # nose is the center
      results_3D, center_offset = find_arm_coordinates(depth_image, results, ground_points)

      # t1_3d  = np.array(to_3d_coordinate_camera_center(depth_image, t1c[0], t1c[1]))- np.array(center_offset)
      # t2_3d  = np.array(to_3d_coordinate_camera_center(depth_image, t2c[0], t2c[1]))- np.array(center_offset)
      # t3_3d  = np.array(to_3d_coordinate_camera_center(depth_image, t3c[0], t3c[1])) - np.array(center_offset)
      # t4_3d  = np.array(to_3d_coordinate_camera_center(depth_image, t4c[0], t4c[1])) - np.array(center_offset)
      t1_3d = [1.05, 0, 1.0]
      t2_3d = [.36, 0, 0.72]
      t3_3d = [-.36, 0, 0.72]
      t4_3d = [-1.05, 0, 1.0]
      
      print("INTERSECTION POINT ::::::: ")
      for k, v in results_3D.items():
         if k == "wrist":
            continue
         la = v
         lb = results_3D["wrist"]
         angle, intersection = plane_line_intersection(la, lb, t1_3d, t1_3d+np.array([1, 0, 0]),  t1_3d+np.array([0, 0, 1]))
         print(k, intersection)

