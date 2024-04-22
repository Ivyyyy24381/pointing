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
print()
# import raw depth data, color image
# Get the path of the current Python script
script_path = os.path.abspath(__file__)

# Get the root directory
root_path = os.path.abspath(os.path.join(script_path, os.pardir))
print(script_path)
print(root_path)
# unit (pixel, pixel, meter)
trimmed_height = 500
# Scale factor for downscaling

scale_factor = 0.38
width = 1280
height = 720
color_w = 1280 
color_h = 720
def detect_2d_landmark(color_image, img_name):
    # start mediapipe detection
   # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(image_rgb[:trimmed_height, :])

    color_image_new = copy.deepcopy(color_image)
    
    # Draw the pose landmarks on the image
    mp_pose = mp.solutions.pose
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            color_image_new[:trimmed_height, :], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        #mp.solutions.drawing_utils.draw_landmarks(
            #depth_image_new[:, :], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
   
    # Display the image
   # Display the original and scaled images side by side 
    cv2.imshow("Pose Detection Images", color_image_new[:trimmed_height, :])
    # Join the folder path and file name to get the complete file path
    save_path = f'{root_path}/gesture_image/' + img_name + '.jpg'
    # Save the image
    cv2.imwrite(save_path, color_image_new)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if results is None or results.pose_landmarks is None:
        return None
    # 2D pixel convert back to pixel
    return results.pose_landmarks.landmark

def find_perpendicular_vector(point1, point2, point3):
    # Vector from point1 to point2
    vector1 = np.array(point2) - np.array(point1)

    # Vector from point1 to point3
    vector2 = np.array(point3) - np.array(point1)

    # Cross product to find a vector perpendicular to the plane
    perpendicular_vector = np.cross(vector1, vector2)

    return [point1, point1 + perpendicular_vector]

# 3. find corresponding pixels in depths image
def get_depth(dimage, xpixel, ypixel):
   # dimage = np.array(dimage, dtype=np.float32)
   depth = dimage[ypixel][xpixel] / 1000
   # get color of the corresponding depth. 
   if depth < 0.01:
      return get_depth(dimage, xpixel-1, ypixel-1)
   
   return depth

def scale_point_from_center(original_point, scale_factor, center):
    """
    Scale a point from a specified center.

    Parameters:
        - original_point: Tuple (x, y) representing the original coordinates of the point.
        - scale_factor: Scaling factor (e.g., 0.5 for 50% scale down, 2.0 for doubling the size).
        - center: Tuple (cx, cy) representing the center of scaling.

    Returns:
        - Tuple (new_x, new_y) representing the new coordinates of the scaled point.
    """
    x, y = original_point
    cx, cy = center

    # Translate the point to the origin
    translated_x = x - cx
    translated_y = y - cy

    # Scale the translated point
    scaled_x = translated_x * scale_factor
    scaled_y = translated_y * scale_factor

    # Translate the scaled point back to its original position
    new_x = scaled_x + cx
    new_y = scaled_y + cy

    return new_x, new_y

def filter_landmarks(results, arm, width, height, pad_width, pad_height):
   NOSE = 0
   LEFT_EYE = 2
   RIGHT_EYE = 5
   LEFT_SHOULDER = 11
   RIGHT_SHOULDER = 12
   LEFT_ELBOW = 13
   RIGHT_ELBOW = 14
   LEFT_WRIST = 15
   RIGHT_WRIST = 16

   LEFT_BROW = 1
   RIGHT_BROW = 4
   MOUTH = 10

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
   facing = {}   
   g3 =  results[LEFT_BROW]
   g2 =  results[RIGHT_BROW]
   g1 =  results[MOUTH]
   head_points = [g1, g2, g3]
   head = ["g1", "g2", "g3"]

   nose = results[NOSE]
   points = [nose, eye, shoulder, elbow, wrist]
   names = ["nose", "eye", "shoulder", "elbow", "wrist"]
   for i in range(len(names)):
      name = names[i]
      point = points[i]
      x_old = np.round(point.x * color_w) 
      y_old = np.round(point.y * trimmed_height) 
      x, y = scale_point_from_center((x_old, y_old), scale_factor, (0, 0))
      filtered_landmarks[name] = {"x": int(x+pad_width), "y": int(y+pad_height)}
   for j in range(len(head)):
      name = head[j]
      point = head_points[j]
      x_old = np.round(point.x * color_w) 
      y_old = np.round(point.y * trimmed_height) 
      x, y = scale_point_from_center((x_old, y_old), scale_factor, (0, 0))
      facing[name] = {"x": int(x+pad_width), "y": int(y+pad_height)}
   
   # distance[m] =  distance[pixel] * scale factor 
   p1_x_old = np.round(results[RIGHT_EYE].x*color_w) 
   p1_y_old = np.round(results[RIGHT_EYE].y* trimmed_height) 
   p1_x, p1_y = scale_point_from_center((p1_x_old, p1_y_old), scale_factor, (0, 0))
   point1 = np.array([p1_x+pad_width, p1_y+pad_height])
   p2_x_old = np.round(results[LEFT_EYE].x*color_w) 
   p2_y_old = np.round(results[LEFT_EYE].y* trimmed_height) 
   p2_x, p2_y = scale_point_from_center((p2_x_old, p2_y_old), scale_factor, (0, 0))
   point2 = np.array([p2_x+pad_width, p2_y+pad_height])

   dist = np.sqrt(np.sum((point2 - point1)**2))
   # 0.062 is average pupil distance in m
   pixel_scale_factor = 0.062 / dist
   return filtered_landmarks, facing, pixel_scale_factor

def find_3d_coordinates(dimage, results):
   # center at nose x, y, ground
   coordinates_3d = {}
   offset = 5
   # Plot the depth image
   plt.imshow(depth_image, cmap='viridis')  # You can choose a different colormap
   for key, val in  results.items():
      x = val["x"] - offset
      y=  val["y"] + offset
      loc = get_depth(dimage, x, y)
      # Draw the point on the image
      plt.scatter(x, y, s=3, label=key)
      plt.title('Depth Image with Point')
      plt.legend()
      coordinates_3d[key] = [x, y, loc]
    
   res = {}
   for k, v in coordinates_3d.items():
      
      res[k] = np.array(v)

   plt.colorbar(label='Depth Value')
   plt.show()
   return res

def plane_line_intersection( LA, LB, A1, A2, A3):
    # Calculate the normal vector of the plane
    N = np.cross(A2 - A1, A3 - A1)

    # Calculate the direction vector of the line
    L = LB - LA

    # Calculate the parameter t
    t = np.dot(N, (A1 - LA)) / np.dot(N, L)

    # Calculate the intersection point
    intersection_point = LA + t * L

    return intersection_point.tolist()

# def plane_line_intersection(la, lb, p0, p1, p2):

#     # the line passing through la and lb is la + lab*t, where t is a scalar parameter
#     la = np.array(la)
#     lb = np.array(lb)
#     lab = lb-la # vector from point 1 to point 2

#     # the plane passing through p0, p1, p2 is p0 + p01*u + p02*v, where u and v are scalar parameters
#     # ground plane (y-0)

#    #  p0 = np.array([0,0-y_offset,0]) # point 0 on plane
#    #  p1 = np.array([0,0-y_offset,1]) # point 1 on plane
#    #  p2 = np.array([1,0-y_offset,0]) # point 2 on plane

#     p01 = np.array(p1)-np.array(p0) # vector from point 0 to point 1
#     p02 = np.array(p2)-np.array(p0) # vector from point 0 to point 2

#     # setting this up as a system of linear equations and solving for t,u,v
#     A = np.array([-lab, p01, p02]).T # the matrix of coefficients
#     b = np.array([la-p0]).T# the vector of constants
    
#     tuv = np.matmul(np.linalg.inv(A),b) # solve the system of linear equations
#     intersection =  la+lab*tuv[0] # the solution is the point of intersection
    
#     # calculate the angle between the vector and plane
#     # n = np.cross(p01, p02)
#     # angle = math.pi/2 - np.arccos(abs(np.dot(n, lab)) / np.linalg.norm(n) * np.linalg.norm(lab))
#     return np.round(intersection, decimals = 4).tolist()


# Replace 'your_folder_path' with the actual path to your folder
# folder_path = '/Users/ivyhe/Desktop/pointing_vconfig/'
folder_path = '/Users/ivyhe/Downloads/HConfig/'
# folder_path = f'{root_path}/'

# Get a list of all files in the folder
files = os.listdir(folder_path)
columns = ['img_name', 'assumed_pointing_arm', 'vector', 
           'VGI_x[px]', 'VGI_y[px]', 'VGI_z[m]', 
           't1_d_x[m]', 't1_d_y[m]','t1_d_z[m]', 't1_d[m]',
           't2_d_x[m]', 't2_d_y[m]','t2_d_z[m]', 't2_d[m]',
           't3_d_x[m]', 't3_d_y[m]','t3_d_z[m]', 't3_d[m]',
           't4_d_x[m]', 't4_d_y[m]','t4_d_z[m]', 't4_d[m]', 'target_guess']

table_data = pd.DataFrame(columns=columns)

# Specify the prefix you want to filter files with
# dates = ['1125', '1203','1207','1208', '1212','0126', '1208']
# date = input("Enter a date [MMDD]: ")
dates = ['0219']
# dates = ['1101', '1102', '1103']
trials = ['t21', 't22', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't23', 't24']
# Loop through files and perform actions for each unique starting prefix
checked_list = []
i = 0
# vertical config
targets = {'target_1': {'x': 862, 'y': 468}, 'target_2': {'x': 819, 'y': 510}, 
           'target_3': {'x': 546, 'y': 513}, 'target_4': {'x': 512, 'y': 466}}

# horizontal config
targets = {'target_1': {'x': 872, 'y': 565}, 'target_2': {'x': 859, 'y': 503}, 
           'target_3': {'x': 499, 'y': 576}, 'target_4': {'x': 520, 'y': 511}}
# TODO: UNCOMMENT AFTER THIS
# targets = {'target_1': {'x': 425, 'y': 225}, 'target_2': {'x': 363, 'y': 214}, 
#            'target_3': {'x': 306, 'y': 217}, 'target_4': {'x': 247, 'y': 227}}
configs = ['ground', 'vertical']
for config in configs:
    for date in dates:
        for trial in trials:
            for file in files:  
                # print(file)  
                # if f'{date}_{trial}_' in file:
                # date = '1208'
                # trial = 't7'
                if (f'{date}_{trial}_' in file) and (file not in checked_list):
                    print("processing", date, ': ', trial)
                    color_path = f'{date}_{trial}_Color.png'
                    depth_path = f'{date}_{trial}_Depth.png'
                    raw_path = f'{date}_{trial}_Color.raw'
                    csv_path = f'{date}_{trial}_Depth_metadata.csv'
                    checked_list.append(color_path)
                    checked_list.append(depth_path)
                    checked_list.append(raw_path)
                    checked_list.append(csv_path)
                    
                    # get images
                    color_img_path = folder_path + f'{date}_{trial}_Color.png'
                    depth_data_path = folder_path + f'{date}_{trial}_Color.raw'

                    color_image_unscaled = cv2.imread(color_img_path)
                    
                    # scale the color image
              

                    # Resize the image using cv2's resize function
                    resized_image = cv2.resize(color_image_unscaled, (0, 0), fx=scale_factor, fy=scale_factor)
                    # Calculate padding dimensions
                    pad_height = (height - resized_image.shape[0]) // 2
                    pad_width = (width - resized_image.shape[1]) // 2

                    # Pad the resized image with black edges using cv2's copyMakeBorder
                    color_image = cv2.copyMakeBorder(resized_image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
                    cv2.imshow("color",color_image)
                    # depth_data = np.fromfile(depth_data_path, dtype=np.uint16).reshape((height, width))
                    depth_data = np.fromfile(depth_data_path, dtype=np.uint16).reshape((720, 1920))*0.0010000000474974513
                    
                    # to change:
                    blurred_depth = cv2.GaussianBlur(depth_data, (5, 5), 0)
                    depth_data = cv2.resize(blurred_depth, (width, height)) 
                    # Normalize the depth data to [0, 255] for visualization
                    normalized_depth = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                    # Convert to 8-bit for proper display with cv2.imshow
                    depth_image = np.uint16(normalized_depth)


                    raw_results = detect_2d_landmark(color_image_unscaled, f'{date}_{trial}_')
                    if raw_results is None:
                        print('redo this trial')
                        continue

                    left_results, head, pixel_scale_factor = filter_landmarks(raw_results, 'l', width, height, pad_width, pad_height)

                    if int(head['g3']['x']) == int(head['g2']['x']) and int(head['g3']['y']) == int(head['g2']['y']):

                        head['g2']['x'] += 2
                        head['g2']['y'] -= 2
                    right_results, _, _ = filter_landmarks(raw_results, 'r', width, height, pad_width, pad_height)
                    # find head direction
                    # head_1, head2 = find_perpendicular_vector(head[0], head[1], head[2])
                    #{key: x, y, z}
                    right_3d = find_3d_coordinates(depth_data, right_results)
    
                    left_3d = find_3d_coordinates(depth_data, left_results)

                    head_3d = find_3d_coordinates(depth_data, head)

                    targets_3d = find_3d_coordinates(depth_data, targets)
                    # print("3D Corrdinates in pixel, pixel, m")
                    # print(right_3d)
                    # print(targets_3d)
                    
                    if targets_3d['target_2'][2] < 2:
                        targets_3d['target_2'][2] = targets_3d['target_3'][2]
                    img_name = file[:8]

                    for k, v in right_3d.items():
                        
                        if k == "wrist":
                            continue
                        la = v
                        lb = right_3d["wrist"]
                        pointing_arm = 'right'
                        v_name = f'{k}-to-wrist'

                        row = [img_name, pointing_arm,v_name]
                        # TODO: uncomment
                        # intersection = plane_line_intersection(la, lb,targets_3d['target_1'], targets_3d['target_2'], targets_3d['target_4'])
                        
                        # comment this out later
                        point1 = targets_3d['target_2']
                        point2 = targets_3d['target_3']
                        point3 = targets_3d['target_4']
                        g1 = np.array([point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]])
                        g2 = np.array([point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2]])
                        cross_prod = np.cross(g1, g2)
                        norm = math.sqrt(cross_prod[0] ** 2 + cross_prod[1] ** 2 + cross_prod[2] ** 2)
                        v11 = [cross_prod[0] / norm, cross_prod[1] / norm, cross_prod[2] / norm]
                        t1 = targets_3d['target_2']
                        t2 = targets_3d['target_3']
                        t3 =  t1+v11
                        # vertical plane
                        intersection = plane_line_intersection(la, lb,t1, t2, t3)
                        # VIG_x = intersection[0] 
                        # VIG_y = intersection[1] 
                        # VIG_z = intersection[2]
                        row += intersection
                        t_dist_list = []
                        for _, tv in targets_3d.items():
                            v_diff = np.array(intersection) - tv 
                            x_m = pixel_scale_factor * v_diff[0]
                            y_m = pixel_scale_factor * v_diff[1]
                            z_m =  v_diff[2]
                            t_dist = np.sqrt(np.sum(np.array([x_m, y_m, z_m])**2))
                            t_dist_list += [t_dist]
                            row += np.round(np.array([x_m, y_m, z_m, t_dist]), decimals = 4).tolist()       
                        guess_target = t_dist_list.index(min(t_dist_list)) + 1
                        row += [guess_target]
                        table_data = table_data.append(pd.Series(row, index=columns), ignore_index=True)
                        

                    for k, v in left_3d.items():
                        
                        if k == "wrist":
                            continue
                        la = v
                        lb = left_3d["wrist"]
                        pointing_arm = 'left'
                        v_name = f'{k}-to-wrist'

                        row = [img_name, pointing_arm,v_name]
                        # TODO
                        if config == 'ground':
                            intersection = plane_line_intersection(la, lb,targets_3d['target_1'], targets_3d['target_2'], targets_3d['target_3'])
                        else:
                            intersection = plane_line_intersection(la, lb,t1, t2, t3)
                        # VIG_x = intersection[0] 
                        # VIG_y = intersection[1] 
                        # VIG_z = intersection[2]
                        row += intersection
                        t_dist_list = []
                        for tk, tv in targets_3d.items():
                            v_diff = np.array(intersection) - tv 
                            x_m = pixel_scale_factor * v_diff[0]
                            y_m = pixel_scale_factor * v_diff[1]
                            z_m =  v_diff[2]
                            t_dist = np.sqrt(np.sum(np.array([x_m, y_m, z_m])**2))
                            t_dist_list += [t_dist]
                            row += np.round(np.array([x_m, y_m, z_m, t_dist]), decimals = 4).tolist()      
                        guess_target = t_dist_list.index(min(t_dist_list)) + 1
                        row += [guess_target]
                        table_data = table_data.append(pd.Series(row, index=columns), ignore_index=True)

                    point1 = head_3d['g1']
                    point2 = head_3d['g2']
                    point3 = head_3d['g3']
                    vector1 = np.array([point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]])
                    vector2 = np.array([point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2]])
                    cross_prod = np.cross(vector1, vector2)
                    norm = math.sqrt(cross_prod[0] ** 2 + cross_prod[1] ** 2 + cross_prod[2] ** 2)
                    vector11 = [cross_prod[0] / norm, cross_prod[1] / norm, cross_prod[2] / norm]
                    p1, p2 = find_perpendicular_vector(head_3d['g1'], head_3d['g2'], head_3d['g3']) 
                    # TODO:
                    
                    # intersection with ground plane
                    if config == 'ground':
                        head_intersection = plane_line_intersection(point1, point1+vector11, targets_3d['target_1'], targets_3d['target_2'], targets_3d['target_4'])
                    else:
                        # intersection with vertical plane
                        head_intersection = plane_line_intersection(point1, point1+vector11,t1, t2, t3)
                    print(targets_3d) 
                    v_name = 'head'
                    row = [img_name, 'NA', v_name]
                    row += head_intersection

                    t_dist_list = []
                    for _, tv in targets_3d.items():
                        
                        v_diff = np.array(head_intersection) - tv 
                        x_m = pixel_scale_factor * v_diff[0]
                        y_m = pixel_scale_factor * v_diff[1]
                        z_m =  v_diff[2]
                        t_dist = np.sqrt(np.sum(np.array([x_m, y_m, z_m])**2))
                        t_dist_list += [t_dist]
                        row += np.round(np.array([x_m, y_m, z_m, t_dist]), decimals = 4).tolist()       
                    # add head intersection 
                    guess_target = t_dist_list.index(min(t_dist_list)) + 1
                    row += [guess_target]
                    print(row)
                    table_data = table_data.append(pd.Series(row, index=columns), ignore_index=True)


                    
    # Output the table to a CSV file
    if config == 'ground':
        output_file_name = f'{date}_pointing_ground.csv'
    else:
        output_file_name = f'{date}_pointing_vertical.csv'
    output_csv_path = f'{folder_path}/output/{output_file_name}'
    table_data.to_csv(output_csv_path, index=False)
    print(f"Table appended to CSV file: {output_csv_path}")

