import deeplabcut

# update the paths to your video files
# PLEASE EDIT THESE:
path = 'train'
deeplabcut.create_new_project(f'dlc-dog-pose','ivy', \
                              [f'{path}/coco_train01.mov', \
                               f'{path}/coco_train02.mov', \
                               f'{path}/coco_train03.mov', \
                               f'{path}/coco_train04.mov', \
                               f'{path}/coco_train05.mov', \
                               f'{path}/coco_train06.mov', \
                               f'{path}/coco_train07.mov', \
                               f'{path}/coco_train08.mov', \
                               f'{path}/coco_train09.mov', \
                               f'{path}/coco_train10.mov'],
              copy_videos=True, multianimal=False)

# Setup your project variables:
# PLEASE EDIT THESE:
ProjectFolderName = 'PVP-dog-pose'
VideoType = 'mov' 


# This creates a path variable to your project's config.yaml file
path_config_file = 'dog_pose_estimation/dlc-dog-pose-ivy-2025-04-09/config.yaml'
path_config_file


# This step is to extract frames from the videos you added to the project
# deeplabcut.extract_frames(path_config_file, mode='automatic', algo='kmeans', crop=True)


# Label your frames
# IMPORTANT: You must run this step from a computer with a display or use VNC (GUI required)
deeplabcut.label_frames(path_config_file)


# There are many more functions you can set here, including which network you use
# check the docstring for full options you can use
deeplabcut.create_training_dataset(path_config_file, net_type='resnet_50', augmenter_type='imgaug')



# Typically, you want to train to 200,000 + iterations.
# more info and there are more things you can set: https://github.com/AlexEMG/DeepLabCut/blob/master/docs/functionDetails.md#g-train-the-network

deeplabcut.train_network(path_config_file, shuffle=1, displayiters=100,saveiters=500)

# This will run until you stop it (CTRL+C), or hit "STOP" icon, or when it hits the end (default, 1.03M iterations). 
# Whichever you chose, you will see what looks like an error message, but it's not an error - don't worry....


deeplabcut.evaluate_network(path_config_file,plotting=True)

# Here you want to see a low pixel error! Of course, it can only be as good as the labeler, 
#so be sure your labels are good! (And you have trained enough ;)



# This is the location of the videos to analyze
videofile_path = ['/home/ryan/code/repos/dog-pose-estimation/videos']
videofile_path


deeplabcut.analyze_videos(path_config_file,videofile_path, videotype='mov')

deeplabcut.plot_trajectories(path_config_file,videofile_path, videotype='mov')


deeplabcut.create_labeled_video(path_config_file, videofile_path, videotype='mov', draw_skeleton=True)


# PLACEHOLDER: This step is where we will convert to OpenVINO for faster inference 
# export frozen TF graph (.pb format)
deeplabcut.export_model(config_path, iteration=None, shuffle=1, trainingsetindex=0, snapshotindex=None, TFGPUinference=False, overwrite=False, make_tar=True)