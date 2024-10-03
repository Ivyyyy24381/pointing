
import subprocess

# Define the commands to be run
commands = [
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/2m_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/2m_data.csv",
    "python pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/videos/3m_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/3m_data.csv",
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/4m_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/4m_data.csv",
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/5m_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/5m_data.csv",
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/6m_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/6m_data.csv",
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_left_cross_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/ground_left_cross_data.csv",
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_right_cross_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/ground_right_cross_data.csv",
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_left_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/ground_left_data.csv",
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_right_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/ground_right_data.csv",
    "python3 pose_detection/gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_sit_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/ground_sit_data.csv"

]

# commands = [
    
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/2m_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_2m_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/3m_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_3m_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/4m_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_4m_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/5m_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_5m_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/6m_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_6m_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/ground_left_cross_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_left_cross_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/ground_right_cross_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_right_cross_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/ground_left_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_left_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/ground_right_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_right_data.csv --threshold 0.4",
#     "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/ground_sit_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_sit_data.csv --threshold 0.3"
    
# ]

"python pose_detection/eval/data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/2m_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_2m_data.csv --threshold 0.4"
'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/2m_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_2m_data.csv --target_locations -1 0.46 2 1 0.74 2'
"python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/2m_video --output_video /Users/ivy/Desktop/spot_gesture_eval/2m_processed.mp4"

"python pose_detection/eval/data_cleanup.py --input csv_1 --output csv_2 --threshold 0.4"
'python pose_detection/eval/pointing_eval.py  --video_path video_1 --csv_path csv_2 --target_locations -1 0.46 2 1 0.74 2'
"python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/2m_video --output_video video_2"

# commands = [
    
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/2m_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_2m_data.csv --target_locations -1 0.46 2 1 0.74 2',
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/3m_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_3m_data.csv --target_locations -1 0.46 3 1 0.74 3',
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/4m_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_4m_data.csv --target_locations -1 0.46 4 1 0.74 4',
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/5m_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_5m_data.csv --target_locations -1 0.46 5 1 0.74 5',
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_sit_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_sit_data.csv --target_locations -1 0 2 1 0 2 -1 0 3 1 0 3 -1 0 4 1 0 4 -1 0 5 1 0 5',
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_left_cross_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_left_cross_data.csv --target_locations -1 0 2 1 0 2 -1 0 3 1 0 3 -1 0 4 1 0 4 -1 0 5 1 0 5',
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_right_cross_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_right_cross_data.csv --target_locations -1 0 2 1 0 2 -1 0 3 1 0 3 -1 0 4 1 0 4 -1 0 5 1 0 5',
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_left_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_left_data.csv --target_locations -1 0 2 1 0 2 -1 0 3 1 0 3 -1 0 4 1 0 4 -1 0 5 1 0 5',
#     'python pose_detection/eval/pointing_eval.py  --video_path /Users/ivy/Desktop/spot_gesture_eval/ground_right_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_ground_sit_data.csv --target_locations -1 0 2 1 0 2 -1 0 3 1 0 3 -1 0 4 1 0 4 -1 0 5 1 0 5',
    
# ]
# commands = [
    
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/2m_video --output_video /Users/ivy/Desktop/spot_gesture_eval/2m_processed.mp4 --fps 30", 
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/3m_video --output_video /Users/ivy/Desktop/spot_gesture_eval/3m_processed.mp4 --fps 30", 
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/4m_video --output_video /Users/ivy/Desktop/spot_gesture_eval/4m_processed.mp4 --fps 30", 
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/5m_video --output_video /Users/ivy/Desktop/spot_gesture_eval/5m_processed.mp4 --fps 30", 
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/ground_sit_video --output_video /Users/ivy/Desktop/spot_gesture_eval/ground_sit_processed.mp4 --fps 30", 
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/ground_left_cross_video --output_video /Users/ivy/Desktop/spot_gesture_eval/ground_left_cross_processed.mp4 --fps 30", 
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/ground_right_cross_video --output_video /Users/ivy/Desktop/spot_gesture_eval/ground_right_cross_processed.mp4 --fps 30", 
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/ground_left_video --output_video /Users/ivy/Desktop/spot_gesture_eval/ground_left_processed.mp4 --fps 30", 
#     "python pose_detection/eval/video_output.py --image_dir /Users/ivy/Desktop/spot_gesture_eval/ground_right_video --output_video /Users/ivy/Desktop/spot_gesture_eval/ground_right_processed.mp4 --fps 30", 
# ]

# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_left_1.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_left_1.csv  --threshold 0.1 --target_locations -1 0.46 5
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_left_2.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_left_2.csv  --threshold 0.1 --target_locations -1 0.46 5
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_right_3.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_right_3.csv  --threshold 0.1 --target_locations 1 0.74 5
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_right_4.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_right_4.csv  --threshold 0.1 --target_locations 1 0.74 5
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/6m_left_1.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/6m_left_1.csv  --threshold 0.1 --target_locations -1 0.46 6
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/6m_left_2.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/6m_left_2.csv  --threshold 0.1 --target_locations -1 0.46 6
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/6m_right_3.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/6m_right_3.csv  --threshold 0.1 --target_locations 1 0.74 6
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/6m_right_4.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/6m_right_4.csv  --threshold 0.1 --target_locations 1 0.74 6
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_6.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_6.csv  --threshold 0.1 --target_locations -1 0 6
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_5.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_5.csv  --threshold 0.1 --target_locations -1 0 5
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_4.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_4.csv  --threshold 0.1 --target_locations -1 0 4
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_right_6.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_right_6.csv  --threshold 0.1 --target_locations 1 0 6
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_right_5.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_right_5.csv  --threshold 0.1 --target_locations 1 0 5
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_right_4.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_right_4.csv  --threshold 0.1 --target_locations 1 0 4
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_cross_6.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_cross_6.csv  --threshold 0.1 --target_locations -1 0 6
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_cross_5.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_left_cross_5.csv  --threshold 0.1 --target_locations -1 0 5
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_right_cross_6.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/ground_right_cross_6.csv  --threshold 0.1 --target_locations 1 0 6
# python pose_detection/eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/5m_video.csv  --threshold 0.1 --target_locations -1 0.46 5 1 0.74 5


commands = [
    'python eval/run_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/0926/2m_left_1.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/0926/2m_left_1.csv  --threshold 0.3 --target_locations -1 0.46 2',
    
]
# Run each command in the list
for command in commands:
    try:
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {command}")
        print(e)
