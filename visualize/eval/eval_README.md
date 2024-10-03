##1. evaluation process:
1. process videos:  "python3 gesture_detection.py --mode video --video_path /Users/ivy/Desktop/spot_gesture_eval/2m_video.mp4 --csv_path ~/Desktop/spot_gesture_eval/2m_data.csv"
2. data clean up: "python data_cleanup.py --input /Users/ivy/Desktop/spot_gesture_eval/2m_data.csv --output /Users/ivy/Desktop/spot_gesture_eval/cleaned_2m_data.csv --threshold 0.4"
3. evaluate cleaned data using csv and the original video: --> pointing_eval.py
4. visualize evaluation results