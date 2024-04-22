# Data Summary

## data source & data collection
We have collected data through recording the experimental session in rbg-d video stream form using Intel's Realsense D435 camera. We extracted the raw streaming data from rosbag, and manually selected the frame where the gesture command is made as training data. Human's deictic pointing gesture is detected using MediaPipe gesture detection library where we are able to further process the x, y, z location of the joint and where it intersect with the ground plane. 
- sample images(Test Set): https://drive.google.com/drive/folders/1LYZF_GtnNNMW-Sel-njEgA2rhf4u7U6o



## Data Attribute Explained
### dog selection
- sample: https://github.com/csci1951a-spring-2024/final-project-gestubots/blob/main/data/sample/dog_selection.csv 
- Selected Attribute explained:

| Attribute  | Type | Range/format| Distribution | unique? | required? | usage | Other notes |
| ---------  | ---- | -------------- | -------------| --------| ----------| ------| ------------ | 
| data date  | str | MM-DD-YYYY | 12 trials per experiment day | no | yes | used to identify the time information and as key | This alongside with participant_id and their name could contain sensitive information that we must be careful with disclosure guidelines | 
| trial #  | int | 1-12 | uniformly distributed | no | yes | used to classify the images. trial # + date is unique identification of each image | during data collection, we are aware to have clear distinguish between start and end of each trial. In addition, we only captured the first gesture information of each trial for analysis |
| target_location  | int | 1-4 | uniform distrubution between 1-4 | no | yes | use to determine which target is the correct answer | to ensure the consistency, we randomly chose target locations for each trial |
| dog selection  | (int, int ..) | max len = 4, selection[i] ranges from 1-4 | Most of the dogs performed 2-3 attempts before makignt the correct selection | no | yes | comparison between human and dog performance and study whether dog can interpret gesture command or not | |

### Vector Intersection (main)
- sample: https://github.com/csci1951a-spring-2024/final-project-gestubots/blob/main/data/sample/sample_intersection.csv
- - Selected Attribute explained:

| Attribute  | Type | Range | Distribution | unique? | required? | usage | Other notes |
| ---------  | ---- | -------------- | -------------| --------| ----------| ------| ------------ | 
| img_name  | str  | format: date_trial# | 5 vectors per image | yes | yes | unique identification about each image | |
| vector selection | option | format: one of the 5 vectors that we are aiming to study | | no | no | compare to find the best vector we naturally use during pointing | 
| Vector intersection | float | pixel and meter values of where the vector project to the ground | pixel locations ranges from 200 to 400 pxls | no | Yes
| target distance | float | 200 px to 400 px| distnace to the vector | yes | yes | to find the distance between ground intersection point to the targets in the scene
| closest target | option | similar to target_target, return the location and the distance to the target | | no | yes | determine whether CV is able to find the right target simply using euclidean distance | 


### Target Selection Probability
- sample: https://github.com/csci1951a-spring-2024/final-project-gestubots/blob/main/data/sample/sample_probability.csv 
- Selected Attribute explained:

| Attribute  | Type | Range | Distribution | unique? | required? | usage | Other notes |
| ---------  | ---- | -------------- | -------------| --------| ----------| ------| ------------| 
| probability  | float | 0-1 | range of where the target is and the level of confidnece|no | yes| no| | 
| perplexity | float | < 4 | 4 is the baseline when all targets are equally likely to be selected, this is adapted from perplexity from nlp context. | yes |  no | perplexity is a more relieble indicatior than probability because it compares the relative differnece between vectors and show how confident we are with our data. 


## Link to full data
- raw rosbag files: https://drive.google.com/drive/folders/1ZI_RzGH1PqrzBaxy9gK_cM0-TbR8W_S4 
- raw images: https://drive.google.com/drive/folders/1LYZF_GtnNNMW-Sel-njEgA2rhf4u7U6o 
- Dog Selection: https://docs.google.com/spreadsheets/d/1VaCtzZUv_i3KWNS2QQa5x-Y99OU8Yc2ECzxO2o-AFns/edit#gid=0
