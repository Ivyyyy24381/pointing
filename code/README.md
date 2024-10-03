# 1. Pointing Trial Video recording

# 2. after video is recorded, split and extract each trial:

# 2.1 download the csv
cd directory to where the bag is: (i.e. cd Downloads/)

run: 
python ~/Documents/GitHub/pointing/code/batch_split_bag.py --csv_filepath ADD_YOUR_CSV_PATH


This csv runs the below sample code(note to self), no need to run:
python ~/Documents/GitHub/pointing/code/bag_to_video.py --bag_filepath 20240319_105021.bag --date 0319 --trial 2 --start_sec 220 --end_sec 237

# 3. after the code finish running, add the output to the corresponding folder. ensure that: 
3.1 video for each trial with color and heatmap

refer to 0319-data for reference 


