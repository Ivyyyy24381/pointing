#!/bin/bash
source /home/dylan/anaconda3/bin/activate point_production
cd "$(dirname "$0")"  # Navigate to the directory where the script is located
python GUI/skeleton_gui2.py  # Run the Python GUI script (adjust the Python path if needed)
