#!/bin/bash
source /home/dylan/anaconda3/bin/activate pointing
cd "$(dirname "$0")"  # Navigate to the directory where the script is located
python GUI/bag_slice_gui.py  # Run the Python GUI script (adjust the Python path if needed)
