#!/bin/bash

SRC="/home/xhe71/Desktop/dog_data/baby"
DST="/media/xhe71/TOSHIBA/point_production"

# Graceful exit on Ctrl+C
trap 'echo "⚠️ Script interrupted by user. Exiting safely..."; exit 1' INT

for dir in "$SRC"/*; do
    if [ -d "$dir" ]; then
        folder_name=$(basename "$dir")
        echo "=== Copying folder: $folder_name ==="
        
        # Run rsync with timeout, show live progress while capturing output
        rsync_output=$(timeout 600 rsync -a --info=progress2 --itemize-changes "$dir" "$DST/" | tee /dev/stderr)
        rsync_status=$?
        
        if [ $rsync_status -eq 0 ] && [ -n "$rsync_output" ]; then
            echo ">>> Changes detected, flushing cache to disk..."
            sync
            echo "=== Finished $folder_name with updates, resting for 60 seconds... ==="
            sleep 60
        elif [ $rsync_status -eq 124 ]; then
            echo "❌ Timeout reached while copying $folder_name. Skipping to next folder."
        else
            echo "=== No changes for $folder_name, skipping rest. ==="
        fi
    fi
done

echo "✅ All folders copied successfully!"