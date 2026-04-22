#!/bin/bash
# Process all CCD Point Production subjects one at a time with disk space management
# Each subject: extract zip → MediaPipe skeleton → pointing analysis → output → cleanup
#
# Usage: bash run_all_ccd_production.sh

set +e  # Don't exit on error - continue with next subject

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SSD="/media/tigerli/Extreme SSD/pointing_data/point_production_CCD"
OUTPUT="/home/tigerli/Documents/pointing_data/point_production_CCD_output"
LOG="/tmp/ccd_production_reprocess.log"

echo "=== CCD Production Reprocess $(date) ===" | tee "$LOG"

# All subjects (front camera, adult pointer)
SUBJECTS=(
    CCD0194_PVPT_008E_front
    CCD0306_PVPT_002E_front
    CCD0346_PVPT_0012E_front
    CCD0384_PVPT_004E_front
    CCD0385_PVPT_007E_front
    CCD0390_PVPT_003E_front
    CCD0391_PVPT_006E_front
    CCD0392_PVPT_005E_front
    CCD0413_PVPT_010E_front
    CCD0425_PVPT_009E_front
    CCD0427_PVPT_011E_front
    CCD0430_PVPT_014E_front
    CCD0431_PVPT_0013E_front
    CCD0442_PVPT_017E_front
    CCD0443_PVPT_019E_front
    CCD0444_PVPT_015E_front
    CCD0445_PVPT_018E_front
    CCD0447_PVPT_021E_front
    CCD0448_PVPT_022E_front
    CCD0455_PVPT_016E_P1_front
    CCD0467_PVPT_020E_front
    CCD0482_PVPT_023E_front
    CCD0491_PVPT_025E_front
    CCD0495_PVPT_026E_front
)

TOTAL=${#SUBJECTS[@]}
OK=0
FAIL=0
SKIP=0

for i in "${!SUBJECTS[@]}"; do
    subj="${SUBJECTS[$i]}"
    n=$((i + 1))
    echo "" | tee -a "$LOG"
    echo "[$n/$TOTAL] $subj" | tee -a "$LOG"

    # Check if subject dir exists on SSD
    if [ ! -d "$SSD/$subj" ]; then
        echo "  SKIP: Not found on SSD" | tee -a "$LOG"
        SKIP=$((SKIP + 1))
        continue
    fi

    # Check disk space
    avail=$(df --output=avail /home/tigerli/Documents/ | tail -1)
    avail_gb=$((avail / 1024 / 1024))
    echo "  Disk space: ${avail_gb}GB available" | tee -a "$LOG"
    if [ "$avail_gb" -lt 10 ]; then
        echo "  WARNING: Low disk space (${avail_gb}GB). Cleaning work dir..." | tee -a "$LOG"
        rm -rf /home/tigerli/Documents/pointing_data/_ccd_prod_work/*
        avail=$(df --output=avail /home/tigerli/Documents/ | tail -1)
        avail_gb=$((avail / 1024 / 1024))
        echo "  After cleanup: ${avail_gb}GB available" | tee -a "$LOG"
        if [ "$avail_gb" -lt 10 ]; then
            echo "  SKIP: Still not enough disk space" | tee -a "$LOG"
            SKIP=$((SKIP + 1))
            continue
        fi
    fi

    # Run processing with 15 min timeout
    timeout 900 python batch_reprocess_CCD_production.py --subject "$subj" 2>&1 | tee -a "$LOG"
    exit_code=$?

    if [ $exit_code -eq 124 ]; then
        echo "  TIMEOUT after 15 minutes" | tee -a "$LOG"
    fi

    # Check result
    result_dir="$OUTPUT/$subj"
    if [ -d "$result_dir" ]; then
        trial_count=$(find "$result_dir" -name "processed_gesture.csv" 2>/dev/null | wc -l)
        echo "  Result: $trial_count trials with CSV" | tee -a "$LOG"
        if [ "$trial_count" -gt 0 ]; then
            OK=$((OK + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    else
        echo "  FAIL: No output directory" | tee -a "$LOG"
        FAIL=$((FAIL + 1))
    fi

    # Cleanup work dir
    rm -rf /home/tigerli/Documents/pointing_data/_ccd_prod_work/*
    echo "  Cleaned up work dir" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "====================================" | tee -a "$LOG"
echo "DONE: $OK OK, $FAIL failed, $SKIP skipped / $TOTAL total" | tee -a "$LOG"

# Generate global CSV
echo "Generating global CSV..." | tee -a "$LOG"
python -c "
from batch_reprocess_CCD_production import generate_global_csv
from pathlib import Path
generate_global_csv(Path('$OUTPUT'))
" 2>&1 | tee -a "$LOG"

echo "Full log: $LOG"
