#!/bin/bash
# Process all CCD subjects one at a time with disk space management
# Each subject: extract zip → SAM3 → mask-based trace → output → cleanup
#
# Usage: bash run_all_ccd_reprocess.sh
# Full reprocessing from scratch (SAM + mask depth + smoothing + optimal path)

set +e  # Don't exit on error - continue with next subject

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SSD="/media/tigerli/Extreme SSD/pointing_data/point_comprehension_CCD"
OUTPUT="/home/tigerli/Documents/pointing_data/point_comprehension_CCD_output"
LOG="/tmp/ccd_full_reprocess.log"

echo "=== CCD Full Reprocess $(date) ===" | tee "$LOG"

# List of subjects on SSD (excluding CCD0392 which has unsplit data)
SUBJECTS=(
    CCD0346_PVPT_0012E_side
    CCD0384_PVPT_004E_side
    CCD0390_PVPT_003E_side
    CCD0391_PVPT_006E_side
    CCD0413_PVPT_010E_side
    CCD0425_PVPT_009E_side
    CCD0427_PVPT_011E_side
    CCD0430_PVPT_014E_side
    CCD0431_PVPT_0013E_side
    CCD0442_PVPT_017E_side
    CCD0443_PVPT_019E_side
    CCD0444_PVPT_015E_side
    CCD0445_PVPT_018E_side
    CCD0447_PVPT_021E_side
    CCD0448_PVPT_022E_side
    CCD0455_PVPT_016E_P1_side
    CCD0467_PVPT_020E_side
    CCD0482_PVPT_023E_side
    CCD0491_PVPT_025E_side
    CCD0495_PVPT_026E_side
    CCD0540_PVPTC_001
    CCD0656_PVPTE_001
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

    # Check disk space before processing
    avail=$(df --output=avail /home/tigerli/Documents/ | tail -1)
    avail_gb=$((avail / 1024 / 1024))
    echo "  Disk space: ${avail_gb}GB available" | tee -a "$LOG"
    if [ "$avail_gb" -lt 8 ]; then
        echo "  WARNING: Low disk space (${avail_gb}GB). Cleaning work dir..." | tee -a "$LOG"
        rm -rf /home/tigerli/Documents/pointing_data/_ccd_work/*
        avail=$(df --output=avail /home/tigerli/Documents/ | tail -1)
        avail_gb=$((avail / 1024 / 1024))
        echo "  After cleanup: ${avail_gb}GB available" | tee -a "$LOG"
        if [ "$avail_gb" -lt 8 ]; then
            echo "  SKIP: Still not enough disk space" | tee -a "$LOG"
            SKIP=$((SKIP + 1))
            continue
        fi
    fi

    # Remove existing segmented_color for zip-based subjects to force SAM rerun
    # (New format subjects on SSD keep their segmented_color)

    # Run processing with timeout (20 min per subject for full pipeline)
    timeout 1200 python batch_reprocess_CCD.py --subject "$subj" 2>&1 | tee -a "$LOG"
    exit_code=$?

    if [ $exit_code -eq 124 ]; then
        echo "  TIMEOUT after 20 minutes" | tee -a "$LOG"
    fi

    # Check result
    result_dir="$OUTPUT/$subj"
    if [ -d "$result_dir" ]; then
        trial_count=$(find "$result_dir" -name "processed_subject_result_table.csv" 2>/dev/null | wc -l)
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
    rm -rf /home/tigerli/Documents/pointing_data/_ccd_work/*
    echo "  Cleaned up work dir" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "====================================" | tee -a "$LOG"
echo "DONE: $OK OK, $FAIL failed, $SKIP skipped / $TOTAL total" | tee -a "$LOG"

# Generate global CSVs
echo "Generating global CSVs..." | tee -a "$LOG"
python process_comprehension_optimal_path.py "$OUTPUT" 2>&1 | tee -a "$LOG"

echo "Full log: $LOG"
