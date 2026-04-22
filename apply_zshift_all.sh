#!/bin/bash
# Apply Z-shift to all studies that haven't been shifted yet.
# This re-runs post-processing with the corrected human Z anchoring.
set +e

OUTPUT_DIR="/home/tigerli/Documents/pointing_data/output"
ANNOTATIONS="/home/tigerli/Documents/pointing_data/PVPO_Production_Data - Output_Tracker.csv"
PROJECT_DIR="/home/tigerli/Documents/GitHub/pointing"
LOG_DIR="/home/tigerli/Documents/pointing_data/logs"
CAMERA="cam1"
DOG_DEPTH_MAX="2.5"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  APPLY Z-SHIFT TO ALL STUDIES"
echo "  $(date)"
echo "=========================================="

OK=0
SKIP=0
FAIL=0
TOTAL=0

for study in "$OUTPUT_DIR"/*_output/; do
    study_name=$(basename "$study")
    TOTAL=$((TOTAL + 1))

    # Check if already shifted
    has_shift=$(grep -rl "z_shift_applied" "$study" 2>/dev/null | head -1)
    if [ -n "$has_shift" ]; then
        echo "[$TOTAL] SKIP (already shifted): $study_name"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo ""
    echo "[$TOTAL] Processing: $study_name"
    log_file="$LOG_DIR/${study_name}_zshift.log"

    python3 "$PROJECT_DIR/run_all_postprocess.py" \
        "$study" \
        --camera "$CAMERA" \
        --dog-depth-max "$DOG_DEPTH_MAX" \
        --annotations-csv "$ANNOTATIONS" \
        > "$log_file" 2>&1 || true

    if grep -q "STUDY DONE" "$log_file" 2>/dev/null; then
        shift_count=$(grep -c "Z shift" "$log_file" 2>/dev/null || echo 0)
        echo "  Result: OK ($shift_count trials shifted)"
        OK=$((OK + 1))
    else
        echo "  Result: FAILED - retrying..."
        retry_log="$LOG_DIR/${study_name}_zshift_retry.log"
        python3 "$PROJECT_DIR/run_all_postprocess.py" \
            "$study" \
            --camera "$CAMERA" \
            --dog-depth-max "$DOG_DEPTH_MAX" \
            --annotations-csv "$ANNOTATIONS" \
            > "$retry_log" 2>&1 || true

        if grep -q "STUDY DONE" "$retry_log" 2>/dev/null; then
            echo "  Result: OK (on retry)"
            OK=$((OK + 1))
        else
            echo "  Result: FAILED"
            FAIL=$((FAIL + 1))
        fi
    fi
done

echo ""
echo "=========================================="
echo "  DONE: OK=$OK SKIP=$SKIP FAIL=$FAIL / $TOTAL"
echo "  $(date)"
echo "=========================================="
