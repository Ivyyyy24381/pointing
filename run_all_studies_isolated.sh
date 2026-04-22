#!/bin/bash
# =============================================================================
# FULL REPROCESS + POST-PROCESS + ARM VERIFICATION PIPELINE
# =============================================================================
# Runs each study in an isolated subprocess with timeout to handle MediaPipe
# cleanup hangs. Includes automatic retry for failures.
#
# What this does:
#   1. Fix targets (within-study reference, with cross-study fallback)
#   2. Reprocess pointing analysis (with arm overrides from annotations CSV)
#   3. Apply dog depth filter (only keep frames where dog is near start)
#   4. Remove bad human frames (handler in bottom 1/4 of image)
#   5. Regenerate all plots
#   6. Verify arm alignment (detected vs annotated)
#
# Usage:
#   bash run_all_studies_isolated.sh
# =============================================================================

OUTPUT_DIR="/home/tigerli/Documents/pointing_data/output"
ANNOTATIONS="/home/tigerli/Documents/pointing_data/PVPO_Production_Data - Output_Tracker.csv"
RAW_ROOT="/home/tigerli/Documents/pointing_data"
LOG_DIR="/home/tigerli/Documents/pointing_data/logs"
PROJECT_DIR="/home/tigerli/Documents/GitHub/pointing"
CAMERA="cam1"
DOG_DEPTH_MAX="2.5"
TIMEOUT=180  # 3 minutes per study (actual processing ~30-60s, rest is exit hang)

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  FULL REPROCESS + POST-PROCESS PIPELINE"
echo "  $(date)"
echo "  Timeout: ${TIMEOUT}s per study"
echo "  Camera: $CAMERA"
echo "  Dog depth max: ${DOG_DEPTH_MAX}m"
echo "  Annotations: $ANNOTATIONS"
echo "=========================================="

# ── STEP 1: Process each study in isolation ──
run_study() {
    local study="$1"
    local log_file="$2"

    timeout "$TIMEOUT" python3 "$PROJECT_DIR/run_all_postprocess.py" \
        "$study" \
        --camera "$CAMERA" \
        --dog-depth-max "$DOG_DEPTH_MAX" \
        --annotations-csv "$ANNOTATIONS" \
        --raw-data-root "$RAW_ROOT" \
        > "$log_file" 2>&1

    # Check if study actually completed (look for "STUDY DONE" in log)
    if grep -q "STUDY DONE" "$log_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

TOTAL=$(ls -d "$OUTPUT_DIR"/*/ 2>/dev/null | wc -l)
CURRENT=0
OK=0
FAIL=0
FAILED_STUDIES=()

for study in "$OUTPUT_DIR"/*/; do
    study_name=$(basename "$study")
    log_file="$LOG_DIR/${study_name}.log"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "[$CURRENT/$TOTAL] Processing: $study_name"

    run_study "$study" "$log_file"
    if [ $? -eq 0 ]; then
        echo "  Result: OK"
        OK=$((OK + 1))
    else
        echo "  Result: FAILED (attempt 1) - retrying..."
        retry_log="$LOG_DIR/${study_name}_retry.log"
        run_study "$study" "$retry_log"
        if [ $? -eq 0 ]; then
            echo "  Result: OK (on retry)"
            cp "$retry_log" "$log_file"  # Keep the good log
            OK=$((OK + 1))
        else
            echo "  Result: FAILED after retry"
            echo "  Last output:"
            tail -5 "$log_file" | sed 's/^/    /'
            FAIL=$((FAIL + 1))
            FAILED_STUDIES+=("$study_name")
        fi
    fi
done

# ── STEP 2: Run arm verification across all studies ──
echo ""
echo "=========================================="
echo "  ARM ALIGNMENT VERIFICATION"
echo "=========================================="
python3 "$PROJECT_DIR/run_all_postprocess.py" \
    "$OUTPUT_DIR" \
    --camera "$CAMERA" \
    --annotations-csv "$ANNOTATIONS" \
    --verify-only \
    2>&1

# ── SUMMARY ──
echo ""
echo "=========================================="
echo "  SUMMARY: $OK/$TOTAL succeeded, $FAIL failed"
if [ ${#FAILED_STUDIES[@]} -gt 0 ]; then
    echo "  Failed studies:"
    for s in "${FAILED_STUDIES[@]}"; do
        echo "    - $s"
    done
    echo ""
    echo "  Re-run failed studies individually:"
    for s in "${FAILED_STUDIES[@]}"; do
        echo "    python3 $PROJECT_DIR/run_all_postprocess.py $OUTPUT_DIR/$s --camera $CAMERA --dog-depth-max $DOG_DEPTH_MAX --annotations-csv \"$ANNOTATIONS\" --raw-data-root $RAW_ROOT"
    done
fi
echo "  Logs: $LOG_DIR"
echo "  $(date)"
echo "=========================================="
