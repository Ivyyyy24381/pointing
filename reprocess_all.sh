#!/bin/bash
# =============================================================================
# MASTER REPROCESSING SCRIPT
# =============================================================================
# Reprocesses ALL studies from raw data with the updated pipeline:
#   - crop_top_ratio=0.6 (exclude handler from skeleton detection)
#   - YOLO target detection
#   - SAM3 dog detection
#   - Post-processing: fix targets (with cross-study fallback), arm overrides,
#     dog depth filter, arm alignment verification
#
# Raw data is on the external SSD. Output goes to local disk.
#
# Usage:
#   bash reprocess_all.sh             # Process all (skip already-done studies)
#   bash reprocess_all.sh --force     # Reprocess everything from scratch
#   bash reprocess_all.sh --cleanup   # Only flatten raw data, don't process
#   bash reprocess_all.sh --postonly   # Only run post-processing + verification
#
# To resume after interruption, just re-run — it skips studies with existing
# output unless --force is passed.
# =============================================================================

# NOTE: Do NOT use set -e here. MediaPipe SIGABRT (exit 134) propagates through
# timeout and would kill the entire script, even with || true.
set +e

# ── Configuration ──
SSD_RAW="/media/tigerli/Extreme SSD/pointing_data/own_point_production"
LOCAL_STAGING="/home/tigerli/Documents/pointing_data"
OUTPUT_DIR="/home/tigerli/Documents/pointing_data/output"
ANNOTATIONS_LOCAL="/home/tigerli/Documents/pointing_data/PVPO_Production_Data - Output_Tracker.csv"
LOG_DIR="/home/tigerli/Documents/pointing_data/logs"
PROJECT_DIR="/home/tigerli/Documents/GitHub/pointing"
CAMERA="cam1"
DOG_DEPTH_MAX="2.5"
TIMEOUT=0  # No timeout (0 = unlimited). SAM3 can be slow for many trials.
FORCE=false
CLEANUP_ONLY=false
POST_ONLY=false

# Parse args
for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
        --cleanup) CLEANUP_ONLY=true ;;
        --postonly) POST_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ── Check SSD ──
if [ ! -d "$SSD_RAW" ]; then
    echo "ERROR: SSD not mounted at: $SSD_RAW"
    echo "Please plug in the Extreme SSD and try again."
    exit 1
fi

# Copy annotations locally
ANNOTATIONS_SSD="$SSD_RAW/PVPO_Production_Data - Output_Tracker.csv"
if [ -f "$ANNOTATIONS_SSD" ]; then
    cp "$ANNOTATIONS_SSD" "$ANNOTATIONS_LOCAL"
    echo "Copied annotations CSV to local disk"
fi

if [ ! -f "$ANNOTATIONS_LOCAL" ]; then
    echo "ERROR: Annotations CSV not found at: $ANNOTATIONS_LOCAL"
    exit 1
fi

# =============================================================================
# PHASE 0: CLEAN UP RAW DATA STRUCTURE
# =============================================================================
# Some studies have an extra nesting level (extracted from zip with wrapper dir).
# This flattens them so all studies have trial_* directly inside.

echo ""
echo "=========================================="
echo "  PHASE 0: CLEAN UP RAW DATA STRUCTURE"
echo "=========================================="

flatten_if_nested() {
    local dir="$1"
    local clean_name="$2"
    local name=$(basename "$dir")

    # Check if trials are directly inside
    if ls -d "$dir"/trial_* &>/dev/null; then
        echo "  $name: OK (direct trials)"
        return
    fi

    # Check for single nested subfolder with trials
    local nested=""
    for sub in "$dir"/*/; do
        if ls -d "$sub"/trial_* &>/dev/null; then
            nested="$sub"
            break
        fi
    done

    if [ -z "$nested" ]; then
        echo "  $name: WARNING - no trial_* found at any level"
        return
    fi

    nested_name=$(basename "$nested")
    echo "  $name: NESTED ($nested_name) -> flattening..."

    # Move nested contents up: dir/nested/trial_* -> dir/trial_*
    # First move nested dir to a temp name to avoid conflicts
    local tmp="$dir/__flatten_tmp__"
    mv "$nested" "$tmp"
    # Move all trial dirs up
    mv "$tmp"/trial_* "$dir/" 2>/dev/null || true
    # Move any other contents too (like metadata files)
    for item in "$tmp"/*; do
        if [ -e "$item" ]; then
            mv "$item" "$dir/" 2>/dev/null || true
        fi
    done
    rmdir "$tmp" 2>/dev/null || rm -rf "$tmp"

    # Rename the outer dir if it has a different name than desired
    if [ -n "$clean_name" ] && [ "$name" != "$clean_name" ]; then
        local parent=$(dirname "$dir")
        if [ ! -d "$parent/$clean_name" ]; then
            mv "$dir" "$parent/$clean_name"
            echo "    Renamed: $name -> $clean_name"
        fi
    fi

    echo "    Done ($(ls -d "$dir"/trial_* 2>/dev/null | wc -l) trials)"
}

# Flatten known nested studies
flatten_if_nested "$SSD_RAW/BDL265_Elle_OWN026_PVPO_05-010" "BDL265_Elle_OWN026_PVPO_05"
flatten_if_nested "$SSD_RAW/BDL338_Ari_OWN032_PVPO_11-003" "BDL338_Ari_OWN032_PVPO_11"
flatten_if_nested "$SSD_RAW/BDL375_Ozzie_OWN033_PVPO_12-006" "BDL375_Ozzie_OWN033_PVPO_12"
flatten_if_nested "$SSD_RAW/BDL423_Hami_OWN034_PVPO_13PVPO-012" "BDL423_Hami_OWN034_PVPO_13PVPO"

# Check all studies are now flat
echo ""
echo "  Verifying all studies have direct trial_* folders:"
ALL_CLEAN=true
for raw in "$SSD_RAW"/BDL*/; do
    name=$(basename "$raw")
    trial_count=$(ls -d "$raw"/trial_* 2>/dev/null | wc -l)
    if [ "$trial_count" -eq 0 ]; then
        echo "    PROBLEM: $name has 0 trials"
        ALL_CLEAN=false
    else
        printf "    %-55s %2d trials\n" "$name" "$trial_count"
    fi
done

if [ "$ALL_CLEAN" = false ]; then
    echo ""
    echo "  WARNING: Some studies still have issues. Check manually."
fi

if [ "$CLEANUP_ONLY" = true ]; then
    echo ""
    echo "  Done (cleanup only mode)."
    exit 0
fi

# =============================================================================
# PHASE 1: FULL PROCESSING FROM RAW DATA
# =============================================================================

if [ "$POST_ONLY" = true ]; then
    echo ""
    echo "  Skipping Phase 1 (--postonly mode)"
else

echo ""
echo "=========================================="
echo "  PHASE 1: PROCESS FROM RAW DATA"
echo "=========================================="

# Auto-discover all BDL* studies on the SSD (now that they're flattened)
TOTAL=0
OK=0
FAIL=0
SKIP=0
CURRENT=0
FAILED_STUDIES=()

for RAW_PATH in "$SSD_RAW"/BDL*/; do
    RAW_NAME=$(basename "$RAW_PATH")

    # Check it actually has trials
    trial_count=$(ls -d "$RAW_PATH"/trial_* 2>/dev/null | wc -l)
    if [ "$trial_count" -eq 0 ]; then
        continue
    fi

    # Count how many cam1 trials exist (some trials might not have cam1)
    cam1_count=0
    for t in "$RAW_PATH"/trial_*/; do
        if [ -d "$t/cam1/color" ]; then
            cam1_count=$((cam1_count + 1))
        fi
    done

    TOTAL=$((TOTAL + 1))
    OUT_NAME="${RAW_NAME}_output"
    OUT_PATH="$OUTPUT_DIR/$OUT_NAME"
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "[$CURRENT] $RAW_NAME ($trial_count trials, $cam1_count with cam1)"

    # Check if already fully processed (skip unless --force)
    if [ "$FORCE" = false ] && [ -d "$OUT_PATH" ]; then
        skel_count=$(find "$OUT_PATH" -name "skeleton_2d.json" -path "*/cam1/*" 2>/dev/null | wc -l)
        if [ "$skel_count" -ge "$cam1_count" ]; then
            echo "  Already fully processed ($skel_count/$cam1_count skeleton files) - SKIP"
            SKIP=$((SKIP + 1))
            continue
        fi
    fi

    # Retry loop: MediaPipe crashes after ~5 trials due to C-level double-free.
    # The skip-if-complete logic in batch_process_study.py means each re-run
    # picks up where the crash left off.
    MAX_RETRIES=10
    for attempt in $(seq 1 $MAX_RETRIES); do
        # Check how many cam1 trials are already done
        skel_count=0
        if [ -d "$OUT_PATH" ]; then
            skel_count=$(find "$OUT_PATH" -name "skeleton_2d.json" -path "*/cam1/*" 2>/dev/null | wc -l)
        fi
        # Also check staging area (from current run)
        CREATED_OUTPUT="$LOCAL_STAGING/${RAW_NAME}_output"
        if [ -d "$CREATED_OUTPUT" ]; then
            staging_count=$(find "$CREATED_OUTPUT" -name "skeleton_2d.json" -path "*/cam1/*" 2>/dev/null | wc -l)
            skel_count=$((skel_count + staging_count))
        fi

        if [ "$skel_count" -ge "$cam1_count" ]; then
            echo "  All $cam1_count cam1 trials processed!"
            break
        fi

        echo "  Attempt $attempt/$MAX_RETRIES ($skel_count/$cam1_count done so far)"

        # Merge any previous staging output into OUTPUT_DIR before retry
        if [ -d "$CREATED_OUTPUT" ]; then
            mkdir -p "$OUT_PATH"
            cp -rn "$CREATED_OUTPUT"/* "$OUT_PATH/" 2>/dev/null || true
            rm -rf "$CREATED_OUTPUT"
        fi

        # Make output dir look like it's in the staging area so Python can find
        # previously processed trials (for skip-if-complete)
        if [ -d "$OUT_PATH" ]; then
            ln -sfn "$OUT_PATH" "$CREATED_OUTPUT"
        fi

        # Create symlink to raw data
        SYMLINK_PATH="$LOCAL_STAGING/$RAW_NAME"
        rm -f "$SYMLINK_PATH"
        ln -s "$RAW_PATH" "$SYMLINK_PATH"

        log_file="$LOG_DIR/${OUT_NAME}_process_attempt${attempt}.log"
        echo "  Raw: $RAW_PATH"

        # Run full pipeline (skips already-processed trials automatically)
        # Timeout: 30 min per attempt. Generous for processing, catches MediaPipe exit hang.
        # MediaPipe/EGL hangs on process exit (C-level cleanup) — study is done before hang.
        timeout 1800 python3 "$PROJECT_DIR/batch_process_study.py" \
            "$SYMLINK_PATH" \
            --cameras cam1 \
            --use-sam3 \
            --subject dog \
            > "$log_file" 2>&1 || true

        # Merge newly created output into OUTPUT_DIR
        rm -f "$CREATED_OUTPUT" 2>/dev/null  # Remove symlink first
        if [ -d "$CREATED_OUTPUT" ]; then
            # Real directory was created (new trials processed)
            mkdir -p "$OUT_PATH"
            cp -rn "$CREATED_OUTPUT"/* "$OUT_PATH/" 2>/dev/null || true
            rm -rf "$CREATED_OUTPUT"
        fi

        # Clean up symlink
        rm -f "$SYMLINK_PATH"

        # If log shows study completed, no need to retry
        if grep -q "STUDY COMPLETE" "$log_file" 2>/dev/null; then
            echo "  Study completed successfully!"
            break
        fi
    done

    # Final merge: ensure all staging output is in OUTPUT_DIR
    CREATED_OUTPUT="$LOCAL_STAGING/${RAW_NAME}_output"
    if [ -d "$CREATED_OUTPUT" ] && [ ! -L "$CREATED_OUTPUT" ]; then
        mkdir -p "$OUT_PATH"
        cp -rn "$CREATED_OUTPUT"/* "$OUT_PATH/" 2>/dev/null || true
        rm -rf "$CREATED_OUTPUT"
    fi
    # Clean up any leftover symlink
    if [ -L "$CREATED_OUTPUT" ]; then
        rm -f "$CREATED_OUTPUT"
    fi

    # Final check
    final_skel=$(find "$OUT_PATH" -name "skeleton_2d.json" -path "*/cam1/*" 2>/dev/null | wc -l)
    if [ "$final_skel" -ge "$cam1_count" ]; then
        echo "  Result: OK ($final_skel/$cam1_count trials)"
        OK=$((OK + 1))
    elif [ "$final_skel" -gt 0 ]; then
        echo "  Result: PARTIAL ($final_skel/$cam1_count trials after $MAX_RETRIES attempts)"
        FAIL=$((FAIL + 1))
        FAILED_STUDIES+=("$OUT_NAME ($final_skel/$cam1_count)")
    else
        echo "  Result: FAILED (no trials processed)"
        tail -5 "$log_file" 2>/dev/null | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        FAILED_STUDIES+=("$OUT_NAME")
    fi
done

echo ""
echo "=========================================="
echo "  PHASE 1 DONE: OK=$OK  SKIP=$SKIP  FAIL=$FAIL / $TOTAL"
echo "=========================================="

fi  # end of POST_ONLY check

# =============================================================================
# PHASE 2: POST-PROCESSING (isolated per study)
# =============================================================================
echo ""
echo "=========================================="
echo "  PHASE 2: POST-PROCESSING"
echo "=========================================="

PP_OK=0
PP_FAIL=0
PP_TOTAL=$(ls -d "$OUTPUT_DIR"/*_output/ 2>/dev/null | wc -l)
PP_CURRENT=0
PP_FAILED=()

for study in "$OUTPUT_DIR"/*_output/; do
    study_name=$(basename "$study")
    log_file="$LOG_DIR/${study_name}_postprocess.log"
    PP_CURRENT=$((PP_CURRENT + 1))

    echo ""
    echo "[$PP_CURRENT/$PP_TOTAL] Post-processing: $study_name"

    python3 "$PROJECT_DIR/run_all_postprocess.py" \
        "$study" \
        --camera "$CAMERA" \
        --dog-depth-max "$DOG_DEPTH_MAX" \
        --annotations-csv "$ANNOTATIONS_LOCAL" \
        > "$log_file" 2>&1 || true

    if grep -q "STUDY DONE" "$log_file" 2>/dev/null; then
        echo "  Result: OK"
        PP_OK=$((PP_OK + 1))
    else
        echo "  Result: FAILED - retrying..."
        retry_log="$LOG_DIR/${study_name}_postprocess_retry.log"
        python3 "$PROJECT_DIR/run_all_postprocess.py" \
            "$study" \
            --camera "$CAMERA" \
            --dog-depth-max "$DOG_DEPTH_MAX" \
            --annotations-csv "$ANNOTATIONS_LOCAL" \
            > "$retry_log" 2>&1 || true

        if grep -q "STUDY DONE" "$retry_log" 2>/dev/null; then
            echo "  Result: OK (on retry)"
            PP_OK=$((PP_OK + 1))
        else
            echo "  Result: FAILED after retry"
            tail -3 "$log_file" 2>/dev/null | sed 's/^/    /'
            PP_FAIL=$((PP_FAIL + 1))
            PP_FAILED+=("$study_name")
        fi
    fi
done

# =============================================================================
# PHASE 3: ARM ALIGNMENT VERIFICATION
# =============================================================================
echo ""
echo "=========================================="
echo "  PHASE 3: ARM ALIGNMENT VERIFICATION"
echo "=========================================="

python3 "$PROJECT_DIR/run_all_postprocess.py" \
    "$OUTPUT_DIR" \
    --camera "$CAMERA" \
    --annotations-csv "$ANNOTATIONS_LOCAL" \
    --verify-only \
    2>&1

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "=========================================="
echo "  FINAL SUMMARY"
echo "=========================================="
if [ "$POST_ONLY" = false ]; then
    echo "  Phase 1 (Processing):      OK=$OK  SKIP=$SKIP  FAIL=$FAIL / $TOTAL"
fi
echo "  Phase 2 (Post-processing): OK=$PP_OK  FAIL=$PP_FAIL / $PP_TOTAL"
if [ ${#FAILED_STUDIES[@]} -gt 0 ]; then
    echo ""
    echo "  Failed in Phase 1:"
    for s in "${FAILED_STUDIES[@]}"; do
        echo "    - $s"
    done
fi
if [ ${#PP_FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  Failed in Phase 2:"
    for s in "${PP_FAILED[@]}"; do
        echo "    - $s"
    done
fi
echo ""
echo "  Logs: $LOG_DIR/"
echo "  Output: $OUTPUT_DIR/"
echo "  $(date)"
echo "=========================================="
