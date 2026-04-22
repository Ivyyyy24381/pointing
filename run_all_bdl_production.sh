#!/bin/bash
# Run BDL production processing for all subjects.
# Each subject runs in an isolated subprocess with timeout to avoid
# MediaPipe cleanup hangs.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="/media/tigerli/Extreme SSD/pointing_data/pointing_production_BDL"
OUTPUT_DIR="/home/tigerli/Documents/pointing_data/point_production_BDL_output"
TIMEOUT=900  # 15 minutes per subject

mkdir -p "$OUTPUT_DIR"

# Get list of zip files (subjects)
ZIPS=($(ls "$SOURCE_DIR"/*.zip 2>/dev/null | sort))
TOTAL=${#ZIPS[@]}

echo "================================================================"
echo "BDL Production Processing"
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Subjects: $TOTAL"
echo "Timeout: ${TIMEOUT}s per subject"
echo "================================================================"
echo ""

OK=0
FAIL=0
SKIP=0

for i in "${!ZIPS[@]}"; do
    ZIP="${ZIPS[$i]}"
    SUBJECT=$(basename "$ZIP" .zip)
    IDX=$((i + 1))

    # Check if already processed
    TRIAL_COUNT=$(ls -d "$OUTPUT_DIR/$SUBJECT"/trial_* 2>/dev/null | wc -l)
    if [ "$TRIAL_COUNT" -gt 0 ]; then
        echo "[$IDX/$TOTAL] $SUBJECT: already has $TRIAL_COUNT trials, skipping"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo ""
    echo "[$IDX/$TOTAL] Processing: $SUBJECT"
    echo "================================================================"

    # Check disk space (need at least 20GB free)
    FREE_GB=$(df --output=avail /home | tail -1 | awk '{print int($1/1024/1024)}')
    if [ "$FREE_GB" -lt 20 ]; then
        echo "WARNING: Only ${FREE_GB}GB free, need at least 20GB. Stopping."
        break
    fi

    # Run in subprocess with timeout
    timeout "$TIMEOUT" python "$SCRIPT_DIR/batch_reprocess_BDL_production.py" \
        --subject "$SUBJECT" \
        --source "$SOURCE_DIR" \
        --output "$OUTPUT_DIR" \
        2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        OK=$((OK + 1))
        echo "[$IDX/$TOTAL] $SUBJECT: SUCCESS"
    elif [ $EXIT_CODE -eq 124 ]; then
        FAIL=$((FAIL + 1))
        echo "[$IDX/$TOTAL] $SUBJECT: TIMEOUT (${TIMEOUT}s)"
    else
        FAIL=$((FAIL + 1))
        echo "[$IDX/$TOTAL] $SUBJECT: FAILED (exit $EXIT_CODE)"
    fi

    # Clean up work dir if it exists
    rm -rf "/home/tigerli/Documents/pointing_data/_bdl_prod_work/$SUBJECT"
done

echo ""
echo "================================================================"
echo "SUMMARY: $OK succeeded, $FAIL failed, $SKIP skipped / $TOTAL total"
echo "================================================================"

# Generate global CSV
if [ $OK -gt 0 ]; then
    echo "Generating global CSV..."
    python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from batch_reprocess_BDL_production import generate_global_csv
from pathlib import Path
generate_global_csv(Path('$OUTPUT_DIR'))
"
fi
