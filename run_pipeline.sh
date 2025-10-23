#!/bin/bash
# Pointing Gesture Analysis Pipeline Runner
# Uses the point_production conda environment

PYTHON="/opt/anaconda3/envs/point_production/bin/python"

# Check if Python environment is available
if [ ! -f "$PYTHON" ]; then
    echo "❌ Error: Python environment not found at $PYTHON"
    echo "Please update the PYTHON variable in this script to point to your environment"
    exit 1
fi

# Display usage if no arguments
if [ $# -eq 0 ]; then
    echo "Pointing Gesture Analysis Pipeline"
    echo "=================================="
    echo ""
    echo "Usage: $0 <step> [options]"
    echo ""
    echo "Steps:"
    echo "  0 - Data Loading & Target Detection UI"
    echo "  2 - Skeleton Extraction (with UI)"
    echo "  3 - Subject Extraction (SAM2)"
    echo "  4 - Pointing Calculation"
    echo ""
    echo "Examples:"
    echo "  $0 0                              # Launch data loading UI"
    echo "  $0 2                              # Launch skeleton extraction UI"
    echo "  $0 2 ui                           # Launch skeleton extraction UI"
    echo "  $0 2 trial_input/trial_1/cam1     # Batch process skeletons"
    echo ""
    exit 0
fi

STEP=$1
shift  # Remove first argument, keep the rest

case $STEP in
    0)
        echo "🚀 Launching Step 0: Data Loading UI"
        $PYTHON step0_data_loading/ui_data_loader.py "$@"
        ;;
    2)
        echo "🚀 Launching Step 2: Skeleton Extraction"
        if [ "$1" == "ui" ] || [ -z "$1" ]; then
            echo "📊 Starting interactive UI..."
            $PYTHON step2_skeleton_extraction/ui_skeleton_extractor.py
        else
            echo "📁 Batch processing trial: $1"
            $PYTHON step2_skeleton_extraction/batch_processor.py --trial "$1" --output trial_output
        fi
        ;;
    3)
        echo "🚀 Running Step 3: Subject Extraction"
        echo "⚠️ SAM2 not yet implemented. See step3_subject_extraction/README.md"
        ;;
    4)
        echo "🚀 Running Step 4: Pointing Calculation"
        echo "⚠️ Calculation pipeline not yet implemented. See step4_calculation/README.md"
        ;;
    *)
        echo "❌ Unknown step: $STEP"
        echo "Valid steps: 0, 2, 3, 4"
        exit 1
        ;;
esac
