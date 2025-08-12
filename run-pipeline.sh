#!/bin/bash

# Default values
source_recording=""
output_file="transactions_categorized.csv"
transactions_uncategorized_file="transactions_uncategorized.csv"
stitched_output_file="stitched_output.png"
# All other arguments will be passed through to the python scripts
passthrough_args=""

# The first argument is the source recording
if [[ -n "$1" && ! "$1" =~ ^- ]]; then
    source_recording="$1"
    shift
fi

# The second argument is the output file path
if [[ -n "$1" && ! "$1" =~ ^- ]]; then
    output_file="$1"
    shift
fi

# All remaining arguments are passed through
passthrough_args="$@"

if [ -z "$source_recording" ]; then
    echo "Error: source recording is a required parameter."
    echo "Usage: $0 <path_to_recording> [output_file] [options]"
    echo "Options:"
    echo "  -s, --statement-type <type>   Sets the statement type (e.g., uobone, trust)"
    echo "  --roi <x,y,w,h>               Overrides the ROI from the config"
    echo "  --debug-roi                   Debugs the ROI and exits"
    echo "  ...and other script-specific flags."
    exit 1
fi

echo "Processing recording: $source_recording"
echo "Outputting categorized transactions to: $output_file"

# The passthrough_args variable must be unquoted to be interpreted correctly
python stitcher.py "$source_recording" "$stitched_output_file" $passthrough_args

# If --debug-roi was used, stitcher.py has already saved the debug image and exited.
# The pipeline should not continue.
# We check if --debug-roi is in the passthrough arguments.
if [[ " ${passthrough_args} " =~ " --debug-roi " ]]; then
    echo "Exiting after --debug-roi."
    exit 0
fi

# Pass through relevant arguments to the OCR parser.
# Note: We don't need to pass all stitcher args, but it's simpler and harmless.
python ocr_parser.py "$stitched_output_file" "$transactions_uncategorized_file" $passthrough_args

# The categorizer doesn't need special arguments from the command line yet.
python categorize_transactions.py transaction_model.pkl "$transactions_uncategorized_file" "$output_file"
