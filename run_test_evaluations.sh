#!/bin/bash

# Script to run multiple evaluation configurations
# Automatically discovers and runs all .yaml configs in specified folders
# Each evaluation runs sequentially (one after another) with proper log redirection

# Dataset config
DATASET_CONFIG="configs/dataset/dataset_10_ex_10.yaml"
# Base directory for evaluation configs
EVAL_BASE_DIR="configs/evaluations/"

# Folders to process (mapping: config_folder -> log_folder)
# Define order of evaluation
ORDERED_FOLDERS=(
    "baseline"
)

declare -A FOLDER_MAP=(
    ["baseline"]="baseline/"
)

# Function to run evaluation
run_evaluation() {
    local config_file=$1
    local log_file=$2
    local config_name=$3
    
    # Create error file path (replace .log with .err)
    local err_file="${log_file%.log}.err"
    
    echo "Starting evaluation: $config_name"
    echo "Config: $config_file"
    echo "Log: $log_file"
    echo "Error log: $err_file"
    echo "---"
    
    python evaluate_dataset.py \
        --dataset-config "$DATASET_CONFIG" \
        --runner-config "$config_file"  #\
        # > "$log_file" 2> "$err_file"
    
    echo "Completed: $config_name"
    echo ""
}

# Create log directories and process each folder
for config_folder in "${ORDERED_FOLDERS[@]}"; do
    log_folder="${FOLDER_MAP[$config_folder]}"
    config_dir="$EVAL_BASE_DIR/$config_folder"
    log_dir="logs/$log_folder"
    
    # Create log directory if it doesn't exist
    mkdir -p "$log_dir"
    
    # Check if config directory exists
    if [ ! -d "$config_dir" ]; then
        echo "Warning: Config directory not found: $config_dir"
        continue
    fi
    
    echo "Processing folder: $config_folder -> $log_folder"
    echo "=========================================="
    
    # Find all .yaml files in the config directory and run them
    for config_file in "$config_dir"/*.yaml; do
        # Check if any files were found (glob might not match)
        if [ ! -f "$config_file" ]; then
            continue
        fi
        
        # Extract filename without path and extension
        config_basename=$(basename "$config_file" .yaml)
        
        # Create log file path
        log_file="$log_dir/$config_basename.log"
        
        # Run the evaluation
        run_evaluation "$config_file" "$log_file" "$config_basename"
        # clean up the tmp dir
        rm -rfd tmp
        
    done
    
    echo ""
done

echo "All evaluations completed."
echo "Check individual log files for details."


