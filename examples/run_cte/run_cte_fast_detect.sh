#!/bin/bash

# Set common arguments
RESULT_FILE_PATH="./logs/main_c16/fast"
DATASETS_FOLDER="./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast_train"
SAR_PATH="./weights/model_output_main_c16/subcentroids_head_epoch100.pt"
CLASS_PATH="./weights/model_output_main_c16/class_names.json"

# Array of test files
TEST_FILES=(
    "./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast_test/cmv_dataset.json"
    "./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast_test/sci_dataset.json"
    "./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast_test/wp_dataset.json"
    "./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast_test/xsum_dataset.json"
)

# Loop through the test files and run the evaluation
for FILE_PATH in "${TEST_FILES[@]}"
do
    echo "Running evaluation for $FILE_PATH"
    python CTE.py \
        --file_path "$FILE_PATH" \
        --result_file_path "$RESULT_FILE_PATH" \
        --datasets_folder "$DATASETS_FOLDER" \
        --sar_path "$SAR_PATH" \
        --class_path "$CLASS_PATH"
    echo "---------------------------------"
done

echo "All evaluations complete."
