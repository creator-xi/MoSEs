#!/bin/bash

# Set common arguments
RESULT_FILE_PATH="./logs/main_c16/lastde"
DATASETS_FOLDER="./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_lastde_train"
SAR_PATH="./weights/model_output_main_c16/subcentroids_head_epoch100.pt"
CLASS_PATH="./weights/model_output_main_c16/class_names.json"

# Array of test files.
# User has indicated to run only wp_dataset.json for now.
# Others are commented out but can be enabled by removing the '#'.
TEST_FILES=(
    "./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_lastde_test/cmv_dataset.json"
    "./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_lastde_test/sci_dataset.json"
    "./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_lastde_test/wp_dataset.json"
    "./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_lastde_test/xsum_dataset.json"
)

# Loop through the test files and run the evaluation
for FILE_PATH in "${TEST_FILES[@]}"
do
    echo "Running lastde evaluation for $FILE_PATH"
    python CTE.py \
        --file_path "$FILE_PATH" \
        --result_file_path "$RESULT_FILE_PATH" \
        --datasets_folder "$DATASETS_FOLDER" \
        --sar_path "$SAR_PATH" \
        --class_path "$CLASS_PATH"
    echo "---------------------------------"
done

echo "All lastde evaluations complete."
