#!/bin/bash

# This script runs various ablation studies using the CTE.py evaluation script.
# The SAR model and class paths should point to a pre-trained model.

# --- Base command without any ablations ---
# This runs the default evaluation.
# python ./CTE.py \
#     --file_path "./data/doc4split/filtered_train_main_1000.csv" \
#     --result_file_path "./logs/ablation/results" \
#     --datasets_folder "./data/doc4split" \
#     --sar_path "./weights/model_output_main_c16/subcentroids_head_epoch1.pt" \
#     --class_path "./weights/model_output_main_c16/class_names.json"


# --- Ablation 1: No Router ---
# This disables the nearest neighbor router and loads the full data partition.
# echo "Running Ablation: No Router"
# python ./CTE.py \
#     --file_path "./data/doc4split/filtered_train_main_1000.csv" \
#     --result_file_path "./logs/ablation/results_no_router" \
#     --datasets_folder "./data/doc4split" \
#     --sar_path "./weights/model_output_main_c16/subcentroids_head_epoch1.pt" \
#     --class_path "./weights/model_output_main_c16/class_names.json" \
#     --no_router


# --- Ablation 2: No PCA ---
# This disables PCA by setting pca_dim to -1.
# echo "Running Ablation: No PCA"
# python ./CTE.py \
#     --file_path "./data/doc4split/filtered_train_main_1000.csv" \
#     --result_file_path "./logs/ablation/results_no_pca" \
#     --datasets_folder "./data/doc4split" \
#     --sar_path "./weights/model_output_main_c16/subcentroids_head_epoch1.pt" \
#     --class_path "./weights/model_output_main_c16/class_names.json" \
#     --pca_dim -1


# --- Ablation 3: No Embedding Features ---
# This uses only condition features.
# echo "Running Ablation: No Embedding"
# python ./CTE.py \
#     --file_path "./data/doc4split/filtered_train_main_1000.csv" \
#     --result_file_path "./logs/ablation/results_no_embedding" \
#     --datasets_folder "./data/doc4split" \
#     --sar_path "./weights/model_output_main_c16/subcentroids_head_epoch1.pt" \
#     --class_path "./weights/model_output_main_c16/class_names.json" \
#     --no_embedding


# --- Ablation 4: No Condition Features ---
# This uses only embedding features.
# echo "Running Ablation: No Condition"
# python ./CTE.py \
#     --file_path "./data/doc4split/filtered_train_main_1000.csv" \
#     --result_file_path "./logs/ablation/results_no_condition" \
#     --datasets_folder "./data/doc4split" \
#     --sar_path "./weights/model_output_main_c16/subcentroids_head_epoch1.pt" \
#     --class_path "./weights/model_output_main_c16/class_names.json" \
#     --no_condition

# --- Ablation 5: PCA with 64 dimensions ---
# This runs the evaluation with PCA dimension set to 64.
# echo "Running Ablation: PCA with 64 dimensions"
# python ./CTE.py \
#     --file_path "./data/doc4split/filtered_train_main_1000.csv" \
#     --result_file_path "./logs/ablation/results_pca_64" \
#     --datasets_folder "./data/doc4split" \
#     --sar_path "./weights/model_output_main_c16/subcentroids_head_epoch1.pt" \
#     --class_path "./weights/model_output_main_c16/class_names.json" \
#     --pca_dim 64

echo "Ablation script created. Uncomment the desired command to run."