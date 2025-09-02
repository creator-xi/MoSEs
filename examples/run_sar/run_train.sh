#!/bin/bash

# Example script to train the SAR model

CUDA_VISIBLE_DEVICES=0 python SAR.py train \
    --model_name "BAAI/bge-m3" \
    --datasets_folder data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast \
    --embedding_type encode \
    --num_epochs 100 \
    --batch_size 32 \
    --num_subcentroids 4 \
    --output_dir weights/model_output_main_c16
