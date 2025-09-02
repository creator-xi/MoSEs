#!/bin/bash
# This script runs the split_datasets.py with the 'fast' criterion.

CUDA_VISIBLE_DEVICES=0 python split_datasets.py \
  --input_directory ./data/doc4split \
  --output_directory ./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast \
  --embedding_type encode \
  --embedding_model_name BAAI/bge-m3 \
  --scoring_model_name EleutherAI/gpt-neo-2.7B \
  --batch_size 500 \
  --crit_type fast
