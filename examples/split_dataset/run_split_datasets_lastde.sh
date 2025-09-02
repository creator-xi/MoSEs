#!/bin/bash
# This script runs the split_datasets.py with the 'lastde' criterion.

CUDA_VISIBLE_DEVICES=0 python split_datasets.py \
  --input_directory ./data/doc4split \
  --output_directory ./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_lastde \
  --embedding_type encode \
  --embedding_model_name BAAI/bge-m3 \
  --scoring_model_name EleutherAI/gpt-neo-2.7B \
  --batch_size 500 \
  --crit_type lastde \
  --embed_size 4 \
  --epsilon 8 \
  --tau_prime 15