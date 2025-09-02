#!/bin/bash

python process_csv.py \
  --input_folder ./data/split_datasets_bge_m3_tiny_encode_crit_cond6_main_1000_fast \
  --test_split_size 50 \
  --seed 42
