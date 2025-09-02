#!/bin/bash

# Example script to test the SAR model

CUDA_VISIBLE_DEVICES=0 python SAR.py test \
    --model_name "BAAI/bge-m3" \
    --embedding_type encode \
    --subcentroids_model_name weights/model_output_main_c16/subcentroids_head_epoch100.pt \
    --class_names_path weights/model_output_main_c16/class_names.json \
    --input_text "This is a test sentence to be classified."
