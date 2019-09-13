#!/bin/bash
pwd

# mkdir -p ../../models/snli_tuning
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning_nli.py --output_dir snli_tuning --num_train_epochs 10 --snli

# mkdir -p ../../models/mi_nli
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning_nli.py --output_dir mi_nli --num_train_epochs 10
CUDA_VISIBLE_DEVICES=2 python retrieve_candidate.py
echo 'Finished.'
