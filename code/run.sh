#!/bin/bash
pwd

# mkdir -p ../../models/CR_only
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir CR_only --num_train_epochs 5
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir CR_only --output_dir CR_only

# mkdir -p ../../models/SR_only
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir SR_only --num_train_epochs 5
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir SR_only --output_dir SR_only

mkdir -p ../../models/Smoking_only
CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir Smoking_only --num_train_epochs 5 
CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir Smoking_only --output_dir Smoking_only

mkdir -p ../../models/Diet_only
CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir Diet_only --num_train_epochs 5 
CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir Diet_only --output_dir Diet_only

# mkdir -p ../../models/mi_tuned
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir mi_tuned --num_train_epochs 5
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir mi_tuned --output_dir mi_tuned

echo 'Finished.'