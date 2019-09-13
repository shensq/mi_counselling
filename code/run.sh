#!/bin/bash
pwd

# mkdir -p ../../models/CR_only
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir CR_only --num_train_epochs 5
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir CR_only --output_dir CR_only

# mkdir -p ../../models/SR_only
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir SR_only --num_train_epochs 5
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir SR_only --output_dir SR_only

# mkdir -p ../../models/Smoking_only
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir Smoking_only --num_train_epochs 5 
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir Smoking_only --output_dir Smoking_only

# mkdir -p ../../models/Diet_only
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir Diet_only --num_train_epochs 5 
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir Diet_only --output_dir Diet_only

mkdir -p ../../models/mi_tuned_large
CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --model_dir gpt2_large --output_dir mi_tuned_large --num_train_epochs 10
CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir mi_tuned_large --output_dir mi_tuned_large

# mkdir -p ../../models/mi_tuned_15epo
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir mi_tuned_15epo --num_train_epochs 15
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir mi_tuned_15epo --output_dir mi_tuned_15epo
# CUDA_VISIBLE_DEVICES=2 python retrieve_candidate.py
# mkdir -p ../../models/mi_tuned_aug
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir mi_tuned_aug --num_train_epochs 10 --augment
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir mi_tuned_aug --output_dir mi_tuned_aug --augment

# mkdir -p ../../models/mi_tuned_aug_15epo
# CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --output_dir mi_tuned_aug_15epo --num_train_epochs 15 --augment
# CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --model_dir mi_tuned_aug_15epo --output_dir mi_tuned_aug_15epo --augment
echo 'Finished.'
