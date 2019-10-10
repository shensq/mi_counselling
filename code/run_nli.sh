#!/bin/bash
pwd
echo 'Training sentence matching model'
# CUDA_VISIBLE_DEVICES=3 python gpt_tuning_nli.py --model_dir snli_tuning --eval --snli
# CUDA_VISIBLE_DEVICES=3 python gpt_tuning_nli.py --model_dir mi_nli --eval

# mkdir -p ../../models/snli_tuning
# CUDA_VISIBLE_DEVICES=3 python gpt_tuning_nli.py --output_dir snli_tuning --num_train_epochs 10 --snli

mkdir -p ../models/mi_nli
python gpt_tuning_nli.py --model_dir 345M_Alex --output_dir mi_nli --num_train_epochs 10
# python gpt_tuning_nli.py --model_dir snli_tuning --output_dir mi_nli --num_train_epochs 10
python retrieve_candidate.py


echo 'Finished.'
