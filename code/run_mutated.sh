#!/bin/bash
pwd
for data in x_y_meta_random_sample  x_y_meta_random_sen  x_y_meta_random_word  x_y_meta_replace_phrase
do
	mkdir -p ../../models/mi_tuned_${data:9}
	CUDA_VISIBLE_DEVICES=2 python gpt_tuning.py --special_input $data --output_dir mi_tuned_${data:9} --num_train_epochs 10
	CUDA_VISIBLE_DEVICES=2 python gpt_sample.py --special_input $data --model_dir mi_tuned_${data:9} --output_dir mi_tuned_${data:9}
done
echo 'Finished.'
