#!/bin/bash
export data_dir=/data/tri_training/data
export model_path=/data/tri_training_out/twitter_lm_pt
export output_dir=/data/tri_training_out/fine_tuned
export CUDA_VISIBLE_DEVICES=1
export LR=5e-5
export WD=0.1
export WR=0.1
export EP=10.0
export BS=96

python examples/ner/run_twitter.py \
        --data_dir $data_dir --model_type bert --config_name bert-base-uncased --tokenizer_name bert-base-uncased --model_name_or_path bert-base-uncased --output_dir $output_dir \
        --do_lower_case --max_seq_length 128 --evaluate_during_training --learning_rate $LR --num_train_epochs $EP --supervised_\
        --per_gpu_train_batch_size $BS --per_gpu_eval_batch_size $BS --weight_decay $WD --warmup_ratio $WR --logging_steps 157 --save_steps 157 \
        --do_train --do_predict --do_eval \
        --overwrite_output_dir --fp16 --fp16_opt_level O2
