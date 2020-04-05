#!/bin/bash

export logits_file=/data/corpus/f10000.bin
export eval_file=/data/corpus/dev.txt

export output_dir=/data/NER_self_training
export BS=96
export EP=4.0
export LR=2e-5
export WR=0.1
export WD=0.1
#export CUDA_VISIBLE_DEVICES=0
#python examples/ner/run_ner_strain.py \
#       --model_type bert --model_name_or_path bert-base-uncased --output_dir $output_dir \
#       --max_seq_length 128  --do_eval --do_lower_case --per_gpu_train_batch_size $BS \
#       --per_gpu_eval_batch_size $64 --learning_rate $LR --weight_decay $WD --num_train_epochs $EP \
#       --warmup_ratio $WR --logits_file $logits_file --eval_file $eval_file --overwrite_output_dir \
#       --fp16 --fp16_opt_level O2


python -m torch.distributed.launch \
        --nproc_per_node 4 examples/ner/run_ner_strain.py \
        --model_type bert --model_name_or_path bert-base-uncased --output_dir $output_dir \
        --max_seq_length 128  --do_eval --do_lower_case --per_gpu_train_batch_size $BS \
        --per_gpu_eval_batch_size $64 --learning_rate $LR --weight_decay $WD --num_train_epochs $EP \
        --warmup_ratio $WR --logits_file $logits_file --eval_file $eval_file --overwrite_output_dir \
        --fp16 --fp16_opt_level O2
