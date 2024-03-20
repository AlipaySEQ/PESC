PRETRAINED_DIR="pretrained"
DATE_DIR="data"
OUTPUT_DIR="output"

CUDA_VISIBLE_DEVICES=0 python src/run.py \
    --model_type bert \
    --model_name_or_path $PRETRAINED_DIR \
    --output_dir $OUTPUT_DIR  \
    --do_train --do_eval --do_predict  \
    --data_dir $DATE_DIR \
    --train_file train.pkl \
    --dev_file dev.pkl \
    --order_metric sent-detect-f1  \
    --metric_reverse  \
    --num_save_ckpts 5 \
    --remove_unused_ckpts  \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 50  \
    --learning_rate 5e-5 \
    --num_train_epochs 10  \
    --seed 17 \
    --warmup_steps 10000  \
    --eval_all_checkpoints \
    --overwrite_output_dir
