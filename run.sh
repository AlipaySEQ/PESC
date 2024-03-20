PRETRAINED_DIR="roc_pretrained"
DATE_DIR="0127"


seed=52




EPOCH_NUM=20


OUTPUT_DIR=output_0128/srf_epoch_num_${EPOCH_NUM}_$seed
 

#export http_proxy=http://127.0.0.1:8888 export https_proxy=http://127.0.0.1:8888
#-m torch.distributed.launch --master_port=453$(($RANDOM%90+10)) --nproc_per_node=2
#CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=2 python src/run.py \
#python src/run.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=453$(($RANDOM%90+10)) --nproc_per_node=8  src/run.py \
--model_type rocbert \
--model_name_or_path $PRETRAINED_DIR \
--pinyin_dict_path "roc_pretrained/pinyin_dict.json" \
--output_dir $OUTPUT_DIR  \
--do_train   \
--data_dir $DATE_DIR \
--train_file train.pkl \
--predict_file test.pkl \
--order_metric sent-detect-f1  \
--metric_reverse  \
--num_save_ckpts 5 \
--max_seq_length 128 \
--remove_unused_ckpts  \
--per_gpu_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--per_gpu_eval_batch_size 50  \
--learning_rate 5e-5 \
--num_train_epochs $EPOCH_NUM  \
--seed $seed \
--warmup_steps 10000  \
--eval_all_checkpoints \
--overwrite_output_dir






# for((i=0;i<=${EPOCH_NUM}-1;i++))
# do
# year=1


# python src/test.py \
# --device "cuda:0" \
# --ckpt_dir $OUTPUT_DIR \
# --data_dir $DATE_DIR \
# --testset_year $year \
# --ckpt_num $i \
# --output_dir ${OUTPUT_DIR}_result

# done