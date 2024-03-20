DATE_DIR="0127"
CKPT_DIR="output_0128"
seed=52





EPOCH_NUM=20



OUTPUT_DIR=output_0128/srf_epoch_num_${EPOCH_NUM}_$seed



    
# for((i=0;i<=${EPOCH_NUM}-1;i++))
# do 



#     python src/test.py \
#     --device "cuda:0" \
#     --ckpt_dir $OUTPUT_DIR \
#     --data_dir $DATE_DIR \ 
#     --ckpt_num $i \
#     --output_dir ${OUTPUT_DIR}_result
#-m torch.distributed.launch --master_port=453$(($RANDOM%90+10)) --nproc_per_node=4
# done
python src/test.py \
    --device "cuda:6" \
    --ckpt_dir $OUTPUT_DIR \
    --data_dir $DATE_DIR \
    --ckpt_num 19 \
    --output_dir ${OUTPUT_DIR}_result
