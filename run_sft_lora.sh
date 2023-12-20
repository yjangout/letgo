#!/bin/bash
export DEVICES=0,1,2,3
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export NODE_RANK=0
# shellcheck disable=SC2004
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

# Model Parameter
MODEL_NAME=Baichuan2-13B-Chat
if [ "$MODEL_NAME" = "Qwen-7B-Chat" ];then
  export RANK=128
  export ALPHA=256
  export LORA_TARGET="c_attn"
  export TEMPLATE=qwen
elif [ "$MODEL_NAME" = "Qwen-14B-Chat" ];then
  export RANK=128
  export ALPHA=256
  export LORA_TARGET="c_attn"
  export TEMPLATE=qwen
elif [ "$MODEL_NAME" = "Baichuan2-13B-Chat" ]; then
  export RANK=128
  export ALPHA=256
  export LORA_TARGET="W_pack"
  export TEMPLATE=baichuan2
elif [ "$MODEL_NAME" = "Baichuan2-7B-Chat" ]; then
  export RANK=128
  export ALPHA=256
  export LORA_TARGET="W_pack"
  export TEMPLATE=baichuan2
elif [ "$MODEL_NAME" = "Yi-6B-200K" ]; then
  export RANK=128
  export ALPHA=256
  export LORA_TARGET="q_proj,v_proj"
  export TEMPLATE=yi
elif [ "$MODEL_NAME" = "Yi-6B-Chat" ]; then
  export RANK=128
  export ALPHA=256
  export LORA_TARGET="q_proj,v_proj"
  export TEMPLATE=yi
fi

# Common Parameter
export FINETUNE_TYPE=lora
export DATASET="yidai-sft-train"
export TRAIN_EPOCH=8
export BATCH_SIZE=16
export MAX_SEQ_LEN=4096
export USE_SAMPLE=1000000000000
export STAGE=sft
export LR=5e-5
# shellcheck disable=SC2006
# shellcheck disable=SC2003
ACCUMULATE_STEPS=`expr $BATCH_SIZE / $WORLD_SIZE`
RUN_NAME="${STAGE}_${MODEL_NAME}_E${TRAIN_EPOCH}_${FINETUNE_TYPE}_${RANK}_${ALPHA}_${TEMPLATE}_LR_${LR}_SL${MAX_SEQ_LEN}"
REPORT_TO="none" # wandb, none

# shellcheck disable=SC2166
if [ "$MODEL_NAME" = "Qwen-14B-Chat" -o "$MODEL_NAME" = "Baichuan2-13B-Chat" ];then
  deepspeed --include localhost:${DEVICES} --master_port=9901 train_bash.py \
      --deepspeed config/ds_config_zero3.json \
      --stage $STAGE \
      --model_name_or_path "/home/app/jacoblu/downloads/${MODEL_NAME}" \
      --do_train \
      --dataset $DATASET \
      --finetuning_type $FINETUNE_TYPE \
      --lora_rank $RANK \
      --lora_alpha $ALPHA \
      --lora_target $LORA_TARGET \
      --output_dir outputs/$RUN_NAME \
      --max_samples $USE_SAMPLE \
      --cutoff_len $MAX_SEQ_LEN \
      --optim "adamw_torch" \
      --overwrite_cache \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps $ACCUMULATE_STEPS \
      --lr_scheduler_type cosine \
      --logging_steps 20 \
      --save_steps 400 \
      --learning_rate $LR \
      --num_train_epochs $TRAIN_EPOCH \
      --plot_loss \
      --template $TEMPLATE \
      --report_to $REPORT_TO \
      --run_name $RUN_NAME \
      --overwrite_output_dir \
      --ddp_find_unused_parameters False \
      --fp16
else
  deepspeed --include localhost:${DEVICES} --master_port=9901 train_bash.py \
      --stage $STAGE \
      --model_name_or_path "/home/app/jacoblu/downloads/${MODEL_NAME}" \
      --do_train \
      --dataset $DATASET \
      --finetuning_type $FINETUNE_TYPE \
      --lora_rank $RANK \
      --lora_alpha $ALPHA \
      --lora_target $LORA_TARGET \
      --output_dir outputs/$RUN_NAME \
      --max_samples $USE_SAMPLE \
      --cutoff_len $MAX_SEQ_LEN \
      --optim "adamw_torch" \
      --overwrite_cache \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps $ACCUMULATE_STEPS \
      --lr_scheduler_type cosine \
      --logging_steps 5 \
      --save_steps 100 \
      --learning_rate $LR \
      --num_train_epochs $TRAIN_EPOCH \
      --plot_loss \
      --template $TEMPLATE \
      --report_to $REPORT_TO \
      --run_name $RUN_NAME \
      --overwrite_output_dir \
      --ddp_find_unused_parameters False \
      --fp16
fi