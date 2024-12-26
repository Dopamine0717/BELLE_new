JOB_NAME=${JOB_NAME:-"powerchat_sft"}

GPUS=${1:-8}
model_name_or_path=${2:-"/public/home/hpctest_xjtu/data/hf_home/uergpt2-chinese-cluecorpussmall"}  # /public/home/hpctest_xjtu/data/hf_home/bloom-7b1 # or bloomz-7b1-mt
output_dir=${3:-"work_dirs/power_gpt2_100m_lr2e-4"}
batch_size=${4:-4}
gradient_accumulation_steps=${5:-2}
epochs=${6:-3}
learning_rate=${7:-2e-4}
train_file=${8:-'power_conv_train.json'}
validation_file=${9:-'power_conv_dev.json'}

export HF_HOME=/mnt/afs/luohaichen/hf_home
mkdir -p ${output_dir}
cache_dir=/mnt/afs/luohaichen/hf_power_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024
set -x 
OUTPUT=$output_dir
now=$(date +"%Y%m%d_%H%M%S")

torchrun --nproc_per_node=$GPUS --master_port=39718 \
    train/src/entry_point/sft_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --deepspeed train/configs/deepspeed_config_stage3.json \
    --lora_config train/configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${epochs} \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --learning_rate ${learning_rate} \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --evaluation_strategy "steps" \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --report_to "none" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --eval_steps 500 \
    > $OUTPUT/$now.log 2>&1 \
   # --use_flash_attention
   # --resume_from_checkpoint ${output_dir} \
