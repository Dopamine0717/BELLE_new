#! /bin/bash
# module switch compiler/dtk/22.10
module switch compiler/dtk/23.04
# source ~/env.sh

JOB_NAME=${JOB_NAME:-"train"}

GPUS=${1:-8}
model_name_or_path=${2:-"/public/home/hpctest_xjtu/data/hf_home/uergpt2-chinese-cluecorpussmall"}  # /public/home/hpctest_xjtu/data/hf_home/bloom-7b1 # or bloomz-7b1-mt
output_dir=${3:-"work_dirs/power_gpt2_100m_lr2e-4"}
batch_size=${4:-4}
gradient_accumulation_steps=${5:-2}
epochs=${6:-3}
learning_rate=${7:-2e-4}
train_file=${8:-'power_conv_train.json'}
validation_file=${9:-'power_conv_dev.json'}
negative=${10:-0.2}
PY_ARGS=${@:3}

GPUS_PER_NODE=${GPUS:-4}
if [ $GPUS_PER_NODE -ge 4 ]; then
  GPUS_PER_NODE=4
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

# HOME_PATH=/work/home/ac3y91rcdl
HOME_PATH=/work/home/acehekbmzh/
# export TORCH_EXTENSIONS_DIR=${HOME_PATH}/.cache/torch_extensions_env_pt2
export HF_HOME=${HOME_PATH}/data/hf_home

# train_file=${HOME_PATH}/codes/belle_debug/data/$train_file
# validation_file=${HOME_PATH}/codes/belle_debug/data/$validation_file
mkdir -p ${output_dir}

cache_dir=hf_power_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

set -x 

# export MASTER_PORT=9912

# partition=xahdtest
partition=xahdnormal

SRUN_ARGS=''
OUTPUT=$output_dir
now=$(date +"%Y%m%d_%H%M%S")

# -o $OUTPUT/exp_logger-%j-$now.log
# -w c14r2n[00,02-08]  c13r4n05,c14r2n[06-08],c14r4n01
# c13r4n05 好像有问题 -w c13r4n05

#! The following scripts add LORA now !
srun --partition=${partition} $SRUN_ARGS  \
    --job-name=${JOB_NAME} -n$GPUS --gres=dcu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1  --cpus-per-task=4  -o $OUTPUT/exp_logger-%j-$now.log \
    --mem=110000 \
    python train/src/entry_point/sft_train.py \
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
    --torch_dtype "float16" \
    --fp16 \
    --eval_steps 500 \
   # --use_flash_attention
   # --resume_from_checkpoint ...
