export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"
devices="0,1,2"

ckpt_path='/data/huggingface/qwen2_0.5B'

deepspeed --include localhost:${devices} \
    src/entry_point/zero_inference_backend_without_trainer.py \
    --deepspeed configs/deepspeed_config_stage3_inference.json \
    --ckpt_path ${ckpt_path} \
    --base_port 17860
