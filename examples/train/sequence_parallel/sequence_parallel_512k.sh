# Env: 8 * A100
# Max Length: 512000
# GPU Memory: 8 * 80GiB, Training Speed 150s/it
# NPROC_PER_NODE=8 \
NPROC_PER_NODE=1
CELOSS_PARALLEL_SIZE=2048 \
accelerate launch swift/cli/sft.py \
    --model Qwen/Qwen3-32B \
    --train_type full \
    --dataset './medical-reasoning.jsonl' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 2 \
    --packing true \
    --rope_scaling yarn \
    --max_length 16384 \
    --max_model_len 16384 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --use_liger_kernel true \
    --save_only_model true \
    --attn_impl flash_attn \
    --use_hf true \
    --load_from_cache_file false \
    --deepspeed zero3 \
    --sequence_parallel_size 8
