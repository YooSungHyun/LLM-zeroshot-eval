export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1" 
export TOKENIZERS_PARALLELISM="false"
export NCCL_DEBUG="INFO"

accelerate launch --config_file "./config/fsdp_config.yaml" zeroshot_eval.py \
    --output_dir=./predict \
    --per_device_eval_batch_size=2 \
    --eval_accumulation_steps=1 \
    --remove_unused_columns=False \
    --ddp_find_unused_parameters=False \
    --do_train=False \
    --do_eval=True \
    --do_predict=False \
    --use_liger_kernel=True \
    --batch_eval_metrics=True \
    --group_by_length=True
