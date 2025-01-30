export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1" 
export TOKENIZERS_PARALLELISM="false"
export NCCL_DEBUG="INFO"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 61000 zeroshot_eval.py \
    --model_name_or_path= \
    --dataset_name=none \
    --output_dir=./predict \
    --per_device_eval_batch_size=4 \
    --eval_accumulation_steps=1 \
    --remove_unused_columns=False \
    --ddp_find_unused_parameters=False \
    --do_train=False \
    --do_eval=True \
    --do_predict=False \
    --use_liger=True \
    --attn_implementation=flash_attention_2 \
    --batch_eval_metrics=True \
    --group_by_length=True \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --deepspeed=./config/zero3.json
