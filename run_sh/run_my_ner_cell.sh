WANDB_PROJECT="my-joint-avg-ner"
WANDB_NAME="$3"
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --master_port $2 --nproc_per_node=1 examples/run_myxfun_ser_cell.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --overwrite_output_dir \
        --output_dir $3 \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps 100 \
        --load_best_model_at_end \
        --metric_for_best_model accuracy \
        --save_total_limit 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 2000 \
        --logging_steps 5 \
        --warmup_ratio 0.1 \
        --fp16