WANDB_PROJECT="my-joint-re-loss"
WANDB_NAME="$4"
echo $1,$2
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --master_port $3 --nproc_per_node=1 examples/run_myxfun_joint.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --overwrite_output_dir \
        --output_dir $4 \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps 100 \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --metric_for_best_model f1 \
        --per_device_eval_batch_size 8 \
        --per_device_train_batch_size 8 \
        --do_train \
        --do_eval \
        --do_predict \
        --lang zh \
        --num_train_epochs 100 \
        --logging_steps 5 \
        --warmup_ratio 0.1 \
        --fp16