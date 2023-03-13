WANDB_PROJECT="my-testdata-re"
WANDB_NAME="$4"
CUDA_VISIBLE_DEVICES=$1,$2 python -m torch.distributed.launch --master_port $3 --nproc_per_node=2 examples/run_myxfun_joint.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --overwrite_output_dir \
        --output_dir $4 \
        --evaluation_strategy steps\
        --eval_steps 100 \
        --save_strategy steps \
        --save_steps 100 \
        --load_best_model_at_end \
        --save_total_limit 3 \
        --metric_for_best_model f1 \
        --per_device_eval_batch_size 4 \
        --per_device_train_batch_size 4 \
        --do_train \
        --do_eval \
        --do_predict \
        --lang zh \
        --num_train_epochs 100 \
        --logging_steps 5 \
        --warmup_ratio 0.1 \
        --fp16