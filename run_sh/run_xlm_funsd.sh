WANDB_PROJECT="xfund-re"
WANDB_NAME="$4"
CUDA_VISIBLE_DEVICES=$1,$2 python -m torch.distributed.launch --master_port $3 --nproc_per_node=2 examples/run_funsd_joint.py \
        --model_name_or_path /home/zhanghang-s21/data/model/layoutxlm-base \
        --overwrite_output_dir \
        --output_dir $4 \
        --evaluation_strategy steps \
        --eval_steps 50 \
        --save_strategy steps \
        --save_steps 50 \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --metric_for_best_model f1 \
        --per_device_eval_batch_size 2 \
        --per_device_train_batch_size 2 \
        --do_train \
        --do_eval \
        --learning_rate 3.5e-5 \
        --num_train_epochs 100 \
        --logging_steps 5 \
        --warmup_ratio 0.1 \
        --fp16