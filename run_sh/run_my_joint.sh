CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --master_port 45663 --nproc_per_node=2 examples/run_myxfun_joint.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --overwrite_output_dir \
        --output_dir ./mytmp/split-jointmean \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps 100 \
        --load_best_model_at_end \
        --save_total_limit 2 \
        --metric_for_best_model f1 \
        --per_device_eval_batch_size 4 \
        --per_device_train_batch_size 4 \
        --do_train \
        --do_eval \
        --do_predict \
        --lang zh \
        --max_steps 5000 \
        --logging_steps 5 \
        --warmup_ratio 0.1 \
        --fp16