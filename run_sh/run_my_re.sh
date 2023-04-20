# WANDB_PROJECT="my-testdata-re"
# WANDB_DISABLED=true
WANDB_NAME="$3"
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --master_port $2 --nproc_per_node=1 examples/run_myxfun_re.py \
        --overwrite_output_dir \
        --model_name_or_path microsoft/layoutxlm-base \
        --output_dir $3 \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps 100 \
        --load_best_model_at_end \
        --save_total_limit 2 \
        --metric_for_best_model f1 \
        --do_train \
        --do_eval \
        --do_predict \
        --logging_steps 5 \
        --lang zh \
        --num_train_epochs 100 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --fp16

CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --master_port $2 --nproc_per_node=1 examples/run_xfun_re_infer.py \
        --overwrite_output_dir \
        --model_name_or_path $3 \
        --output_dir $3/test/ \
        --do_eval \
        --do_predict \
        --lang zh \
        --fp16