CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --master_port 45663 --nproc_per_node=2 examples/run_xfun_re.py \
        --model_name_or_path PaddlePaddle/ERNIE-Layout \
        --output_dir ./ernie-layoutx-base-uncased/models/xfund_zh \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 2500 \
        --per_device_train_batch_size 2 \
        --warmup_ratio 0.1 \
        --fp16
