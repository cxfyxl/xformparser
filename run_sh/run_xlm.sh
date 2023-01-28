CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --master_port 45662 --nproc_per_node=2  examples/run_xfun_ser.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --output_dir /home/zhanghang-s21/data/layoutlmft/tmp/test-ner \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16