# rm -rf ~/.cache/huggingface/datasets
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 45663 --nproc_per_node=2 examples/run_xfun_re.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --output_dir ./tmp/test-ner-all-RE-2 \
        --do_train \
        --do_eval \
        --lang zh \
        --additional_langs all \
        --num_train_epochs 100 \
        --per_device_train_batch_size 2 \
        --learning_rate 1e-5 \
        --warmup_ratio 0.1 \
        --fp16
