CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 45662 --nproc_per_node=2 examples/run_xfun_re.py \
        --model_name_or_path /home/zhanghang-s21/data/layoutlmft/tmp/test-ner-RE \
        --output_dir /home/zhanghang-s21/data/layoutlmft/tmp/test-ner-RE \
        --do_predict \
        --lang zh