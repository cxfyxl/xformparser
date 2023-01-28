CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --master_port 45662 --nproc_per_node=2 examples/run_xfun_ser.py \
        --model_name_or_path /home/zhanghang-s21/data/layoutlmft/tmp/test-myner \
        --output_dir /home/zhanghang-s21/data/layoutlmft/tmp/test-myner \
        --do_predict \
        --lang zh