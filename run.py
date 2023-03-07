import subprocess,os
os.chdir('/home/zhanghang-s21/data/layoutlmft/') 
import time

run_sh_ner = f"""
WANDB_PROJECT="my-joint-avg-re"
WANDB_NAME="./mytmp/avg-jointmean-8-100-reloss"
CUDA_VISIBLE_DEVICES={0},{1} python -m torch.distributed.launch --master_port 33881 --nproc_per_node=2 examples/run_myxfun_joint.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --overwrite_output_dir \
        --output_dir {"./mytmp/avg-jointmean-8-100-reloss"} \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps 100 \
        --load_best_model_at_end \
        --save_total_limit 1 \
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
"""



run_sh_ner_2 = f"""
WANDB_PROJECT="my-joint-avg-re"
WANDB_NAME="./mytmp/avg-jointmean-8-100-reloss+nerloss+div2"
CUDA_VISIBLE_DEVICES={0},{1} python -m torch.distributed.launch --master_port 55772 --nproc_per_node=2 examples/run_myxfun_joint.py \
        --model_name_or_path microsoft/layoutxlm-base \
        --overwrite_output_dir \
        --output_dir {"./mytmp/avg-jointmean-8-100-reloss+nerloss+div2"} \
        --evaluation_strategy steps --eval_steps 100 \
        --save_strategy steps --save_steps 100 \
        --load_best_model_at_end \
        --save_total_limit 1 \
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
"""
log = open('/home/zhanghang-s21/data/layoutlmft/mylog/somefile.txt', 'w')

subprocess.run(run_sh_ner
, shell=True,stdout=log)
time.sleep(600)
subprocess.run(run_sh_ner_2
, shell=True,stdout=log)