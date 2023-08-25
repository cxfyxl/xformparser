# layoutlmft
**Multimodal (text + layout/format + image) fine-tuning toolkit for document understanding**

## Introduction

## Supported Models
Popular Language Models: BERT, UniLM(v2), RoBERTa, InfoXLM

LayoutLM Family: LayoutLM, LayoutLMv2, LayoutXLM

## Installation

~~~bash
conda create -n layoutlmft python=3.7
conda activate layoutlmft
git clone https://github.com/microsoft/unilm.git
cd unilm
cd layoutlmft
pip install -r requirements.txt
pip install -e .
~~~

## Train
### 概述
大概训练流程是，通过一个包含参数信息脚本运行python程序，一个训练思路对应这一个trainer，一个model文件
### 文件对应位置
- 训练脚本在run_sh目录  
- PLM模型在layoutlmft/layoutlmft/models  
- 下游模型在layoutlmft/layoutlmft/modules  
- trainers配置在layoutlmft/trainers  
- 训练py程序在examples目录  
### 训练指令
```
训练指令结构为
sh 训练脚本 卡编号0 卡编号1 端口号 模型存储文件夹 > LOG文件
```
#### 扩充版数据集训练姐
```
nohup sh run_sh/run_my_joint.sh 0 1 18851 ./mytmp/avg-jointmean-8-100-sorted  > ./mylog/avg-jointmean-8-100-sorted 2>&1 &
```
#### funsd训练集
```
nohup sh run_sh/run_xlm_funsd.sh 0 3 55519 ./myre/0711-funsd-joint > ./myrelog/0711-funsd-joint 2>&1 &
```
#### xfund训练集
##### zh
```
nohup sh run_sh/run_my_joint_xfund.sh 1 3 33232 ./myre/0717-xfund-joint-nodel-nosplit > ./myrelog/0717-xfund-joint-nodel-nosplit   2>&1 &
```
##### all language
```
nohup sh run_sh/run_my_joint_xfund.sh 2 3 33219 ./myre/0628-xfund-joint-lstm-nosplit > ./myrelog/0628-xfund-joint-lstm-nosplit  2>&1 &
```
### 工具
#### Onnx导出
tools/huggingface2onnx.py
```
        {
            "name": "xfund_cell_jointonnx",
            "type": "python",
            "request": "launch",
            "program": "/home/zhanghang-s21/data/layoutlmft/tools/huggingface2onnx.py",
            "console": "integratedTerminal",
            "python": "/home/zhanghang-s21/miniconda3/envs/layoutlmft/bin/python",
            "cwd": "/home/zhanghang-s21/data/layoutlmft",
            "args": [
                "--model_name_or_path", "/home/zhanghang-s21/data/layoutlmft/myre/0618-joint-lstm-softlabel", //microsoft/layoutxlm-base
                "--output_dir", "/home/zhanghang-s21/data/layoutlmft/myre/0618-joint-lstm-softlabel/onnx",
                "--overwrite_output_dir",
                //"--do_train",
                "--do_eval",
                "--do_predict",
                "--lang","zh",
                "--eval_steps","10",
                "--save_steps","10",
                "--evaluation_strategy","steps",
                "--save_strategy","steps",
                "--load_best_model_at_end",
                "--save_total_limit","1",
                "--per_device_eval_batch_size","8",
                "--logging_steps","5",
                "--max_steps","100",
                "--warmup_ratio","0.1",
                //"--fp16",
                "--no_cuda"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }, 
            "justMyCode": false
        },
```