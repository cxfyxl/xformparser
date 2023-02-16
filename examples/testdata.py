import logging
import os
import sys
import json
import numpy as np
from datasets import ClassLabel, load_dataset

import layoutlmft.data.datasets.xfun
import transformers
from layoutlmft import AutoModelForRelationExtraction
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.data.data_collator import DataCollatorForKeyValueExtraction
from layoutlmft.evaluation import re_score
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import XfunReTrainer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
datasets = load_dataset(
    '/home/zhanghang-s21/data/layoutlmft/layoutlmft/data/datasets/myxfunsplit.py',
    "myxfunsplit.zh",
    # additional_langs=data_args.additional_langs,
    keep_in_memory=True,
)