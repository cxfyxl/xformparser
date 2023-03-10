#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import layoutlmft.data.datasets.xfun
import transformers
from transformers import EarlyStoppingCallback
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForCellClassification
from layoutlmft.trainers import XfunSerTrainer, XfunCellSerTrainer
# from layoutlmft.evaluation import re_score
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import evaluate
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score,classification_report

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")


logger = logging.getLogger(__name__)

os.environ["WANDB_PROJECT"] = "my-testdata-ner"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

def main():

    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    datasets = load_dataset(
        '/home/zhanghang-s21/data/layoutlmft/layoutlmft/data/datasets/myxfunsplit_new.py',
        "myxfunsplit_new.zh",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
    )
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "input_ids"
    label_column_name = "labels"

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    # shuffled_ds = datasets.shuffle(seed=training_args.seed)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = LayoutLMv2ForCellClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False
    test_name = "validation"
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        # eval_dataset = datasets["train"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # Metrics
    metric = evaluate.load("/home/zhanghang-s21/data/mycache/metrics/seqeval")
    # metric = load_metric("accuracy")
    # metric = load_metric("/home/zhanghang-s21/data/mycache/metrics/seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions_result,labels_result = [],[]
        for i, pedictions in enumerate(predictions):
            prediction = predictions[i].cpu()
            prediction = np.argmax(prediction, axis=-1).int()
            prediction = prediction.int().numpy().tolist()
            label = labels[i]
            predictions_result.extend(prediction)
            labels_result.extend(label)
        
        # labels = labels.cpu().int().numpy().tolist()
        # predictions_result = [i for i in prediction]
        results = {}
        results["accuracy_score"] = results["eval_accuracy"] = accuracy_score(labels_result,predictions_result)
        results["f1_score_micro"] = f1_score(labels_result,predictions_result,average="micro")
        results["f1_score_macro"] = f1_score(labels_result,predictions_result,average='macro')
        print("classification report:\n{}".format(classification_report(labels_result,predictions_result)))
        # results = metric.compute(predictions=[predictions_result], references=[labels_result])
        

        # if data_args.return_entity_level_metrics:
        #     # Unpack nested dictionaries
        #     final_results = {}
        #     for key, value in results.items():
        #         if isinstance(value, dict):
        #             for n, v in value.items():
        #                 final_results[f"{key}_{n}"] = v
        #         else:
        #             final_results[key] = value
        #     return final_results
        # else:
        #     return {
        #         "precision": results["overall_precision"],
        #         "recall": results["overall_recall"],
        #         "f1": results["overall_f1"],
        #         "accuracy": results["overall_accuracy"],
        # }

        return {
                "precision": results["accuracy_score"],
                "recall": results["accuracy_score"],
                "f1": results["accuracy_score"],
                "accuracy": results["accuracy_score"],
            }
        # return results
 
    from transformers import AdamW
    
    # optimizer = AdamW([{"params":model.classifier.parameters(),"lr":1e-5},
    #                    {"params":model.layoutlmv2.parameters()},
    #                    {"params":model.extractor.parameters()}],lr=2e-5)
    

    trainer = XfunCellSerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # test_dataset=test_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(3, 0.0)]
    )
    trainer.test_dataset = test_dataset
    # trainer.create_optimizer_and_scheduler(trainer.args.max_steps)
    # trainer.optimizer = AdamW([{"params":model.parameters(),"lr":1e-5}],lr=5e-5)
    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        # logger.info("*** test train***")
        # metrics = trainer.evaluate(train_dataset)
        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        new_predictions = []
        for prediction in predictions:
            new_predictions.append(np.argmax(prediction.cpu(), axis=-1).int().tolist())
            # prediction = 


        # Remove ignored index (special tokens)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # predictions, labels, metrics = trainer.predict(train_dataset)
        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, test_name + "_data_test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in new_predictions:
                    writer.write(" ".join(str(v) for v in prediction) + "\n")


        logger.info("*** val ***")

        predictions, labels, metrics = trainer.predict(eval_dataset)
        new_predictions = []
        for prediction in predictions:
            new_predictions.append(np.argmax(prediction.cpu(), axis=-1).int().tolist())
            # prediction = 


        # Remove ignored index (special tokens)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        output_test_predictions_file = os.path.join(training_args.output_dir, test_name + "_data_val_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in new_predictions:
                    writer.write(" ".join(str(v) for v in prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
