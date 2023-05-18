import collections
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers.trainer_utils import EvalPrediction, PredictionOutput, speed_metrics
from transformers.utils import logging

from .funsd_trainer import FunsdTrainer
from sklearn.metrics import f1_score, accuracy_score


import collections
import gc
import math
import os
import time
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import evaluate
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    init_deepspeed,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)




if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)


class XfunSerTrainer(FunsdTrainer):
    pass


class XfunReTrainer(FunsdTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_names.append("relations")
        self.test_dataset = None

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        labels = tuple(inputs.get(name) for name in self.label_names)
        return outputs, labels

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        re_labels = None
        pred_relations = None
        pred_entities = None
        entities = None
        for step, inputs in enumerate(dataloader):
            outputs, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            re_labels = labels[1] if re_labels is None else re_labels + labels[1]
            pred_relations = (
                outputs.pred_relations if pred_relations is None else pred_relations + outputs.pred_relations
            )
            entities = outputs.entities if entities is None else entities + outputs.entities
            pred_entities = outputs.pred_entities if pred_entities is None else pred_entities + outputs.pred_entities
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        gt_relations = []
        pt_relations = []
        # if 'pred_label' in entities[0].keys():
        #     for b in range(len(pred_relations)):
        #         rel_sent = []
        #         for rel in pred_relations[b]:
        #         # for head, tail in zip(pred_relations[b]["head"], pred_relations[b]["tail"]):
        #             # if rel['head_type'] == 1 and (rel['tail_type'] == 2 or rel['tail_type'] == 4):
        #             #     rel_sent.append(rel)
        #             if rel['head_pred_type'] == 1 and (rel['tail_pred_type'] == 2 or rel['tail_pred_type'] == 4):
        #                 rel_sent.append(rel)
        #         pt_relations.append(rel_sent)
        # else:
        #     pt_relations = pred_relations
        for b in range(len(re_labels)):
            rel_sent = []
            for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
                rel = {}
                rel["head_id"] = head
                rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
                rel["head_type"] = entities[b]["label"][rel["head_id"]]

                rel["tail_id"] = tail
                rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
                rel["tail_type"] = entities[b]["label"][rel["tail_id"]]

                rel["type"] = 1

                rel_sent.append(rel)

            gt_relations.append(rel_sent)
        ner_labels,ner_pred_labels = None,None
        
        # for i in range(len(entities)):
        #     if len(entities[i]['label']) == 0:
        #         continue
        #     ner_labels = entities[i]['label'] if ner_labels is None else ner_labels + entities[i]['label']
        #     ner_pred_labels = pred_entities[i]['label'] if ner_pred_labels is None else ner_pred_labels + pred_entities[i]['label']
        #     if len(entities[i]['label']) != len(pred_entities[i]['label']):
        #         print("wrong")      
        # ner_metrics = accuracy_score(ner_labels,ner_pred_labels)
        re_metrics = self.compute_metrics(EvalPrediction(predictions=pred_relations, label_ids=gt_relations))

        re_metrics = {
            # "ner_precision":ner_metrics,
            "precision": re_metrics["ALL"]["p"],
            "recall": re_metrics["ALL"]["r"],
            "f1": re_metrics["ALL"]["f1"],
        }
        re_metrics[f"{metric_key_prefix}_loss"] = outputs.loss.mean().item()

        metrics = {}

        # # Prefix all keys with metric_key_prefix + '_'
        for key in list(re_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = re_metrics.pop(key)
            else:
                metrics[f"{key}"] = re_metrics.pop(key)
        if metric_key_prefix == "test":
            return pred_relations, gt_relations, metrics
        return metrics
    
    def predict(
        self,
        test_dataset: Optional[Dataset] = None, # args.local_rank
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",  
    ):
        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self.args.local_rank = -1
        test_dataloader = self.get_test_dataloader(test_dataset)
        # 不使用多卡
        # self.args.local_rank = torch.distributed.get_rank()

        start_time = time.time()

        predictions, labels, metrics = self.prediction_loop(
            test_dataloader,
            description="test",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(test_dataset if test_dataset is not None else self.eval_dataset)
        metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return predictions, labels, metrics
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self.args.local_rank = -1
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # self.args.local_rank = torch.distributed.get_rank()

        start_time = time.time()

        metrics = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
    
        return metrics



    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if "ner_loss" in outputs.keys():
                ner_loss = outputs["ner_loss"] if isinstance(outputs, dict) else outputs[0]
                re_loss = outputs["re_loss"] if isinstance(outputs, dict) else outputs[0]
                
                self.log({"ner_loss": ner_loss.item(),"re_loss":re_loss.item()})

        return (loss, outputs) if return_outputs else loss
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)
            predictions, labels, test_metrics = self.predict(self.test_dataset)
            self._report_to_hp_search(trial, epoch, test_metrics)
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    
class XfunCellSerTrainer(FunsdTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_names.append("entities")
        self.test_dataset = None

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        labels = tuple(inputs.get(name) for name in self.label_names)
        return outputs, labels

    def prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        entity_labels = None
        preds = None
        loss = None
        for step, inputs in enumerate(dataloader):
            outputs, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            
            # temp_labels = torch.tensor([i['label'] for i in labels[1]],device=self.args.device)
            for b in range(len(inputs['entities'])):
                temp = inputs['entities'][b]['label']
                if len(temp) == 0:
                    continue
                # continue
                loss = outputs.loss.mean() if loss == None else loss + outputs.loss.mean()
                entity_labels =  (temp, ) if entity_labels is None else entity_labels + (temp,)
                preds = (outputs['logits'][b], ) if preds is None else preds + (outputs['logits'][b], )
            # 
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        
        if self.compute_metrics is not None and preds is not None and entity_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=entity_labels))
        else:
            metrics = {}
       # re_metrics[f"{metric_key_prefix}_loss"] = outputs.loss.mean().item()
        metrics[f"{metric_key_prefix}_loss"] = loss.item()
        metrics[f"{metric_key_prefix}_accuracy"] = metrics["accuracy"]
        return PredictionOutput(preds, entity_labels, metrics)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self.args.local_rank = -1
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # test_dataloader = self.get_eval_dataloader(test_dataset)
        description="Evaluation"
        # if test_dataset != None:
        #     description="Prediction"
        #     metric_key_prefix = "test"


        start_time = time.time()

        outputPrediction = self.prediction_loop(
            eval_dataloader,
            description=description,
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        metrics = outputPrediction.metrics

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics
    
    
    # def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
    #     if self.control.should_log:
    #         logs: Dict[str, float] = {}
    #         tr_loss_scalar = tr_loss.item()
    #         # reset tr_loss to zero
    #         tr_loss -= tr_loss

    #         logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
    #         logs["learning_rate"] = self._get_learning_rate()

    #         self._total_loss_scalar += tr_loss_scalar
    #         self._globalstep_last_logged = self.state.global_step

    #         self.log(logs)

    #     metrics = None
    #     if self.control.should_evaluate:
    #         metrics = self.evaluate()
    #         self._report_to_hp_search(trial, epoch, metrics)
    #         metrics = self.evaluate(self.test_dataset,metric_key_prefix="test")
    #         self._report_to_hp_search(trial, epoch, metrics)

    #     if self.control.should_save:
    #         self._save_checkpoint(model, trial, metrics=metrics)
    #         self.control = self.callback_handler.on_save(self.args, self.state, self.control)


class XfunJointTrainer(FunsdTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_names.append("relations")
        self.test_dataset = None

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        labels = tuple(inputs.get(name) for name in self.label_names)
        return outputs, labels

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        re_labels = None
        pred_relations = None
        pred_entities = None
        entities = None
        for step, inputs in enumerate(dataloader):
            outputs, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            re_labels = labels[1] if re_labels is None else re_labels + labels[1]
            pred_relations = (
                outputs.pred_relations if pred_relations is None else pred_relations + outputs.pred_relations
            )
            entities = outputs.entities if entities is None else entities + outputs.entities
            pred_entities = outputs.pred_entities if pred_entities is None else pred_entities + outputs.pred_entities
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        gt_relations = []
        for b in range(len(re_labels)):
            rel_sent = []
            for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
                rel = {}
                rel["head_id"] = head
                rel["head"] = (entities[b]["start"][rel["head_id"]], entities[b]["end"][rel["head_id"]])
                rel["head_type"] = entities[b]["label"][rel["head_id"]]

                rel["tail_id"] = tail
                rel["tail"] = (entities[b]["start"][rel["tail_id"]], entities[b]["end"][rel["tail_id"]])
                rel["tail_type"] = entities[b]["label"][rel["tail_id"]]

                rel["type"] = 1

                rel_sent.append(rel)

            gt_relations.append(rel_sent)
        ner_labels,ner_pred_labels = None,None
        for i in range(len(entities)):
            if len(entities[i]['label']) == 0:
                continue
            ner_labels = entities[i]['label'] if ner_labels is None else ner_labels + entities[i]['label']
            ner_pred_labels = pred_entities[i]['label'] if ner_pred_labels is None else ner_pred_labels + pred_entities[i]['label']
            if len(entities[i]['label']) != len(pred_entities[i]['label']):
                print("wrong in len")      
        ner_metrics = accuracy_score(ner_labels,ner_pred_labels)
        re_metrics = self.compute_metrics(EvalPrediction(predictions=pred_relations, label_ids=gt_relations))

        re_metrics = {
            "ner_precision":ner_metrics,
            "precision": re_metrics["ALL"]["p"],
            "recall": re_metrics["ALL"]["r"],
            "f1": re_metrics["ALL"]["f1"],
        }
        re_metrics[f"{metric_key_prefix}_loss"] = outputs.loss.mean().item()

        metrics = {}

        # # Prefix all keys with metric_key_prefix + '_'
        for key in list(re_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = re_metrics.pop(key)
            else:
                metrics[f"{key}"] = re_metrics.pop(key)
        if metric_key_prefix == "test":
            return pred_relations, gt_relations, metrics
        return metrics
    
    def predict(
        self,
        test_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",  
    ):
        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self.args.local_rank = -1
        test_dataloader = self.get_test_dataloader(test_dataset)
        # 不使用多卡
        self.args.local_rank = torch.distributed.get_rank()

        start_time = time.time()

        predictions, labels, metrics = self.prediction_loop(
            test_dataloader,
            description="test",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(test_dataset if test_dataset is not None else self.eval_dataset)
        metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return predictions, labels, metrics
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        self.args.local_rank = -1
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.args.local_rank = torch.distributed.get_rank()

        start_time = time.time()

        metrics = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
    
        return metrics



    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        inputs['epoch'] = self.state.epoch
        # self.state.epoch
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if "ner_loss" in outputs.keys():
                ner_loss = outputs["ner_loss"] if isinstance(outputs, dict) else outputs[0]
                re_loss = outputs["re_loss"] if isinstance(outputs, dict) else outputs[0]
                log_var_ner = outputs["log_var_ner"] if isinstance(outputs, dict) else outputs[0]
                log_var_re = outputs["log_var_re"] if isinstance(outputs, dict) else outputs[0]
                self.log({"ner_loss": ner_loss.item(),"re_loss":re_loss.item(),"log_var_ner":log_var_re.item(),"log_var_re":log_var_re.item()})

        return (loss, outputs) if return_outputs else loss
    

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)
            predictions, labels, test_metrics = self.predict(self.test_dataset)
            self._report_to_hp_search(trial, epoch, test_metrics)
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)