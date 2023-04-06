from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from transformers.file_utils import ModelOutput


@dataclass
class ReOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    ner_loss:Optional[torch.FloatTensor] = None
    re_loss:Optional[torch.FloatTensor] = None
    log_var_ner:Optional[torch.FloatTensor] = None
    log_var_re:Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pred_entities: Optional[Dict] = None
    entities: Optional[Dict] = None
    relations: Optional[Dict] = None
    pred_relations: Optional[Dict] = None
    gt_relations: Optional[Dict] = None
