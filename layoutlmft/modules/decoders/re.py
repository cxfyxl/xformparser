import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

class REDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(5, config.hidden_size, scale_grad_by_freq=True)
        self.group_emb = nn.Embedding(50, config.hidden_size // 4, scale_grad_by_freq=True)
        self.index_emb= nn.Embedding(50, config.hidden_size // 4, scale_grad_by_freq=True)
        self.mlp_dim = config.hidden_size * 2   + config.hidden_size // 2
        self.use_specialid = True
        self.del_begin = -5
        self.del_end = 50
        # self.minend = -6
        # self.maxend = 50
        projection = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.mlp_dim // 2, self.mlp_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(self.mlp_dim // 4, 2)
        self.loss_fct = CrossEntropyLoss()

    def build_seq(self,relations,entities,b,key):
        all_possible_relations = set(
            [
                (i, j)
                for i in range(len(entities[b][key]))
                for j in range(len(entities[b][key]))
                # if i != j
                if entities[b][key][i] == 1 and (entities[b][key][j] == 2 or entities[b][key][j] == 4) # entities[b]["label"][j] == 2 or entities[b]["label"][j] == 4
                # and (j - i) <= self.del_end
                # and (j - i) >= self.del_begin
            ]
        )
        positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
        if positive_relations.issubset(all_possible_relations):
            pass
        else:
            # pass
            print("wrong")
        if self.training:
            all_possible_relations = all_possible_relations | positive_relations
        if len(all_possible_relations) == 0:
            all_possible_relations = gt_possible_relations = set([(0, 1)])
        
        negative_relations = all_possible_relations - positive_relations
        positive_relations = set([i for i in positive_relations if i in all_possible_relations])
        reordered_relations = list(positive_relations) + list(negative_relations)
        relation_per_doc = {"head": [], "tail": [], "label": []}
        relation_per_doc["head"] = [i[0] for i in reordered_relations]
        relation_per_doc["tail"] = [i[1] for i in reordered_relations]
        relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
        assert len(relation_per_doc["head"]) != 0
        return relation_per_doc
    

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        gt_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0],"group_id":[0, 0],"index_id":[0, 0]}
            if "pred_label" not in entities[b].keys():
                relation_per_doc = self.build_seq(relations,entities,b,"label")
            else:
                relation_per_doc = self.build_seq(relations,entities,b,"pred_label")
                # gt_relation_per_doc = self.build_seq(relations,entities,b,"label")
            new_relations.append(relation_per_doc)
            # gt_relations.append(gt_relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            if "pred_label" in entities.keys():
                rel["head_pred_type"] = entities["pred_label"][rel["head_id"]]
                rel["tail_pred_type"] = entities["pred_label"][rel["tail_id"]]
            pred_relations.append(rel)
        return pred_relations

    def entity_cell_forward(self, hidden_states, b, entities_start_index, entities_end_index, entities_labels,head_entities, \
                            entities_group_index,entities_index_index):
        batch_size, max_n_words, context_dim = hidden_states.size()
        head_start_index = entities_start_index[head_entities]
        head_end_index = entities_end_index[head_entities]
        head_label = entities_labels[head_entities]
        head_group_id = entities_group_index[head_entities]
        head_index_id = entities_index_index[head_entities]
        # if entities_labels_logits != None:
        #     head_logits = entities_labels_logits[head_entities]
        head_entity_repr = None
        for i in enumerate(head_start_index):
            index = i[0]
            start_index, end_index = head_start_index[index], head_end_index[index]
            temp_repr = hidden_states[b][start_index:end_index].mean(dim=0).view(1,context_dim)
            # temp_repr = hidden_states[b][start_index:end_index].max(dim=0)[0].view(1,context_dim)
            head_entity_repr = temp_repr if head_entity_repr == None else torch.cat([head_entity_repr,temp_repr], dim=0)
        return head_group_id, head_index_id, head_label, head_entity_repr

    def forward(self, hidden_states,  entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_end_index = torch.tensor(entities[b]["end"], device=device)
            entities_group_index = torch.tensor(entities[b]["group_id"], device=device)
            entities_index_index = torch.tensor(entities[b]["index_id"], device=device)
            if "pred_label" in entities[b].keys():
                entities_labels = torch.tensor(entities[b]["pred_label"], device=device)
            else:
                entities_labels = torch.tensor(entities[b]["label"], device=device)
            entities_labels_logits = None
            
            # if "label_logits" in entities[b].keys():
            #     entities_labels_logits = torch.tensor(entities[b]["label_logits"], device=device)

            head_group_id, head_index_id, head_label, head_entity_repr = self.entity_cell_forward(hidden_states, b, entities_start_index, \
                                                   entities_end_index, entities_labels , head_entities,entities_group_index,entities_index_index)

            tail_group_id, tail_index_id, tail_label, tail_entity_repr = self.entity_cell_forward(hidden_states, b, entities_start_index, \
                                        entities_end_index, entities_labels , tail_entities,entities_group_index,entities_index_index)
            head_label_repr = self.entity_emb(head_label)
            tail_label_repr = self.entity_emb(tail_label)

            # head_repr, tail_repr = head_entity_repr, tail_entity_repr
            # head_repr = head_entity_repr
            # tail_repr = tail_entity_repr
            if not self.use_specialid:
                head_repr = torch.cat(
                    (head_entity_repr, head_label_repr),
                    dim=-1,
                )
                tail_repr = torch.cat(
                    (tail_entity_repr, tail_label_repr),
                    dim=-1,
                )
            else:
                head_group_repr,head_index_repr = self.group_emb(head_group_id), self.index_emb(head_index_id)
                tail_group_repr,tail_index_repr = self.group_emb(tail_group_id), self.index_emb(tail_index_id)
                head_repr = torch.cat(
                    (head_entity_repr, head_label_repr,head_group_repr,head_index_repr),
                    dim=-1,
                )
                tail_repr = torch.cat(
                    (tail_entity_repr, tail_label_repr,tail_group_repr,tail_index_repr),
                    dim=-1,
                )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            # logits = self.LeakyReLU(logits)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        if torch.isnan(loss).sum() != 0:
            print("wrong! loss")
            print(loss)
        # assert torch.isnan(loss).sum() == 0, print(loss)
        return loss, all_pred_relations
    
    

    
class CellDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log_var_re = torch.nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.entity_emb = nn.Embedding(5, config.hidden_size, scale_grad_by_freq=True)
        self.group_emb = nn.Embedding(50, config.hidden_size // 4, scale_grad_by_freq=True)
        self.index_emb= nn.Embedding(50, config.hidden_size // 4, scale_grad_by_freq=True)
        # self.entity_emb_rand = nn.Embedding(5, config.hidden_size, scale_grad_by_freq=True)
        self.mlp_dim = config.hidden_size * 2 # + config.hidden_size // 2
        projection = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.mlp_dim // 2, self.mlp_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.adaptive_loss = False
        self.multi_task = False
        
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(self.mlp_dim // 4, 2)
        self.loss_fct = CrossEntropyLoss()
        print(f"{self.__class__}:adaptive_loss:{self.adaptive_loss}\tmulti_task:{self.multi_task}")
    def criterion(self, y_pred, y_true, log_vars,device):
        precision = torch.exp(-log_vars)
        diff = self.loss_fct(y_pred,y_true)
        loss = torch.sum(precision * diff + log_vars, -1)
        return loss

    def build_relation(self, relations, pred_entities, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = pred_entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0],"group_id":[0, 0],"index_id":[0, 0]}
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if i != j
                    if entities[b]["label"][i] == 1 and (entities[b]["label"][j] == 2 or entities[b]["label"][j] == 4) # or entities[b]["label"][j] == 4 
                    # and (entities[b]["id"][j] - entities[b]["id"][i] <= 50)
                    # and (entities[b]["id"][j] - entities[b]["id"][i] >= -5)
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            if not positive_relations.issubset(all_possible_relations):
                print("not subset")
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations
    
    
    def entity_cell_forward(self, hidden_states, b, entities_start_index, entities_end_index, entities_labels,head_entities, \
                            entities_group_index,entities_index_index):
        batch_size, max_n_words, context_dim = hidden_states.size()
        head_start_index = entities_start_index[head_entities]
        head_end_index = entities_end_index[head_entities]
        head_label = entities_labels[head_entities]
        head_group_id = entities_group_index[head_entities]
        head_index_id = entities_index_index[head_entities]
        # if entities_labels_logits != None:
        #     head_logits = entities_labels_logits[head_entities]
        head_entity_repr = None
        for i in enumerate(head_start_index):
            index = i[0]
            start_index, end_index = head_start_index[index], head_end_index[index]
            temp_repr = hidden_states[b][start_index:end_index].mean(dim=0).view(1,context_dim)
            # temp_repr = hidden_states[b][start_index:end_index].max(dim=0)[0].view(1,context_dim)
            head_entity_repr = temp_repr if head_entity_repr == None else torch.cat([head_entity_repr,temp_repr], dim=0)
        return head_group_id, head_index_id, head_label, head_entity_repr

   
    def forward(self, hidden_states, pred_entities, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        relations, entities = self.build_relation(relations, pred_entities, entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_end_index = torch.tensor(entities[b]["end"], device=device)
            # 
            entities_group_index = torch.tensor(entities[b]["group_id"], device=device)
            entities_index_index = torch.tensor(entities[b]["index_id"], device=device)
            if self.multi_task:
                entities_labels = torch.tensor(entities[b]["label"], device=device)
            else:
                entities_labels = torch.tensor(pred_entities[b]["label"], device=device)
            
            entities_labels_logits = None
            
            head_group_id, head_index_id, head_label, head_entity_repr = self.entity_cell_forward(hidden_states, b, entities_start_index, \
                                                   entities_end_index, entities_labels , head_entities,entities_group_index,entities_index_index)

            tail_group_id, tail_index_id, tail_label, tail_entity_repr = self.entity_cell_forward(hidden_states, b, entities_start_index, \
                                        entities_end_index, entities_labels , tail_entities,entities_group_index,entities_index_index)
            head_label_repr = self.entity_emb(head_label)
            tail_label_repr = self.entity_emb(tail_label)
            # head_group_repr,head_index_repr = self.group_emb(head_group_id), self.index_emb(head_index_id)
            # tail_group_repr,tail_index_repr = self.group_emb(tail_group_id), self.index_emb(tail_index_id)
                
            
            # if entities_labels_logits != None:
            #     head_label_repr = torch.einsum('sn,nf->snf', head_logits, self.entity_emb.weight).mean(dim=1)
            #     tail_label_repr = torch.einsum('sn,nf->snf', tail_logits, self.entity_emb.weight).mean(dim=1)
            # else:
            head_label_repr = self.entity_emb(head_label)
            tail_label_repr = self.entity_emb(tail_label)
            
            # head_label_repr = torch.randn(head_label_repr.size(), device=device)
            # tail_label_repr = torch.randn(tail_label_repr.size(), device=device)
            
            # head_repr = head_entity_repr
            # tail_repr = tail_entity_repr
            # head_repr = head_entity_repr + head_label_repr
            # tail_repr = tail_entity_repr + tail_label_repr


            # head_repr = torch.cat(
            #     (head_entity_repr, head_label_repr,head_group_repr,head_index_repr),
            #     dim=-1,
            # )
            # tail_repr = torch.cat(
            #     (tail_entity_repr, tail_label_repr,tail_group_repr,tail_index_repr),
            #     dim=-1,
            # )
            head_repr = torch.cat(
                (head_entity_repr, head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (tail_entity_repr, tail_label_repr),
                dim=-1,
            )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            logits = self.rel_classifier(heads, tails)
            if self.adaptive_loss:
                loss += self.criterion(logits,relation_labels,self.log_var_re,device)
            else:
                loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations