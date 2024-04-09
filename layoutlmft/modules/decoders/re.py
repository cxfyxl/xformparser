import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

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

import math
class MyBilinear(torch.nn.Module):
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in1_features, in2_features), **factory_kwargs))

        # if bias:
        #     self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        # else:
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        # if self.bias is not None:
        #     nn.BatchNorm2dinit.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # y = torch.zeros((input1.shape[0],self.weight.shape[0]))
        # for k in range(self.weight.shape[0]):
        buff1 = torch.matmul(input1, self.weight[0])
        buff1 = buff1 * input2 #torch.matmul(buff, input2)
        buff1 = torch.sum(buff1,dim=1).unsqueeze(1)

        buff2 = torch.matmul(input1, self.weight[1])
        buff2 = buff2 * input2 #torch.matmul(buff, input2)
        buff2 = torch.sum(buff2,dim=1).unsqueeze(1)
        # y[:,k] = buff
        # if self.bias is not None:
        #     y += self.bias
        return torch.cat((buff1,buff2),dim=-1)

    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )


class MyBiaffineAttention(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(MyBiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = MyBilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

import torch


class REEmbeddings(nn.Module):
    def __init__(self, config, use_special = False):
        super().__init__()
        self.use_special = use_special
        self.label_embedding = nn.Embedding(5, config.hidden_size, scale_grad_by_freq=True)
        self.row_embeddding = nn.Embedding(50, config.hidden_size // 4, scale_grad_by_freq=True)
        self.column_embeddding = nn.Embedding(50, config.hidden_size // 4, scale_grad_by_freq=True)
        # self.positionalembedding = PositionalEncoding(config.hidden_size // 2,60)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dim = config.hidden_size + config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.begin_epoch = 150
        self.warm_epoch = 10
        print(f"self.begin_epoch:{self.begin_epoch}")

    def forward(self,label,label_logits,row_id,column_id,epoch):
        # label_embedding = self.label_embedding(label)
        if epoch != None and label_logits != None and epoch>=self.begin_epoch and self.training:
            # soft_label = self.softmax(label_logits)
            soft_embedding = torch.einsum('sn,nf->snf', label_logits, self.label_embedding.weight).mean(dim=1)
            hard_embedding = self.label_embedding(label)
            alpha = min(1,(epoch-self.begin_epoch)/self.warm_epoch)
            label_embedding = alpha * soft_embedding + (1-alpha)*hard_embedding
            # label_embedding = soft_embedding
            # label_embedding = torch.einsum('sn,nf->snf', label_logits, self.label_embedding.weight).mean(dim=1)
            # tail_label_repr = torch.einsum('sn,nf->snf', tail_logits, self.entity_emb.weight).mean(dim=1)
            # else:
        # elif not self.training and label_logits != None:
        #     soft_label = self.softmax(label_logits)
        #     label_embedding = torch.einsum('sn,nf->snf', soft_label, self.label_embedding.weight).mean(dim=1)
        else:
            label_embedding = self.label_embedding(label)
        if self.use_special:
            # print("self.use_special in REEmbeddings")
            row_embeddding = self.row_embeddding(row_id)
            column_embeddding = self.column_embeddding(column_id)
    
            # row_embeddding = self.positionalembedding(row_id)
            # column_embeddding = self.positionalembedding(column_id)
            # final_embedings = label_embedding + torch.cat(
            #     (row_embeddding, column_embeddding),
            #     dim=-1,
            # )
            final_embedings = torch.cat(
                (label_embedding, row_embeddding, column_embeddding),
                dim=-1,
            )
            # final_embedings = label_embedding + row_embeddding + column_embeddding
        else:
            final_embedings = label_embedding
        # final_embedings = self.dropout(final_embedings)
        return final_embedings


class EntityData():
    def __init__(self, entity, device):
        self.entities_start_index = torch.tensor(entity["start"], device=device)
        self.entities_end_index = torch.tensor(entity["end"], device=device)
        self.entities_row_index = torch.tensor(entity["row_id"], device=device)
        self.entities_column_index = torch.tensor(entity["column_id"], device=device)
        self.entities_group_index = torch.tensor(entity["group_id"], device=device)
        self.entities_index_index = torch.tensor(entity["index_id"], device=device)
        # self.entities_labels_logits = entity["label_logits"]
        if "pred_label" in entity.keys():
            self.entities_labels = torch.tensor(entity["pred_label"], device=device)
        else:
            self.entities_labels = torch.tensor(entity["label"], device=device)
        
        
    def entity_cell_forward(self, hidden_states, b, head_entities):
        batch_size, max_n_words, context_dim = hidden_states.size()
        head_start_index = self.entities_start_index[head_entities]
        head_end_index = self.entities_end_index[head_entities]
        head_label = self.entities_labels[head_entities]
        
        head_row_id = self.entities_row_index[head_entities]
        head_column_id = self.entities_column_index[head_entities]
        
        head_group_id = self.entities_group_index[head_entities]
        head_index_id = self.entities_index_index[head_entities]
        # if entities_labels_logits != None:
        # head_logits = self.entities_labels_logits[head_entities]
        head_entity_repr = None
        for i in enumerate(head_start_index):
            index = i[0]
            start_index, end_index = head_start_index[index], head_end_index[index]
            temp_repr = hidden_states[b][start_index:end_index].mean(dim=0).view(1,context_dim)
            # temp_repr = hidden_states[b][start_index:end_index].max(dim=0)[0].view(1,context_dim)
            head_entity_repr = temp_repr if head_entity_repr == None else torch.cat([head_entity_repr,temp_repr], dim=0)
        return head_row_id, head_column_id, head_group_id, head_index_id, head_label, head_entity_repr# , head_logits

class REDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.angle_dim = config.hidden_size // 2
        self.group_dim = config.hidden_size // 4
        self.use_angle = False
        self.use_specialid = False
        self.del_begin = 0
        self.del_end = 50
        self.mlp_dim = config.hidden_size * 2   # + config.hidden_size
        self.use_del = True
        # self.entity_emb = nn.Embedding(5, config.hidden_size, scale_grad_by_freq=True)
        # self.group_emb = nn.Embedding(50, config.hidden_size // 4, scale_grad_by_freq=True)
        # self.index_emb= nn.Embedding(50, config.hidden_size // 4, scale_grad_by_freq=True)
        self.re_embedding = REEmbeddings(config)
        self.angle_embedding = nn.Embedding(8,self.angle_dim, scale_grad_by_freq=True)

        # if self.use_specialid:
        #     self.mlp_dim += self.group_dim * 2
        
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
        
        if self.use_angle:
            self.rel_classifier = MyBiaffineAttention(self.mlp_dim // 4, 2)
        else:
            self.rel_classifier = BiaffineAttention(self.mlp_dim // 4, 2)
        
        self.loss_fct = CrossEntropyLoss()
        self.angle_cnt = {}

        print(f"self.use_specialid:{self.use_specialid},self.group_emb:{self.group_dim}")
        print(f"self.del_begin:{self.del_begin},self.del_end:{self.del_end},self.use_del:{self.use_del}")


    def build_seq(self,relations,entities,b,key):
        # ReDocoder
        if self.use_del:
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b][key]))
                    for j in range(len(entities[b][key]))
                    # if i != j
                    if entities[b][key][i] == 1 and (entities[b][key][j] == 2 or entities[b][key][j] == 4) # entities[b]["label"][j] == 2 or entities[b]["label"][j] == 4
                    and (j - i) <= self.del_end
                    and (j - i) >= self.del_begin
                ]
            )
        else:
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b][key]))
                    for j in range(len(entities[b][key]))
                    # if i != j
                    if entities[b][key][i] == 1 and (entities[b][key][j] == 2 or entities[b][key][j] == 4) # entities[b]["label"][j] == 2 or entities[b]["label"][j] == 4
                ]
            )
            
        if len(all_possible_relations) == 0:
            all_possible_relations = gt_possible_relations = set([(0, 1)])
        positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
        if positive_relations.issubset(all_possible_relations):
            pass
        else:
            # pass
            print("wrong, loss relation")
        if self.training:
            all_possible_relations = all_possible_relations | positive_relations

        
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
        # ReDocoder
        batch_size = len(relations)
        new_relations = []
        gt_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0],\
                    "group_id":[0, 0],"index_id":[0, 0], \
                    "column_id":[0, 0],"row_id":[0, 0]}
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

    def entity_cell_forward(self, hidden_states, b, entities_start_index, \
                            entities_end_index, entities_labels,head_entities, \
                            entities_row_index, entities_column_index, \
                            entities_group_index,entities_index_index):
        batch_size, max_n_words, context_dim = hidden_states.size()
        head_start_index = entities_start_index[head_entities]
        head_end_index = entities_end_index[head_entities]
        head_label = entities_labels[head_entities]
        head_group_id = entities_group_index[head_entities]
        head_index_id = entities_index_index[head_entities]
        head_row_id = entities_row_index[head_entities]
        head_column_id = entities_column_index[head_entities]
        # if entities_labels_logits != None:
        #     head_logits = entities_labels_logits[head_entities]
        head_entity_repr = None
        for i in enumerate(head_start_index):
            index = i[0]
            start_index, end_index = head_start_index[index], head_end_index[index]
            temp_repr = hidden_states[b][start_index:end_index].mean(dim=0).view(1,context_dim)
            # temp_repr = hidden_states[b][start_index:end_index].max(dim=0)[0].view(1,context_dim)
            head_entity_repr = temp_repr if head_entity_repr == None else torch.cat([head_entity_repr,temp_repr], dim=0)
        return head_row_id, head_column_id, head_group_id, head_index_id, head_label, head_entity_repr

   
    def get_angle(self, x1, y1, x2, y2):
        """
        计算两个点之间的角度，以第一个点为基点，水平向右为0°基准。
        Args:
            x1, y1: 第一个点的坐标。
            x2, y2: 第二个点的坐标。

        Returns:
            以度为单位的角度值。
        """
        import math
        # 将输入转换为 Tensor
        # x1, y1, x2, y2 = torch.tensor(x1), torch.tensor(y1), torch.tensor(x2), torch.tensor(y2)

        # 计算向量AB的x分量和y分量
        dx, dy = x2 - x1, y2 - y1

        # 计算极角，需要注意x分量为0的情况
        mask1 = (dx > 0)
        mask2 = (dx < 0)
        mask3 = (dx == 0) & (dy > 0)
        mask4 = (dx == 0) & (dy < 0)

        angle = torch.zeros_like(dx,dtype=torch.float32)
        angle[mask1] = torch.atan(dy[mask1] / dx[mask1])
        angle[mask2] = torch.atan(dy[mask2] / dx[mask2]) + math.pi
        angle[mask3] = math.pi / 2
        angle[mask4] = -math.pi / 2

        # 将极角转换为角度
        degree = torch.rad2deg(angle)
        degree[degree < 0] += 360
        for i in range(0,360,45):
            mask = (degree>=i) & (degree<=i+45)
            degree[mask] = int(i/45)
            
        
        return degree.long()



    def forward(self, hidden_states,  entities, relations, bbox):
        # ReDocoder
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entitydata = EntityData(entities[b],device)
            if self.use_angle:
                entities_bbox = torch.tensor(bbox[b], device=device)
                head_bbox_x,head_bbox_y = entities_bbox[head_entities][:,1],entities_bbox[head_entities][:,2]
                tail_bbox_x,tail_bbox_y = entities_bbox[tail_entities][:,1],entities_bbox[tail_entities][:,2]
                entity_angle = self.get_angle(head_bbox_x,head_bbox_y,tail_bbox_x,tail_bbox_y)
        
            entities_labels_logits = None
            
            head_row_id, head_column_id, head_group_id, head_index_id, \
                head_label, head_entity_repr = entitydata.entity_cell_forward(hidden_states,b,head_entities)
                
            tail_row_id, tail_column_id, tail_group_id, tail_index_id, \
                tail_label, tail_entity_repr = entitydata.entity_cell_forward(hidden_states,b,tail_entities)


            head_label_repr = self.re_embedding(head_label,head_row_id,head_column_id)
            tail_label_repr = self.re_embedding(tail_label,tail_row_id,tail_column_id)

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
            if not self.use_angle:
                logits = self.rel_classifier(heads, tails)
            else:
                logits = self.rel_classifier(heads,tails,self.angle_embedding(entity_angle))
            # logits = self.LeakyReLU(logits)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        if torch.isnan(loss).sum() != 0:
            print("wrong! loss")
            print(loss)
        # assert torch.isnan(loss).sum() == 0, print(loss)
        return loss, all_pred_relations
    
    


class BiaffineAttentionLayout(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(BiaffineAttentionLayout, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)
        self.linear_layout = torch.nn.Linear(256,out_features,bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2,x_embed,y_embed):
        
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1)) + self.linear_layout(torch.cat((x_embed, y_embed), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters() 



class CellDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log_var_re = torch.nn.Parameter(torch.zeros((1,), requires_grad=True))
        self.entity_emb = nn.Embedding(5, config.hidden_size, scale_grad_by_freq=True)
        self.group_dim = config.hidden_size // 2
        # self.group_emb = nn.Embedding(50, self.group_dim, scale_grad_by_freq=True)
        # self.index_emb= nn.Embedding(50, self.group_dim, scale_grad_by_freq=True)
        self.use_specialid = False
        self.use_group = False
        self.use_index = False
        self.adaptive_loss = False
        self.multi_task = True
        self.use_del = False
        self.use_angle = False
        self.head_nums = 8
        self.config = config
        # self.entity_emb_rand = nn.Embedding(5, config.hidden_size, scale_grad_by_freq=True)
        self.mlp_dim = config.hidden_size * 2
        # self.transformer = nn.TransformerEncoderLayer(self.mlp_dim, 1, self.mlp_dim, config.hidden_dropout_prob)
        # self.transformer = nn.Transformer(d_model=self.mlp_dim,num_encoder_layers=1,num_decoder_layers=1,dropout=config.hidden_dropout_prob)
        self.re_embedding = REEmbeddings(config,self.use_specialid)
        self.lstm_layer = nn.LSTM(self.mlp_dim, self.mlp_dim // 2, 1, batch_first=True, bidirectional=True)
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.mlp_dim,nhead=16,dropout=config.hidden_dropout_prob)
        # self.transformer_layer = nn.Transformer(d_model=self.mlp_dim,dropout=config.hidden_dropout_prob)
        # self.attention_layer = nn.MultiheadAttention(self.mlp_dim, self.head_nums, config.hidden_dropout_prob)
        # self.dense = nn.Linear(self.mlp_dim,self.mlp_dim)
        self.angle_embedding = nn.Embedding(8, 256, scale_grad_by_freq=True)
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size, scale_grad_by_freq=True)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size, scale_grad_by_freq=True)
        # self.tails_attention = nn.MultiheadAttention(self.mlp_dim, self.head_nums, config.hidden_dropout_prob)
        # if self.use_specialid:
        #     self.mlp_dim = config.hidden_size * 2 + 2 * self.group_dim
        # else:
        #     self.mlp_dim = config.hidden_size * 2   # + config.hidden_size
        
        self.del_begin = 0
        self.del_end = 50
        
        projection = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.mlp_dim // 2, self.mlp_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        if self.use_angle:
            self.rel_classifier = BiaffineAttentionLayout(self.mlp_dim // 4, 2)
        else:
            self.rel_classifier = BiaffineAttention(self.mlp_dim // 4, 2)
        self.loss_fct = CrossEntropyLoss()
        print(f"self.use_angle{self.use_angle}")
        print(f"self.use_specialid:{self.use_specialid}\tself.use_group:{self.use_group}\tself.use_index:{self.use_index}")
        print(f"{self.__class__}:adaptive_loss:{self.adaptive_loss}\tmulti_task:{self.multi_task}")   
        print(f"self.del_begin:{self.del_begin},self.del_end:{self.del_end},self.use_del:{self.use_del}")
        
        
    def criterion(self, y_pred, y_true, log_vars,device):
        precision = log_vars**2
        diff = self.loss_fct(y_pred,y_true)
        loss = torch.sum(diff/precision,torch.log(log_vars))
        return loss

    def build_relation(self, relations, pred_entities, entities):
        batch_size = len(entities)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = pred_entities[b] = {"end": [1, 1], "label": [0, 0], \
                                                  "group_id":[0, 0],"index_id":[0, 0], \
                                                  "start": [0, 0],"column_id":[0, 0],"row_id":[0, 0],}
            if self.use_del:
                all_possible_relations = set(
                    [
                        (i, j)
                        for i in range(len(entities[b]["label"]))
                        for j in range(len(entities[b]["label"]))
                        # if i != j
                        if entities[b]["label"][i] == 1 and (entities[b]["label"][j] == 2 or entities[b]["label"][j] == 4) # or entities[b]["label"][j] == 4 
                        and (j - i) <= self.del_end
                        and (j - i) >= self.del_begin
                    ]
                )
            else:
                all_possible_relations = set(
                    [
                        (i, j)
                        for i in range(len(entities[b]["label"]))
                        for j in range(len(entities[b]["label"]))
                        if entities[b]["label"][i] == 1 and (entities[b]["label"][j] == 2 or entities[b]["label"][j] == 4) # or entities[b]["label"][j] == 4 
                    ]
                )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            if not positive_relations.issubset(all_possible_relations):
                print("loss relations")
                pass
            # if self.training:
            #     all_possible_relations = all_possible_relations | positive_relations
            #     print("merge")
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


    def get_angle(self, x1, y1, x2, y2):
        """
        计算两个点之间的角度，以第一个点为基点，水平向右为0°基准。
        Args:
            x1, y1: 第一个点的坐标。
            x2, y2: 第二个点的坐标。

        Returns:
            以度为单位的角度值。
        """
        import math
        # 将输入转换为 Tensor
        # x1, y1, x2, y2 = torch.tensor(x1), torch.tensor(y1), torch.tensor(x2), torch.tensor(y2)

        # 计算向量AB的x分量和y分量
        dx, dy = x2 - x1, y2 - y1

        # 计算极角，需要注意x分量为0的情况
        mask1 = (dx > 0)
        mask2 = (dx < 0)
        mask3 = (dx == 0) & (dy > 0)
        mask4 = (dx == 0) & (dy < 0)

        angle = torch.zeros_like(dx,dtype=torch.float32)
        angle[mask1] = torch.atan(dy[mask1] / dx[mask1])
        angle[mask2] = torch.atan(dy[mask2] / dx[mask2]) + math.pi
        angle[mask3] = math.pi / 2
        angle[mask4] = -math.pi / 2

        # 将极角转换为角度
        degree = torch.rad2deg(angle)
        degree[degree < 0] += 360
        for i in range(0,360,45):
            mask = (degree>=i) & (degree<=i+45)
            degree[mask] = int(i/45)
            
        
        return degree.long()
    
    
    def forward(self, hidden_states,bbox, pred_entities, entities, relations, epoch, all_logits):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        if self.training:
            relations, entities = self.build_relation(relations, pred_entities, entities)
        else:
            # relations, entities = self.build_relation(relations, pred_entities, entities)
            relations, entities = self.build_relation(relations, pred_entities, pred_entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entitydata = EntityData(pred_entities[b],device)    
            entities_labels_logits = None
            head_logits, tail_logits = None, None

            if self.use_angle:
                head_bbox_x,head_bbox_y = bbox[b][head_entities][:,0],bbox[b][head_entities][:,1]
                tail_bbox_x,tail_bbox_y = bbox[b][tail_entities][:,0],bbox[b][tail_entities][:,1]
                # x_embed = self.x_position_embeddings(self.config.max_2d_position_embeddings + (head_bbox_x - tail_bbox_x))
                # y_embed = self.y_position_embeddings(self.config.max_2d_position_embeddings + (head_bbox_y - tail_bbox_y))
                x_embed = self.x_position_embeddings(abs(head_bbox_x - tail_bbox_x))
                y_embed = self.y_position_embeddings(abs(head_bbox_y - tail_bbox_y))
                # entity_angle = self.get_angle(head_bbox_x,head_bbox_y,tail_bbox_x,tail_bbox_y)
                # angle_embed = self.angle_embedding(entity_angle)

                
            if 'label_logits' in pred_entities[b].keys():
                entities_labels_logits = all_logits[b]
            head_row_id, head_column_id, head_group_id, head_index_id, \
                head_label, head_entity_repr = entitydata.entity_cell_forward(hidden_states,b,head_entities)
                
            tail_row_id, tail_column_id, tail_group_id, tail_index_id, \
                tail_label, tail_entity_repr = entitydata.entity_cell_forward(hidden_states,b,tail_entities)
                                   
            if entities_labels_logits != None:
                head_logits, tail_logits = entities_labels_logits[head_entities],entities_labels_logits[tail_entities]
            #     head_label_repr = torch.einsum('sn,nf->snf', head_logits, self.entity_emb.weight).mean(dim=1)
            #     tail_label_repr = torch.einsum('sn,nf->snf', tail_logits, self.entity_emb.weight).mean(dim=1)
            # else:
            now_head_row_id, now_head_column_id = head_row_id,head_column_id
            now_tail_row_id, now_tail_column_id = tail_row_id,tail_column_id
            if self.use_group:
                now_head_row_id = head_group_id
                now_tail_row_id = tail_group_id
            if self.use_index:
                now_head_column_id = head_index_id
                now_tail_column_id = tail_index_id
                
            head_label_repr = self.re_embedding(head_label,head_logits,now_head_row_id,now_head_column_id,epoch)
            tail_label_repr = self.re_embedding(tail_label,tail_logits,now_tail_row_id,now_tail_column_id,epoch)
            # head_repr = head_entity_repr
            # tail_repr = tail_entity_repr
            head_repr = torch.cat(
                (head_entity_repr, head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (tail_entity_repr, tail_label_repr),
                dim=-1,
            )
            head_repr = head_repr.unsqueeze(0)
            tail_repr = tail_repr.unsqueeze(0)
            # head_repr = self.transformer_layer(head_repr)
            # head_repr = self.attention_layer(head_repr,tail_repr,head_repr)[0]
            head_repr,(h_n, c_n) = self.lstm_layer(head_repr)
            
            
            

            # # tail_repr = self.transformer_layer(tail_repr)
            # # tail_repr = self.attention_layer(tail_repr,head_repr,tail_repr)[0]
            tail_repr,(h_n, c_n) = self.lstm_layer(tail_repr)
            tail_repr = self.dropout(tail_repr)
            head_repr = self.dropout(head_repr)

            head_repr = head_repr.squeeze(0)
            tail_repr = tail_repr.squeeze(0)
            # head_repr = self.dense(head_repr)
            # tail_repr = self.dense(tail_repr)
            # attn_heads, head_weighted = self.heads_attention(head_repr,head_repr,head_repr)
            # attn_tails, tail_weighted = self.tails_attention(tail_repr,tail_repr,tail_repr)
            # heads = self.ffnn_head(attn_heads.squeeze(0))
            # tails = self.ffnn_tail(attn_tails.squeeze(0))
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            # logits = self.rel_classifier(heads, tails)
            if self.use_angle:
                logits = self.rel_classifier(heads, tails, x_embed,y_embed)
            else:
                logits = self.rel_classifier(heads,tails)
            if self.adaptive_loss:
                loss += self.criterion(logits,relation_labels,self.log_var_re,device)
            else:
                loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations
    




class EntityDataOnnx():
    def __init__(self, entity, device):
        self.entities_start_index = entity["start"]
        self.entities_end_index = entity["end"]
        self.entities_labels = torch.tensor(entity["label"],device=device)
        
        
    def entity_cell_forward(self, hidden_states, b, head_entities):
        batch_size, max_n_words, context_dim = hidden_states.size()
        head_start_index = self.entities_start_index[head_entities]
        head_end_index = self.entities_end_index[head_entities]
        head_label = self.entities_labels[head_entities]

        # if entities_labels_logits != None:
        #     head_logits = entities_labels_logits[head_entities]
        head_entity_repr = None
        index_list = torch.ones_like(head_start_index)
        index = 0
        for i in torch.unbind(head_start_index):
            # index = i[0]
            # start_index, end_index = head_start_index[index], head_end_index[index]
            # temp_repr = hidden_states[b][start_index:end_index].mean(dim=0).view(1,context_dim)
            index_array = torch.arange(head_start_index[index],head_end_index[index])
            temp_repr = torch.index_select(hidden_states[b],0,index_array).mean(dim=0).view(1,context_dim)
            # temp_repr = hidden_states[b][start_index:end_index].max(dim=0)[0].view(1,context_dim)
            head_entity_repr = temp_repr if head_entity_repr == None else torch.cat([head_entity_repr,temp_repr], dim=0)
            index+=1
        return head_label, head_entity_repr

class MyModel(nn.Module):
    def __init__(self, mlp_dim, hidden_dropout_prob):
        super(MyModel, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer1 = nn.Linear(self.mlp_dim, self.mlp_dim // 2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.hidden_dropout_prob)
        self.layer2 = nn.Linear(self.mlp_dim // 2, self.mlp_dim // 4)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

class CellDecoderOnnx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(5, config.hidden_size)
        self.mlp_dim = config.hidden_size * 2
        self.del_begin = 0
        self.del_end = 50
        # self.ffnn_head = MyModel(self.mlp_dim,config.hidden_dropout_prob)
        # self.ffnn_tail = MyModel(self.mlp_dim,config.hidden_dropout_prob)
        self.lstm_layer = nn.LSTM(self.mlp_dim, self.mlp_dim // 2, 1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ffnn_head  = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.mlp_dim // 2, self.mlp_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.ffnn_tail = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.mlp_dim // 2, self.mlp_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        # self.ffnn_head = copy.deepcopy(projection)
        # self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = MyBiaffineAttention(self.mlp_dim // 4, 2)
        self.loss_fct = CrossEntropyLoss()

        
    # def get_predicted_relations(self, logits, relations, entities):
    #     pred_relations = []
    #     for i, pred_label in enumerate(logits.argmax(-1)):
    #         if pred_label != 1:
    #             continue
    #         rel = {}
    #         rel["head_id"] = relations["head"][i]
    #         rel["head"] = entities["start"][rel["head_id"]]
    #         rel["head_type"] = entities["label"][rel["head_id"]]

    #         rel["tail_id"] = relations["tail"][i]
    #         rel["tail"] = entities["start"][rel["tail_id"]]
    #         rel["tail_type"] = entities["label"][rel["tail_id"]]
    #         rel["type"] = 1
    #         pred_relations.append(rel)
    #     return pred_relations
    
   
    def forward(self, hidden_states,
                labels,
                head_id,tail_id,
                relations_head_mask,
        relations_head_len,
        relations_tail_mask,
        relations_tail_len):
        # batch_size, max_n_words, context_dim = hidden_states.size()
        # device = hidden_states.device
        # relations, entities = self.build_relation(pred_entities, pred_entities)

        # labels = torch.argmax(logits,dim=1)
        head_label, tail_label = labels[head_id], labels[tail_id]
        head_label_repr = self.entity_emb(head_label)
        tail_label_repr = self.entity_emb(tail_label)
        head_entity_repr = torch.matmul(relations_head_mask, hidden_states[0]) / relations_head_len
        tail_entity_repr = torch.matmul(relations_tail_mask, hidden_states[0]) / relations_tail_len

        head_repr = torch.cat(
            (head_entity_repr, head_label_repr),
            dim=1,
        )
        tail_repr = torch.cat(
            (tail_entity_repr, tail_label_repr),
            dim=1,
        )
        head_repr = head_repr.unsqueeze(0)
        tail_repr = tail_repr.unsqueeze(0)

        head_repr,(h_n, c_n) = self.lstm_layer(head_repr)
        tail_repr,(h_n, c_n) = self.lstm_layer(tail_repr)
        
        tail_repr = self.dropout(tail_repr)
        head_repr = self.dropout(head_repr)  
        head_repr = head_repr.squeeze(0)
        tail_repr = tail_repr.squeeze(0)
        
        heads = self.ffnn_head(head_repr)
        tails = self.ffnn_tail(tail_repr)



        logits = self.rel_classifier(heads, tails)
            # all_logits.append(logits)
            # pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            # all_pred_relations.append(pred_relations)
        return logits # , pred_relations