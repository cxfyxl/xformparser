import logging
import os
import sys
import json
import numpy as np
from datasets import ClassLabel, load_dataset
# import wandb
import layoutlmft.data.datasets.xfun
import transformers
from layoutlmft import AutoModelForRelationExtraction
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.data.data_collator import DataCollatorForKeyValueExtraction
from layoutlmft.evaluation import re_score
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForJointCellClassificationOnnx,LayoutLMv2ForJointCellClassificationREOnnx
# from layoutlmft.modules.decoders.re import CellDecoderOnnx
from layoutlmft.trainers import XfunJointTrainer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from transformers import AdamW
import numpy as np
import onnxruntime
import torch
import torch.onnx

# from callback.progressbar import ProgressBar
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
# from models.transformers import BertConfig
# from models.transformers.configuration_electra import ElectraConfig
# from models.bert_for_ner import BertForNER, ElectraForNER
# from processors.utils_ner import CNerTokenizer, get_entities
# from processors.ner_seq import ner_processors as processors
# from evaluate import get_f1_score
# set the wandb project where this run will be logged


# save your trained model checkpoint to wandb
# os.environ["WANDB_LOG_MODEL"]="true"

# # turn off watch to log faster
# os.environ["WANDB_WATCH"]="false"


logger = logging.getLogger(__name__)

class ONNXModel:
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self,input_ids=None,
        bbox=None,
        image=None,
        entities_mask=None,
        entities_len=None,
        head_id=None,
        tail_id=None,
        relations_head_mask=None,
        relations_head_len=None,
        relations_tail_mask=None,
        relations_tail_len=None, 
        ):

        input_feed = {
            'input_ids': input_ids,
            'bbox': bbox,
            'image':image,
            'entities_mask':entities_mask,
            'entities_len':entities_len,
            'head_id':head_id,
            'tail_id':tail_id,
            'relations_head_mask':relations_head_mask,
            'relations_head_len':relations_head_len,
            'relations_tail_mask':relations_tail_mask,
            'relations_tail_len':relations_tail_len, 
            # 'entities_label':entities_label,
        }
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output

    
def padding(x, pad):
    x += [pad] * (128 - len(x))
    return x
def get_mask_len(start,end):
    mask = torch.zeros(512)
    mask[start:end] = 1
    mask = mask.view(1,-1)
    temp = torch.full((1,768),end-start)
    return mask, temp
# def translate(exp_dir, features, n_batch, device,config, model):
def translate(nermodel,remodel, test_dataloader, model_args, label_list, n_batch):

    ner_onnxfile = os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx")
    re_onnxfile = os.path.join(model_args.model_name_or_path,"generic_re_1.0.0.onnx")
    for step, inputs in enumerate(test_dataloader):
        # pass
        del inputs['id']
        del inputs['len']
        # del inputs['image']
        input_ids = inputs['input_ids']
        image = inputs['image']
        bbox = inputs['bbox']
        labels = inputs['labels']


        entities = inputs['entities']
        # relations = inputs['relations']
        entities_mask = None
        entities_len = None
        
        for start, end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
            mask = torch.zeros(512)
            mask[start:end] = 1
            mask = mask.view(1,-1)
            entities_mask = mask if entities_mask==None else torch.cat((entities_mask,mask),dim=0)
            temp = torch.full((1,768),end-start)
            entities_len = temp if entities_len==None else torch.cat((entities_len,temp),dim=0)
            pass
        i,j=0,0
        head_id = None
        tail_id = None
        relations_head_mask,relations_tail_mask = None, None
        relations_head_len,relations_tail_len = None, None
        for head_start, head_end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
            head_mask, head_len = get_mask_len(head_start,head_end)
            j=0            
            for tail_start, tail_end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
                tail_mask, tail_len = get_mask_len(tail_start,tail_end)
                relations_head_mask = head_mask if relations_head_mask==None else torch.cat((relations_head_mask,head_mask),dim=0)
                relations_head_len =  head_len if relations_head_len==None else torch.cat((relations_head_len,head_len),dim=0)
                
                relations_tail_mask = tail_mask if relations_tail_mask==None else torch.cat((relations_tail_mask,tail_mask),dim=0)
                relations_tail_len =  tail_len if relations_tail_len==None else torch.cat((relations_tail_len,tail_len),dim=0)
                
                head_id = torch.tensor(i).view(1) if head_id == None else torch.cat((head_id, torch.tensor(i).view(1)),dim=0)
                tail_id = torch.tensor(j).view(1) if tail_id == None else torch.cat((tail_id, torch.tensor(j).view(1)),dim=0)
                j+=1
            i+=1
        # for entity in inputs['entities']:
        #     for k,v in entity.items():
        #         entity[k] = [int(i) for i in entity[k]]
        #         entity[k] = torch.tensor(entity[k]).to(device)
        ner_sample_input = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'bbox': inputs['bbox'].squeeze(0),
            'image':image.tensor.squeeze(0),
            'entities_mask':entities_mask,
            'entities_len':entities_len,
            'head_id':head_id,
            'tail_id':tail_id,
            'relations_head_mask':relations_head_mask,
            'relations_head_len':relations_head_len,
            'relations_tail_mask':relations_tail_mask,
            'relations_tail_len':relations_tail_len,  
        }

        with torch.no_grad():
            nermodel.eval()
            torch.onnx.export(
                model=nermodel,
                args=tuple(ner_sample_input.values()),
                f=ner_onnxfile,
                input_names=list(ner_sample_input.keys()),#,'re_logits','pred_relations'
                output_names=['logits','re_logits'], # logits, re_logits, pred_relations,'re_logits','re_logits'
                opset_version=12,
                export_params = True,
                do_constant_folding=False,
                # verbose=False,
                # opset_version=13,
                # do_constant_folding=True,
                dynamic_axes={
                    'entities_mask':{0: 'entity_len'},
                    'entities_len':{0: 'entity_len'},
                    'head_id':{0: 'pred_re_len'},
                    'tail_id':{0: 'pred_re_len'},
                    'relations_head_mask':{0: 'pred_re_len'},
                    'relations_head_len':{0: 'pred_re_len'},
                    'relations_tail_mask':{0: 'pred_re_len'},
                    'relations_tail_len':{0: 'pred_re_len'}, 
                    # 'logits': {0: 'entity_len'},
                    're_logits': {0: 'pred_re_len'},
                    'logits': {0: 'entity_len'},
                },
            )

            # print("NERComplete")
            # torch.onnx.export(
            #     model=remodel,
            #     args=tuple(re_sample_input.values()),
            #     f=re_onnxfile,
            #     input_names=list(re_sample_input.keys()),#,'re_logits','pred_relations'
            #     output_names=['re_logits'], # logits, re_logits, pred_relations,'re_logits','re_logits'
            #     opset_version=12,
            #     export_params = True,
            #     do_constant_folding=False,
            #     # verbose=False,
            #     # opset_version=13,
            #     # do_constant_folding=True,
            #     dynamic_axes={
            #         'head_id':{0: 'pred_re_len'},
            #         'tail_id':{0: 'pred_re_len'},
            #         'labels':{0: 'entity_len'},
            #         'relations_head_mask':{0: 'pred_re_len'},
            #         'relations_head_len':{0: 'pred_re_len'},
            #         'relations_tail_mask':{0: 'pred_re_len'},
            #         'relations_tail_len':{0: 'pred_re_len'}, 
            #         # 'logits': {0: 'entity_len'},
            #         're_logits': {0: 'pred_re_len'},
            #     },
            # )

            # torch.onnx.export(model, (input_ids,bbox,labels,entities), onnxfile,
            #                     verbose=False, input_names=input_names, output_names=output_names)
            # print(model(input_ids, bbox,labels,entities,relations).size())
            print("REComplete")
        break

def build_relation(labels):
    # new_relations = []

        # all_possible_relations = set([(0,1)])
    all_possible_relations = set(
        [
            (i, j)
            for i in range(len(labels))
            for j in range(len(labels))
            # if i != j
            if labels[i] == 1 and (labels[j] == 2 or labels[j] == 4) # or entities[b]["label"][j] == 4 
            and (j - i) <= 50
            and (j - i) >= 0
        ]
    )
    reordered_relations = sorted(list(all_possible_relations))
    
    relation_per_doc = {"head": [], "tail": [], "label": []}
    relation_per_doc["head"] = [i[0] for i in reordered_relations]
    relation_per_doc["tail"] = [i[1] for i in reordered_relations]
    relation_per_doc["label"] = [1] * len(reordered_relations)
    # assert len(relation_per_doc["head"]) != 0
    # new_relations.append(relation_per_doc)
    return relation_per_doc, [i[0]*len(labels)+i[1] for i in reordered_relations]


def get_predicted_relations(logits, entities,relations, labels):
    pred_relations = []
    for i, pred_label in enumerate(logits.argmax(-1)):
        if pred_label != 1:
            continue
        rel = {}
        rel["head_id"] = relations["head"][i]
        rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
        rel["head_type"] = labels[rel["head_id"]]

        rel["tail_id"] = relations["tail"][i]
        rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
        rel["tail_type"] = labels[rel["tail_id"]]
        rel["type"] = 1
        pred_relations.append(rel)
    return pred_relations

def accuracy(list1, list2):
    """
    计算分类准确率
    """
    total = len(list1)
    correct = sum(1 for i in range(total) if list1[i] == list2[i])
    acc = correct / total
    return acc
def test1(ner_onnxmodel,ner_model,test_dataloader):
    base_labels = []
    onnx_labels = []
    re_labels = None
    pred_relations = None
    pred_entities = None
    entities = None
    gt_relations = []
    pred_result = []
    for step, inputs in enumerate(test_dataloader):
        # pass
        if step != 10:
            continue
        del inputs['id']
        del inputs['len']
        del inputs['labels']
        # del inputs['image']
        input_ids = inputs['input_ids']
        image = inputs['image']
        bbox = inputs['bbox']
        entities = inputs['entities']
        # labels = inputs['labels']
        entities_mask = None
        entities_len = None
        relations_head_mask,relations_tail_mask = None, None
        relations_head_len,relations_tail_len = None, None
        i,j=0,0
        head_id = None
        tail_id = None
        for start, end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
            mask = torch.zeros(512)
            mask[start:end] = 1
            mask = mask.view(1,-1)
            entities_mask = mask if entities_mask==None else torch.cat((entities_mask,mask),dim=0)
            temp = torch.full((1,768),end-start)
            entities_len = temp if entities_len==None else torch.cat((entities_len,temp),dim=0)
            # pass
        for head_start, head_end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
            head_mask, head_len = get_mask_len(head_start,head_end)
            j=0
            for tail_start, tail_end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
                tail_mask, tail_len = get_mask_len(tail_start,tail_end)
                relations_head_mask = head_mask if relations_head_mask==None else torch.cat((relations_head_mask,head_mask),dim=0)
                relations_head_len =  head_len if relations_head_len==None else torch.cat((relations_head_len,head_len),dim=0)
                
                relations_tail_mask = tail_mask if relations_tail_mask==None else torch.cat((relations_tail_mask,tail_mask),dim=0)
                relations_tail_len =  tail_len if relations_tail_len==None else torch.cat((relations_tail_len,tail_len),dim=0)
                head_id = torch.tensor(i).view(1) if head_id == None else torch.cat((head_id, torch.tensor(i).view(1)),dim=0)
                tail_id = torch.tensor(j).view(1) if tail_id == None else torch.cat((tail_id, torch.tensor(j).view(1)),dim=0)
                j+=1
            i+=1

        sample_input = {
            'input_ids': inputs['input_ids'],
            'bbox': inputs['bbox'],
            'image':image.tensor,
            'entities_mask':entities_mask,
            'entities_len':entities_len,
            'head_id':head_id,
            'tail_id':tail_id,
            'relations_head_mask':relations_head_mask,
            'relations_head_len':relations_head_len,
            'relations_tail_mask':relations_tail_mask,
            'relations_tail_len':relations_tail_len,            
        }
        # tp_labels = [i for i in inputs['labels'].view(-1).tolist() if i!=-100]
        base_labels.extend(inputs['entities'][0]['label'])
        onnx_sample_input = {
            'input_ids': inputs['input_ids'].squeeze(0).numpy(),
            'bbox': inputs['bbox'].squeeze(0).numpy(),
            'image':image.tensor.squeeze(0).numpy(),

            'entities_mask':entities_mask.numpy(),
            'entities_len':entities_len.numpy(),
            'head_id':head_id.numpy(),
            'tail_id':tail_id.numpy(),
            'relations_head_mask':relations_head_mask.numpy(),
            'relations_head_len':relations_head_len.numpy(),
            'relations_tail_mask':relations_tail_mask.numpy(),
            'relations_tail_len':relations_tail_len.numpy(),            
        }
        # ner_model.eval()
            # import numpy as np
        
        with torch.no_grad():
            onnx_output = ner_onnxmodel.forward(**onnx_sample_input)
            onnx_pre_labels = list(np.argmax(onnx_output[0],axis=1))
            onnx_labels.extend(onnx_pre_labels)
            
            onnx_relations,onnx_pred_index = build_relation(onnx_pre_labels)
            onnx_re_logits = onnx_output[1][onnx_pred_index]
            onnx_result = get_predicted_relations(onnx_re_logits,entities[0],onnx_relations,onnx_pre_labels)
            pred_result.append(onnx_result)
            # get_predicted_relations
            # onnx_relations = build_relation(onnx_pre_labels)
            # onnx_result = get_predicted_relations(onnx_output[1],onnx_relations,onnx_pre_labels)

            base_output = ner_model.forward(**inputs)
            base_pre_labels = list(base_output[0].argmax(dim=-1).numpy())
            # # base_relations = build_relation(base_pre_labels)
            # # base_result = get_predicted_relations(base_output[1].numpy(),base_relations,base_pre_labels)
            print(base_pre_labels==onnx_pre_labels)
        rel_sent = []
        

        for head, tail in zip(inputs['relations'][0]["head"], inputs['relations'][0]["tail"]):
            rel = {}
            rel["head_id"] = head
            rel["head"] = (entities[0]["start"][rel["head_id"]], entities[0]["end"][rel["head_id"]])
            rel["head_type"] = entities[0]["label"][rel["head_id"]]

            rel["tail_id"] = tail
            rel["tail"] = (entities[0]["start"][rel["tail_id"]], entities[0]["end"][rel["tail_id"]])
            rel["tail_type"] = entities[0]["label"][rel["tail_id"]]

            rel["type"] = 1

            rel_sent.append(rel)

        gt_relations.append(rel_sent)
        
        assert len(onnx_pre_labels) == len(inputs['entities'][0]['label'])
        # break
    score = re_score(pred_result, gt_relations, mode="boundaries")
    print(accuracy(base_labels,onnx_labels))
    

def test(ner_model,ner_onnxmodel,re_model,re_onnxmodel, test_dataloader, model_args, label_list, n_batch):
    for step, inputs in enumerate(test_dataloader):
        if step != 10:
            continue
        # pass
        del inputs['id']
        del inputs['len']
        del inputs['labels']
        # del inputs['image']
        input_ids = inputs['input_ids']
        image = inputs['image']
        bbox = inputs['bbox']
        # labels = inputs['labels']
        entities_mask = None
        entities_len = None
        relations_head_mask,relations_tail_mask = None, None
        relations_head_len,relations_tail_len = None, None
        i,j=0,0
        head_id = None
        tail_id = None
        for start, end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
            mask = torch.zeros(512)
            mask[start:end] = 1
            mask = mask.view(1,-1)
            entities_mask = mask if entities_mask==None else torch.cat((entities_mask,mask),dim=0)
            temp = torch.full((1,768),end-start)
            entities_len = temp if entities_len==None else torch.cat((entities_len,temp),dim=0)
            # pass
        for head_start, head_end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
            head_mask, head_len = get_mask_len(head_start,head_end)
            j=0
            for tail_start, tail_end in zip(inputs['entities'][0]['start'],inputs['entities'][0]['end']):
                tail_mask, tail_len = get_mask_len(tail_start,tail_end)
                relations_head_mask = head_mask if relations_head_mask==None else torch.cat((relations_head_mask,head_mask),dim=0)
                relations_head_len =  head_len if relations_head_len==None else torch.cat((relations_head_len,head_len),dim=0)
                
                relations_tail_mask = tail_mask if relations_tail_mask==None else torch.cat((relations_tail_mask,tail_mask),dim=0)
                relations_tail_len =  tail_len if relations_tail_len==None else torch.cat((relations_tail_len,tail_len),dim=0)
                head_id = torch.tensor(i).view(1) if head_id == None else torch.cat((head_id, torch.tensor(i).view(1)),dim=0)
                tail_id = torch.tensor(j).view(1) if tail_id == None else torch.cat((tail_id, torch.tensor(j).view(1)),dim=0)
                j+=1
            i+=1
        # for entity in inputs['entities']:
        #     for k,v in entity.items():
        #         entity[k] = [int(i) for i in entity[k]]
        #         entity[k] = torch.tensor(entity[k]).to(device)

        entities = inputs['entities']
        # relations = inputs['relations']

        ner_sample_input = {
            'input_ids': inputs['input_ids'],
            'bbox': inputs['bbox'],
            'image':image.tensor,
            'entities_mask':entities_mask,
            'entities_len':entities_len,
        }

        re_sample_input = {
            'input_ids': inputs['input_ids'],
            'bbox': inputs['bbox'],
            #'attention_mask':torch.zeros((1,512), dtype=torch.int64) + 1,
            'image':image.tensor,
            # 'entities_start':inputs['entities'][0]['start'],
            # 'entities_end': inputs['entities'][0]['end'],
            'labels':torch.flatten(inputs['labels'])[:len(inputs['entities'][0]['start'])],
            # 'entities_mask':entities_mask,
            # 'entities_len':entities_len,
            'head_id':head_id,
            'tail_id':tail_id,
            'relations_head_mask':relations_head_mask,
            'relations_head_len':relations_head_len,
            'relations_tail_mask':relations_tail_mask,
            'relations_tail_len':relations_tail_len,            
            # 'entities_label':inputs['entities'][0]['label'].view(1,-1),
        }
        sample_input = {
            'input_ids': inputs['input_ids'],
            'bbox': inputs['bbox'],
            #'attention_mask':torch.zeros((1,512), dtype=torch.int64) + 1,
            'image':image.tensor,
            'entities_mask':entities_mask,
            'entities_len':entities_len,
            'head_id':head_id,
            'tail_id':tail_id,
            'relations_head_mask':relations_head_mask,
            'relations_head_len':relations_head_len,
            'relations_tail_mask':relations_tail_mask,
            'relations_tail_len':relations_tail_len,  
            # 'entities_start':inputs['entities'][0]['start'],
            # 'entities_end': inputs['entities'][0]['end'],
            # 'entities_label':inputs['entities'][0]['label'].view(1,-1),
        }
        onnx_sample_input = {
            'input_ids': inputs['input_ids'].numpy(),
            'bbox': inputs['bbox'].numpy(),
            #'attention_mask':torch.zeros((1,512), dtype=torch.int64) + 1,
            'image':image.tensor.numpy(),
            'entities_mask':entities_mask.numpy(),
            'entities_len':entities_len.numpy(),
            'head_id':head_id.numpy(),
            'tail_id':tail_id.numpy(),
            'relations_head_mask':relations_head_mask.numpy(),
            'relations_head_len':relations_head_len.numpy(),
            'relations_tail_mask':relations_tail_mask.numpy(),
            'relations_tail_len':relations_tail_len.numpy(),  
            # 'entities_start':inputs['entities'][0]['start'].numpy(),
            # 'entities_end': inputs['entities'][0]['end'].numpy(),
            # 'entities_label':inputs['entities'][0]['label'].view(1,-1),
        }
        model.eval()
        # import numpy as np
        with torch.no_grad():
            onnx_output = onnxmodel.forward(**onnx_sample_input)
            onnx_pre_labels = list(np.argmax(onnx_output[0],axis=1))
            # onnx_relations = build_relation(onnx_pre_labels)
            # onnx_result = get_predicted_relations(onnx_output[1],onnx_relations,onnx_pre_labels)

            base_output = model.forward(**sample_input)
            base_pre_labels = list(base_output[0].argmax(dim=-1).numpy())
            # base_relations = build_relation(base_pre_labels)
            # base_result = get_predicted_relations(base_output[1].numpy(),base_relations,base_pre_labels)
            print(base_pre_labels==onnx_pre_labels)
        
        # break


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("-"*10)
    print(data_args)
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
    # datasets = load_dataset(
    #     os.path.abspath(layoutlmft.data.datasets.xfun.__file__),
    #     f"xfun.{data_args.lang}",
    #     additional_langs=data_args.additional_langs,
    #     keep_in_memory=True,
    # )
    
    datasets = load_dataset(
        '/home/zhanghang-s21/data/layoutlmft/layoutlmft/data/datasets/myxfunsplit_new.py',
        "myxfunsplit_new.zh",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
    )
    # shuffled_ds = datasets.shuffle(seed=training_args.seed)
    # test_name = "test"
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features
        # column_names = datasets[test_name].column_names
        # features = datasets[test_name].features
    
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
    # LayoutLMv2ForJointCellClassificationOnnx
    ner_model = LayoutLMv2ForJointCellClassificationOnnx.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # re_model = LayoutLMv2ForJointCellClassificationREOnnx.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    
    # optimizer = AdamW([{"params":model.classifier,"lr":1e-5}],lr=5e-5)
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

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]# .sort("len",reverse=False)
        # eval_dataset = eval_dataset
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")       
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            print("begin predict")
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

        # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,collate_fn=data_collator)
 
    
    # translate(ner_model,None, test_dataloader, model_args, label_list, 1)
    
    # import onnx.helper as helper
    # onnxmodel = onnx.load(os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx"))
    
    # onnx_model = onnx.load(os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx"))
    # onnx.checker.check_model(os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx"))
    ner_onnxmodel = ONNXModel(os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx"))
    # re_onnxmodel = ONNXREModel(os.path.join(model_args.model_name_or_path,"generic_re_1.0.0.onnx"))
    # test(ner_model,ner_onnxmodel,re_model,re_onnxmodel, test_dataloader, model_args, label_list, 1)
    test1(ner_onnxmodel,ner_model, test_dataloader)





def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()