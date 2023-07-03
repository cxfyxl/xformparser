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
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForJointCellClassification,LayoutLMv2ForJointCellClassificationOnnx
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
        relations_tail_len=None):

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
def translate(model, test_dataloader, model_args, label_list, n_batch):

    n_batch = 1
    device = torch.device("cpu")
    configfile = os.path.join(model_args.model_name_or_path,"config.json")
    binfile = os.path.join(model_args.model_name_or_path,"pytorch_model.bin")
    onnxfile = os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx")
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

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
        sample_input = {
            'input_ids': inputs['input_ids'],
            'bbox': inputs['bbox'],
            #'attention_mask':torch.zeros((1,512), dtype=torch.int64) + 1,
            'image':image.tensor,
            # 'entities_start':inputs['entities'][0]['start'],
            # 'entities_end': inputs['entities'][0]['end'],
            'entities_mask':entities_mask,
            'entities_len':entities_len,
            'head_id':head_id,
            'tail_id':tail_id,
            'relations_head_mask':relations_head_mask,
            'relations_head_len':relations_head_len,
            'relations_tail_mask':relations_tail_mask,
            'relations_tail_len':relations_tail_len,            
            # 'entities_label':inputs['entities'][0]['label'].view(1,-1),
        }

        with torch.no_grad():
            model.eval()
            torch.onnx.export(
                model=model,
                args=tuple(sample_input.values()),
                f=onnxfile,
                input_names=list(sample_input.keys()),#,'re_logits','pred_relations'
                output_names=['logits','re_logits'], # logits, re_logits, pred_relations,'re_logits','re_logits'
                opset_version=12,
                export_params = True,
                # do_constant_folding=False,
                # verbose=False,
                # opset_version=13,
                # do_constant_folding=True,
                dynamic_axes={
                    # 'input_ids': {0: 'batch_size'},
                    # 'bbox': {0: 'batch_size'},
                    # 'entities_start':{0: 'entity_len'},
                    # 'entities_end': {0: 'entity_len'},
                    'entities_mask':{0: 'entity_len'},
                    'entities_len':{0: 'entity_len'},
                    # 'entities_label':{0: 'batch_size', 1: 'entity_len'},
                    'head_id':{0: 'pred_re_len'},
                    'tail_id':{0: 'pred_re_len'},
                    'relations_head_mask':{0: 'pred_re_len'},
                    'relations_head_len':{0: 'pred_re_len'},
                    'relations_tail_mask':{0: 'pred_re_len'},
                    'relations_tail_len':{0: 'pred_re_len'}, 
                    'logits': {0: 'entity_len'},
                    're_logits': {0: 'pred_re_len'},
                },
            )
            # torch.onnx.export(model, (input_ids,bbox,labels,entities), onnxfile,
            #                     verbose=False, input_names=input_names, output_names=output_names)
            # print(model(input_ids, bbox,labels,entities,relations).size())
            print("Complete")
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

    reordered_relations = all_possible_relations
    relation_per_doc = {"head": [], "tail": [], "label": []}
    relation_per_doc["head"] = [i[0] for i in reordered_relations]
    relation_per_doc["tail"] = [i[1] for i in reordered_relations]
    relation_per_doc["label"] = [1] * len(reordered_relations)
    # assert len(relation_per_doc["head"]) != 0
    # new_relations.append(relation_per_doc)
    return relation_per_doc

def get_predicted_relations(logits, relations, labels):
    pred_relations = []
    for i, pred_label in enumerate(logits.argmax(-1)):
        if pred_label != 1:
            continue
        rel = {}
        rel["head_id"] = relations["head"][i]
        # rel["head"] = entities["start"][rel["head_id"]]
        rel["head_type"] = labels[rel["head_id"]]

        rel["tail_id"] = relations["tail"][i]
        # rel["tail"] = entities["start"][rel["tail_id"]]
        rel["tail_type"] = labels[rel["tail_id"]]
        rel["type"] = 1
        pred_relations.append(rel)
    return pred_relations
def test(onnxmodel, model, test_dataloader, model_args, label_list, n_batch):

    n_batch = 1
    device = torch.device("cpu")
    configfile = os.path.join(model_args.model_name_or_path,"config.json")
    binfile = os.path.join(model_args.model_name_or_path,"pytorch_model.bin")
    onnxfile = os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx")
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    for step, inputs in enumerate(test_dataloader):
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
    model = LayoutLMv2ForJointCellClassificationOnnx.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
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

    train_mode = "base"
    print(f"train_mode:{train_mode}")
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if train_mode == "shuffle":
            train_dataset = datasets["train"].shuffle(seed=training_args.seed)# .sort("len",reverse=True)
            print(f"train_mode:{train_mode}")
        elif train_mode == "sorted":
            train_dataset = datasets["train"].sort("len")
            print(f"train_mode:{train_mode}")
        elif train_mode == "reverse":
            train_dataset = datasets["train"].sort("len",reverse=True)
            print(f"train_mode:{train_mode}")
        else:
            train_dataset = datasets["train"]
            print(f"train_mode:{train_mode}")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    
    # if training_args.do_train:
    #     if "train" not in datasets:
    #         raise ValueError("--do_train requires a train dataset")
    #     train_dataset = datasets["train"].sort("len")
    #     if data_args.max_train_samples is not None:
    #         train_dataset = train_dataset.select(range(data_args.max_train_samples))

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
 
    
    translate(model, test_dataloader, model_args, label_list, 1)
    import onnx
    # import onnx.helper as helper
    # onnxmodel = onnx.load(os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx"))
    
    # onnx_model = onnx.load(os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx"))
    # onnx.checker.check_model(os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx"))
    onnxmodel = ONNXModel(os.path.join(model_args.model_name_or_path,"generic_ner_1.0.0.onnx"))
    test(onnxmodel,model,test_dataloader, model_args, label_list, 1)
    def compute_metrics(p):
        pred_relations, gt_relations = p
        score = re_score(pred_relations, gt_relations, mode="boundaries")
        return score

    # Initialize our Trainer
    trainer = XfunJointTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # test_dataset=test_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        # optimizer=optimizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.test_dataset = test_dataset
    # trainer.    
    # trainer.create_optimizer_and_scheduler(trainer.args.max_steps)
    
    # trainer.optimizer =  AdamW([{"params":model.classifier.parameters(),"lr":1e-5},
    #                 {"params":model.layoutlmv2.parameters()},
    #                 {"params":model.extractor.parameters()}],lr=5e-5)

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
    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")       
        predictions, labels, metrics = trainer.predict(test_dataset)
        # Save predictions
        print(metrics)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        # output_test_predictions_file = os.path.join(training_args.output_dir, test_name + "_data_test_predictions_re.json")
        # with open(output_test_predictions_file, 'w') as f:
        #     json.dump({'pred':predictions, 'label': labels}, f)
    # wandb.finish()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()