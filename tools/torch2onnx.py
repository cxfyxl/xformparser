import json

import numpy as np
import onnxruntime
import torch
import torch.onnx
import transformers
from layoutlmft import AutoModelForRelationExtraction
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.data.data_collator import DataCollatorForKeyValueExtraction
from layoutlmft.evaluation import re_score
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForJointCellClassification
from layoutlmft.trainers import XfunJointTrainer
# from callback.progressbar import ProgressBar
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
# from models.transformers import BertConfig
# from models.transformers.configuration_electra import ElectraConfig
# from models.bert_for_ner import BertForNER, ElectraForNER
# from processors.utils_ner import CNerTokenizer, get_entities
# from processors.ner_seq import ner_processors as processors
from evaluate import get_f1_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
MODEL_CLASSES = {
    "bert": (BertConfig, BertForNER, CNerTokenizer),
    "electra": (ElectraConfig, ElectraForNER, CNerTokenizer),
}


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

    def forward(self, input_ids, token_type_ids, attention_mask):
        input_feed = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output


def padding(x, pad):
    x += [pad] * (128 - len(x))
    return x


def translate(exp_dir, features, n_batch, device, model_type="bert"):
    configfile = exp_dir + "config.json"
    binfile = exp_dir + "pytorch_model.bin"
    onnxfile = exp_dir + "generic_ner_1.0.0.onnx"

    processor = processors["cluener"]()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(configfile, num_labels=num_labels)
    model = model_class.from_pretrained(binfile, from_tf=False, config=config)

    all_input_ids = torch.tensor([padding(f.input_ids, 0) for f in features[:n_batch]], dtype=torch.long)
    all_token_type_ids = torch.tensor([padding(f.segment_ids, 0) for f in features[:n_batch]], dtype=torch.long)
    all_attention_mask = torch.tensor([padding(f.input_mask, 0) for f in features[:n_batch]], dtype=torch.long)
    test_dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=n_batch)

    for input_ids, token_type_ids, attention_mask in test_dataloader:
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            input_names = ["input_ids", "token_type_ids", "attention_mask"]
            output_names = ["logits"]
            if model_type == "electra":
                torch.onnx.export(model, (input_ids, token_type_ids, attention_mask), onnxfile,
                                  verbose=False, input_names=input_names, output_names=output_names,
                                  opset_version=10)
            else:
                torch.onnx.export(model, (input_ids, token_type_ids, attention_mask), onnxfile,
                                  verbose=False, input_names=input_names, output_names=output_names)
            print(model(input_ids, token_type_ids, attention_mask).size())
            print("Complete")


def test(exp_dir, model, features, n_batch, device):
    processor = processors["cluener"]()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    all_input_ids = torch.tensor([padding(f.input_ids, 0) for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([padding(f.segment_ids, 0) for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([padding(f.input_mask, 0) for f in features], dtype=torch.long)
    test_dataset = TensorDataset(all_input_ids, all_segment_ids, all_input_mask)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=n_batch)

    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, (input_ids, token_type_ids, attention_mask) in enumerate(test_dataloader):
        input_ids = input_ids.to(device).numpy()
        token_type_ids = token_type_ids.to(device).numpy()
        attention_mask = attention_mask.to(device).numpy()
        with torch.no_grad():
            output = model.forward(input_ids, token_type_ids, attention_mask)
        preds = []
        for i in range(len(attention_mask[0])):
            if attention_mask[0][i] != 0:
                preds.append(np.argmax(output[0][0][i]))
        preds = preds[1:-1]
        label_entities = get_entities(preds, id2label, "bios")
        json_d = {
            'id': step,
            'tag_seq': " ".join([id2label[x] for x in preds]),
            'entities': label_entities
        }
        results.append(json_d)
        pbar(step)
    with open(exp_dir + "logits.json", "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')

    output_submit_file = exp_dir + "result.json"
    test_text = []
    with open(exp_dir + "test_processed_split.json", 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {
            'id': x['id'],
            'label': {}
        }
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    with open(output_submit_file, "w", encoding="utf-8") as f:
        for t in test_submit:
            f.write(json.dumps(t) + "\n")


def compare(exp_dir, label):
    result = [json.loads(line.strip()) for line in open(exp_dir + "result.json", "r") if line.strip()]
    result = {r["id"]: r for r in result}
    gold = [json.loads(line.strip()) for line in open(exp_dir + "test_submit.json", "r", encoding="utf-8") if line.strip()]
    gold = {g["id"]: g for g in gold}
    for v in result.values():
        if label not in v["label"].keys():
            v["label"][label] = []
    for v in gold.values():
        if label not in v["label"].keys():
            v["label"][label] = []
    for k, v in result.items():
        for e in v["label"][label]:
            if e not in gold[k]["label"][label]:
                print(e)
    print()
    for k, v in gold.items():
        for e in v["label"][label]:
            if e not in result[k]["label"][label]:
                print(e)
    print()


if __name__ == "__main__":
    """parameters"""
    n_batch = 1
    features = torch.load("/home/zhanghang-s21/data/layoutlmft/myre/0424-joint-base")
    # features = torch.load("data2/cached_crf-test_RoBERTa_base_zh_wwm_128_cluener")
    # features = torch.load("data3/cached_crf-test_Electra_small_ex_zh_128_cluener")
    # features = torch.load("data4/cached_crf-test_Electra_small_zh_128_cluener")
    device = torch.device("cpu")
    exp_dir = "/home/zhanghang-s21/data/layoutlmft/myre/onnx"

    """模型转换及预测"""
    translate(exp_dir, features, n_batch, device, "electra")
    # model = ONNXModel(exp_dir + "generic_ner_1.0.0.onnx")
    # test(exp_dir, model, features, n_batch, device)

    """onnx-python预测结果比较"""
    # compare(exp_dir, "name")
    # compare(exp_dir, "address")
    # compare(exp_dir, "organization")
    # print(get_f1_score(exp_dir + "result.json", exp_dir + "test_processed_split.json"))
    # print(get_f1_score(exp_dir + "test_submit.json", exp_dir + "test_processed_split.json"))

