# coding=utf-8

import json
import os

import datasets

from layoutlmft.data.utils import load_image, normalize_bbox
from layoutlmft.data.utils import *
from transformers import AutoTokenizer
from copy import deepcopy

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""
label2ids =  {
    "O":0,
    'B-HEADER':1,
    'I-HEADER':2,
    'B-QUESTION':3,
    'I-QUESTION':4,
    'B-ANSWER':5,
    'I-ANSWER':6,
}
XFund_label2ids = {
    'HEADER':0,
    'QUESTION':1,
    'ANSWER':2,
    'OTHER':3,
    # "ANSWERNUM":4,
}
tokenizer = AutoTokenizer.from_pretrained("/home/zhanghang-s21/data/model/xlm-roberta-base")

class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)

def get_relations(relations,id2label,entity_id_to_index_map,entities):
    relations = list(set(relations))
    kvrelations = []
    i = 0
    while i < len(relations):
        rel = relations[i]
        if rel[0] not in entity_id_to_index_map.keys() \
            or rel[1] not in entity_id_to_index_map.keys():
            relations.pop(i)
            # print("wrong")
            i-=1
        i+=1
    for rel in relations:
        pair = [id2label[rel[0]], id2label[rel[1]]]           
        if pair == ["question", "answer"] or pair == ["question", "answernum"] :
            kvrelations.append(
                {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
            )
            if entity_id_to_index_map[rel[0]] >= entity_id_to_index_map[rel[1]]:
                print(f"wrong in {rel[0]} and{rel[1]}")
        elif pair == ["answernum", "question"] or pair == ["answer", "question"]:
            kvrelations.append(
                {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
            )
            if entity_id_to_index_map[rel[1]] >= entity_id_to_index_map[rel[0]]:
                print(f"wrong in {rel[1]},{rel[0]}")
        else:
            continue
    def get_relation_span(rel):
        bound = []
        for entity_index in [rel["head"], rel["tail"]]:
            bound.append(entities[entity_index]["start"])
            bound.append(entities[entity_index]["end"])
        return min(bound), max(bound)
    relations = sorted(
            [
                {
                    "head": rel["head"],
                    "tail": rel["tail"],
                    "start_index": get_relation_span(rel)[0],
                    "end_index": get_relation_span(rel)[1],
                }
                for rel in kvrelations
            ],
            key=lambda x: x["head"],
        )
    return relations


def get_boxes(boxes_tmp):
    groups = []
    group = []
    min_y = 30000
    min_x = 30000
    x_list = set()
    y_list = set()
    for tp in boxes_tmp:
        _, bbox, line = tp
        min_y = min(min_y,bbox[3]-bbox[1])
        min_x = min(min_x,bbox[2]-bbox[0])
        x_list.add(bbox[0])
        x_list.add(bbox[2])
        y_list.add(bbox[1])
        y_list.add(bbox[3])
    if len(y_list) != 0:
        y_index = group_by_threshold(list(y_list),min_y // 2)
        x_index = group_by_threshold(list(x_list),min_x // 2)
        
    while len(boxes_tmp) > 0:
        k = 0
        while k < len(boxes_tmp):
            _, bbox, line = boxes_tmp[k]
            if group == []:
                group.append((bbox, line))
                boxes_tmp.pop(k)
                k-=1
            else:
                boxs = []
                for box, _ in group:
                    boxs.append(box)
                group_box = merge_bbox(boxs)
                if get_overlap_byrelative(group_box,bbox,y_index):
                    group.append((bbox, line)) 
                    boxes_tmp.pop(k)
                    k-=1
            k+=1
        boxs = []
        for box, _ in group:
            boxs.append(box)
        group_box = merge_bbox(boxs)
        groups.append((group_box,group))
        group=[]
        
    if group != []:
        groups.append(group)
    return groups,x_index,y_index

def get_groups_from_boxes(boxes_tmp):
    # print("get_groups_from_boxes")
    if len(boxes_tmp) == 0:
        return [],[],[]
    groups,x_index, y_index = get_boxes(boxes_tmp)
    # groups 外部排序
    groups = sorted(groups, key=lambda x: (x[0][1],x[0][0]))
        # groups 内部排序
    for j, group in enumerate(groups):
        groups[j] = (groups[j][0],sorted(groups[j][1], key=lambda x: (y_index[x[0][1]], x_index[x[0][0]]) ))

    new_groups = [i[1] for i in groups] 
    return new_groups,x_index,y_index


def get_groups(doc):
    # print("get_groups")
    document = doc
    boxes = []
    doc_tp = deepcopy(document)
    boxes = [(line['box'][1],line['box'],line) for line in doc_tp]
    boxes = sorted(boxes, key=lambda x: x[0])
    boxes_tmp = deepcopy(boxes)
    groups,x_index,y_index = get_groups_from_boxes(boxes_tmp)
    return groups,x_index,y_index


def get_docs( group_src, size, x_index,y_index):
    group_doc_src = []
    entity_id_to_index_map = {}
    id2label = {}
    # tokenizer = tokenizer
    
    group_total_len = 0
    group_max_len = -1
    for groups in group_src:
        group_doc = {"input_ids": [], "bbox": [], "labels": [], "ner_labels":[]}
        group_entities =[]
        # pre = len(entities)
        for group in groups:
            _, line = group
            if len(line["text"]) == 0:
                continue
            if len(line["text"]) >= 90:
                # print(line["text"])
                line["text"] = "#$$$$$$$#"
                # print(line["label"])
            tokenized_inputs = tokenizer(
                    line["text"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    return_attention_mask=False,
                )

            id2label[line["id"]] = line["label"]
            bbox= [(normalize_bbox(simplify_bbox(line["box"]), size))] * len(tokenized_inputs["input_ids"])

            label = [XFund_label2ids[line['label'].upper()]] * len(tokenized_inputs["input_ids"])
            cur_label = line['label'].upper()
            label_len = len(tokenized_inputs["input_ids"])
            if cur_label == 'OTHER':
                cur_labels = ["O"] * label_len
                for k in range(len(cur_labels)):
                    cur_labels[k] = label2ids[cur_labels[k]]
            else:
                cur_labels = [cur_label] * label_len
                cur_labels[0] = label2ids['B-' + cur_labels[0]]
                for k in range(1, len(cur_labels)):
                    cur_labels[k] = label2ids['I-' + cur_labels[k]]
            
            ner_label = cur_labels
            assert len(bbox) == len(label)
            tokenized_inputs.update({"bbox": bbox, "labels": label, "ner_labels":ner_label})
            # entity_id_to_index_map[line["id"]] = pre + len(group_entities)
            
            group_entities.append(
                        {
                            "start": len(group_doc["input_ids"]),
                            "end": len(group_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                            "id": line["id"],
                            "linking": line["linking"],
                            "row_begin_id":y_index[line["box"][1]],
                            "row_end_id":y_index[line["box"][3]],
                            "column_begin_id":x_index[line["box"][0]],
                            "column_end_id":x_index[line["box"][2]],  
                            "label": line["label"].upper(),
                            # "pred_label": line["pred_label"].upper(),
                        }
                )
            for j in group_doc:
                group_doc[j] = group_doc[j] + tokenized_inputs[j]     
        group_doc_src.append((len(group_doc["input_ids"]), group_entities, group_doc))
        group_total_len += len(group_doc["input_ids"])
        group_max_len = max(group_max_len,len(group_doc["input_ids"]))

    tokenized_doc_src = []
    entities_src = []
    entity_id_to_index_map_src = []
    relations_src = []

    maxsteps = 512
    import math
    if group_total_len > 512:
        j = math.ceil(group_total_len/512)
        maxsteps = min(512 // j + 50, 512)
        if group_max_len > maxsteps:
            maxsteps = min(group_max_len+50,512)
    while len(group_doc_src) > 0:
        i = 0
        # print(len(group_doc_src))
        tokenized_doc = {"input_ids": [],"bbox": [], "labels": [], "ner_labels": []}
        entity_id_to_index_map = {}
        entities = []
        relations = []
        group_id = 0
        row_index = set()
        while i < len(group_doc_src):
            group_len, group_entities, group_doc = group_doc_src[i]
            
            column_index = set()
            row_dict,column_dict = {},{}
            if group_len + len(tokenized_doc['input_ids']) <= maxsteps:
                pre = len(tokenized_doc["input_ids"])
                pre_index = len(entities)
                for j in tokenized_doc:
                    tokenized_doc[j] = tokenized_doc[j] + group_doc[j]
                for n, group_entity in enumerate(group_entities):
                    group_entity["start"] = group_entity["start"] + pre
                    group_entity["end"] = group_entity["end"] + pre
                    entity_id_to_index_map[group_entity["id"]] = pre_index + n
                    group_entity["id"] = pre_index + n
                    group_entity["group_id"] = group_id
                    # row_id_max = max(group_entity["row_id"], row_id_max)
                    group_entity["index_id"] = n
                    row_begin_id,row_end_id,column_begin_id,column_end_id = \
                        group_entity["row_begin_id"], group_entity["row_end_id"], \
                        group_entity["column_begin_id"], group_entity["column_end_id"]
                    row_index.add(row_begin_id)
                    column_index.add(column_begin_id)
                    relations.extend([tuple(sorted(l)) for l in group_entity["linking"]])
                
                column_index = list(column_index)
                column_index.sort()
                column_dict = {k:v for k,v in zip(column_index,list(range(0, len(column_index), 1)))}

                for n, group_entity in enumerate(group_entities):
                    row_begin_id,row_end_id,column_begin_id,column_end_id = \
                        group_entity["row_begin_id"], group_entity["row_end_id"], \
                        group_entity["column_begin_id"], group_entity["column_end_id"]
                    group_entity["column_id"] = column_dict[column_begin_id]
                    
                group_id +=1
                entities.extend(group_entities)
                group_doc_src.pop(i)
                i-=1

            else:
                # print("what!")
                group_id = 0
                break
            i+=1
        
        row_index = list(row_index)
        row_index.sort()
        row_dict = {k:v for k,v in zip(row_index,list(range(0, len(row_index), 1)))}
        for n, group_entity in enumerate(entities):
            row_begin_id = group_entity["row_begin_id"]
            del group_entity["column_begin_id"]
            del group_entity["column_end_id"]
            del group_entity["row_begin_id"]
            del group_entity["row_end_id"]
            group_entity["row_id"] = row_dict[row_begin_id]
            # group_id_max = max(group_entity["group_id"],group_id_max)
            # index_id_max = max(group_entity["index_id"],index_id_max)
            
            # row_id_max = max(group_entity["row_id"], row_id_max)
            # column_id_max = max(group_entity["column_id"],column_id_max)
            
       
        entity_id_to_index_map_src.append(entity_id_to_index_map)
        entities_src.append(entities)
        tokenized_doc_src.append(tokenized_doc)
        relations = get_relations(relations,id2label,entity_id_to_index_map,entities)
        relations_src.append(relations)

    for i in entities_src:
        for j in i:
            del j['linking']
    return tokenized_doc_src, entities_src, entity_id_to_index_map_src, relations_src


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd_new", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "len": datasets.Value("int64"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER","OTHER"])),
                    # ,
                    "ner_labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=['O','B-HEADER','I-HEADER','B-QUESTION','I-QUESTION','B-ANSWER','I-ANSWER']
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER","OTHER"]),
                            "id":datasets.Value(dtype='string'),
                            "row_id":datasets.Value("int64"),
                            "column_id":datasets.Value("int64"),
                            "group_id":datasets.Value("int64"),
                            "index_id":datasets.Value("int64"),
                            # "pred_label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER", "SINGLE", "ANSWERNUM"]),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "start_index": datasets.Value("int64"),
                            "end_index": datasets.Value("int64"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract("/home/zhanghang-s21/data/DATA/funsd/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        base_id = -1 
    
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)

            # for doc in :
            doc = data["form"]
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
                # doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
            # print(guid)
            # tables = ocr_data[id][0]['tables']
            group_src,x_index,y_index  = get_groups(doc)
            index_n = -1
            # for group_n,group_src in enumerate(group_src_new):
            #     x_index,y_index = x_index_src_new[group_n],y_index_src_new[group_n]
            tokenized_doc_src, entities_src, entity_id_to_index_map_src, relations_src= get_docs(group_src, size,x_index,y_index)
            
            # relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
            
            for n, tokenized_doc in enumerate(tokenized_doc_src):
                entities = entities_src[n]
                entity_id_to_index_map = entity_id_to_index_map_src[n]
                relations = relations_src[n]
                item = {}
                for k in tokenized_doc:
                    item[k] = tokenized_doc[k]
                index_n+=1
                item.update(
                    {
                        "id": base_id,
                        "len":len(item["input_ids"]), 
                        "image": image,
                        "entities": entities,
                        "relations": relations,
                    })
            
            base_id+=1
            print(base_id)
            yield f"{base_id}", item
            # yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}
