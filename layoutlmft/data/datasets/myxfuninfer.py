# Lint as: python3
import json
import logging
import os

import datasets

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from layoutlmft.data.utils import *
from transformers import AutoTokenizer
from copy import deepcopy
_URL = "/home/zhanghang-s21/data/bishe/MYXFUND/"

_MYURL = "/home/zhanghang-s21/data/bishe/nowdataset/"

_LANG = ["zh"]
logger = logging.getLogger(__name__)


class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text 
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs


        
        


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""
    row_id_max = -1
    column_id_max = -1
    BUILDER_CONFIGS = [XFUNConfig(name=f"myxfuninfer.{lang}", lang=lang) for lang in _LANG]
    ocr_data = {}
    input_ocr = ["/home/zhanghang-s21/data/bishe/ocr_data/aistrong_ocr_train", \
                "/home/zhanghang-s21/data/bishe/ocr_data/aistrong_ocr_mytrain", \
                "/home/zhanghang-s21/data/bishe/ocr_data/aistrong_ocr_val"]
    for filepath in input_ocr:
        with open(filepath, "r") as f:
            for line in f:
                name, data = line.split('\t')
                name = name.replace('.pdf','.jpg')
                ocr_data[name] = json.loads(data)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    # tokenizer.add_tokens(['<LONGTERM>'], special_tokens=True)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "len": datasets.Value("int64"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=["QUESTION", "ANSWER", "HEADER", "SINGLE", "ANSWERNUM"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER", "SINGLE", "ANSWERNUM"]),
                            "id":datasets.Value(dtype='string'),
                            "row_id":datasets.Value("int64"),
                            "column_id":datasets.Value("int64"),
                            "group_id":datasets.Value("int64"),
                            "index_id":datasets.Value("int64"),
                            "pred_label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER", "SINGLE", "ANSWERNUM"]),
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
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            # "train": [f"{_URL}{self.config.lang}_train.json", f"{_URL}{self.config.lang}.train.zip"],
            # "val": [f"{_URL}{self.config.lang}_val.json", f"{_URL}{self.config.lang}.val.zip"],
            # "train": [f"{_URL}{self.config.lang}_train.align.json", f"{_URL}{self.config.lang}.train.zip"],
            # "val": [f"{_URL}{self.config.lang}_val.align.json", f"{_URL}{self.config.lang}.val.zip"],
            # /home/zhanghang-s21/data/bishe/MYXFUND/mytrain.align.json
            # "train": [f"{_URL}mytrain.align.json", f"{_URL}mytrain.zip"],
            "val": [f"{_URL}myval.new.align.ner.json", f"{_URL}myval.zip"],
            "test": [f"{_URL}mytest.new.align.ner.json", f"{_URL}mytest.zip"],
            # "val": [f"{_URL}myval.align.ner.json", f"{_URL}myval.zip"],
        
            # "val": [f"{_URL}myval.ner.json", f"{_URL}myval.zip"],
            #
            # /home/zhanghang-s21/data/bishe/MYXFUND/mytrain.align.json
            
            # "test": [f"{_URL}{self.config.lang}.mytest.json", f"{_URL}{self.config.lang}.test.zip"],
            # "test": [f"{_MYURL}mytrain.json", f"{_MYURL}mytrain.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        # train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]
        test_files_for_many_langs = [downloaded_files["test"]]
        # if self.config.additional_langs:
        #     additional_langs = self.config.additional_langs.split("+")
        #     if "all" in additional_langs:
        #         additional_langs = [lang for lang in _LANG if lang != self.config.lang]
        #     for lang in additional_langs:
        #         urls_to_download = {"train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]}
        #         additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
        #         train_files_for_many_langs.append(additional_downloaded_files["train"])

        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            # datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs, \
            #     "MODE":"train"}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs, \
                    "MODE":"val"}
            ),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs,"MODE":"val"}),
        ]


    def get_boxes(self, boxes_tmp):
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
    
    def get_groups_from_boxes(self,boxes_tmp):
        # print("get_groups_from_boxes")
        if len(boxes_tmp) == 0:
            return [],[],[]
        groups,x_index, y_index = self.get_boxes(boxes_tmp)
        # groups 外部排序
        groups = sorted(groups, key=lambda x: (x[0][1],x[0][0]))
            # groups 内部排序
        for j, group in enumerate(groups):
            groups[j] = (groups[j][0],sorted(groups[j][1], key=lambda x: (y_index[x[0][1]], x_index[x[0][0]]) ))
 
        new_groups = []
        j = 0
        while j < len(groups):
            new_groups.append(groups[j][1])
            j+=1
            
        return new_groups,x_index,y_index


    def get_groups(self, MODE, doc, tables, size):
        # print("get_groups")
        document = doc["document"]
        id = doc['id']

        # tables = ocr_data[id][0]['tables']
        boxes_src = []
        group_src = []
        ocr_width, ocr_height, image_width, image_height = get_image(MODE,id)
        doc_tp = deepcopy(document)
        group_index = {}
        group_id = 0

        group_src_new = []
        x_index_src_new = []
        y_index_src_new = []
        
        for table_id, table in enumerate(tables):
            group_src = []
            boxes = []
            table_bbox =  normalizebbox(table['bbox'], ocr_width, ocr_height,image_width, image_height)
            
            i = 0
            while i < len(doc_tp):
                line =  doc_tp[i]
                bbox = line['box']
                if bbox_overlap(table_bbox,bbox):
                    boxes.append((bbox[1], bbox, line))
                    doc_tp.pop(i)
                    i-=1
                i+=1

            boxes = sorted(boxes, key=lambda x: x[0])
            boxes_tmp = deepcopy(boxes)
            groups,x_index,y_index = self.get_groups_from_boxes(boxes_tmp)
            # 行检测
            boxes = []
            for group in groups:
                group_src.append(group)
                for box, line in group:
                    # print(line['text'])
                    boxes.append(line)
                group_index[group_id] = len(boxes_src) + len(boxes)
                group_id += 1
            boxes_src.extend(boxes)
            group_src_new.append(group_src)
            x_index_src_new.append(x_index)
            y_index_src_new.append(y_index)
        
        group_src = []     
        boxes = []
        i = 0
        while i < len(doc_tp):
            line =  doc_tp[i]
            bbox = line['box']
            boxes.append((bbox[1], bbox, line))
            i+=1
        boxes = sorted(boxes, key=lambda x: x[0])
        groups,x_index,y_index = self.get_groups_from_boxes(boxes)
        
        boxes = []
        for group in groups:
            group_src.append(group)
            for box, line in group:
                boxes.append(line)
            group_index[group_id] = len(boxes_src) + len(boxes)
            group_id += 1
        boxes_src.extend(boxes)
        if len(group_src) > 0:
            group_src_new.append(group_src)
            x_index_src_new.append(x_index)
            y_index_src_new.append(y_index)
        # assert len(doc_tp) == insert_num
        # print("get_groups end")
        return group_src_new,x_index_src_new,y_index_src_new

    
    def get_relations(self, relations,id2label,entity_id_to_index_map,entities):
        # print("get_relation")
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
            if pair == ["question", "answer"]  or pair == ["question", "answernum"] :
                kvrelations.append(
                    {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                )
                if entity_id_to_index_map[rel[0]] >= entity_id_to_index_map[rel[1]]:
                    print("question,answer")
                    print(f"wrong in {rel[0]},{rel[1]}")
            elif pair == ["answer", "question"] or pair == ["answernum", "question"]:
                kvrelations.append(
                    {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                )
                if entity_id_to_index_map[rel[1]] >= entity_id_to_index_map[rel[0]]:
                    print("answer,question")
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
        # print("get_relation end")
        return relations
    
    
    
    def get_docs(self, group_src, size, x_index,y_index):
        group_doc_src = []
        entity_id_to_index_map = {}
        id2label = {}
        tokenizer = self.tokenizer
        
        group_total_len = 0
        group_max_len = -1
        for groups in group_src:
            group_doc = {"input_ids": [], "bbox": [], "labels": []}
            group_entities =[]
            # pre = len(entities)
            for group in groups:
                _, line = group
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
                label = [f"{line['label'].upper()}"] * len(tokenized_inputs["input_ids"])
                tokenized_inputs.update({"bbox": bbox, "labels": label})
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
                                "pred_label": line["pred_label"].upper(),
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
            tokenized_doc = {"input_ids": [],"bbox": [], "labels": []}
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
                        # self.row_id_max = max(group_entity["row_id"], self.row_id_max)
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
                
                self.row_id_max = max(group_entity["row_id"], self.row_id_max)
                self.column_id_max = max(group_entity["column_id"],self.column_id_max)
                
            # for n, group_entity in enumerate(entities):
            #     group_entity["row_id"] = map_interval(0,self.row_id_max,0,49,group_entity["row_id"])
            #     group_entity["column_id"] = map_interval(0,self.column_id_max,0,49,group_entity["column_id"])
            
            entity_id_to_index_map_src.append(entity_id_to_index_map)
            entities_src.append(entities)
            tokenized_doc_src.append(tokenized_doc)
            relations = self.get_relations(relations,id2label,entity_id_to_index_map,entities)
            relations_src.append(relations)

        for i in entities_src:
            for j in i:
                del j['linking']
        return tokenized_doc_src, entities_src, entity_id_to_index_map_src, relations_src
    
    
    def _generate_examples(self, filepaths, MODE):
        print(filepaths)
        ocr_data = self.ocr_data
        items = []

                
        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            self.row_id_max = -1
            self.column_id_max = -1
            for doc in data["documents"]:
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                id = doc['id']
                # zh_train_80.jpg
                print(id)
                if id == "mytrain_407.jpg":
                    last = True
                else:
                    pass
                    # continue
                tables = ocr_data[id][0]['tables']
                image, size = load_image(doc["img"]["fpath"])
                group_src_new,x_index_src_new,y_index_src_new  = self.get_groups(MODE, doc, tables, size)
                index_n = -1
                for group_n,group_src in enumerate(group_src_new):
                    x_index,y_index = x_index_src_new[group_n],y_index_src_new[group_n]
                    tokenized_doc_src, entities_src, entity_id_to_index_map_src, relations_src= self.get_docs(group_src, size,x_index,y_index)
                    
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
                                "id": f"{doc['id']}_{index_n}",
                                "len":len(item["input_ids"]), 
                                "image": image,
                                "entities": entities,
                                "relations": relations,
                            })
                        print(f"{doc['id']}_{index_n}",len(item['input_ids']))
                        print(f"self.row_id_max:{self.row_id_max};self.column_id_max:{self.column_id_max}")
                        yield f"{doc['id']}_{index_n}", item
                        