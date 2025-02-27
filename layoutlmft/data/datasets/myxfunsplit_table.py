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

    BUILDER_CONFIGS = [XFUNConfig(name=f"myxfunsplit_table.{lang}", lang=lang) for lang in _LANG]
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
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            # "train": [f"{_URL}{self.config.lang}_train.json", f"{_URL}{self.config.lang}.train.zip"],
            # "val": [f"{_URL}{self.config.lang}_val.json", f"{_URL}{self.config.lang}.val.zip"],
            # "train": [f"{_URL}{self.config.lang}_train.align.json", f"{_URL}{self.config.lang}.train.zip"],
            # "val": [f"{_URL}{self.config.lang}_val.align.json", f"{_URL}{self.config.lang}.val.zip"],
            # /home/zhanghang-s21/data/bishe/MYXFUND/mytrain.align.json
            "train": [f"{_URL}mytrain.align.json", f"{_URL}mytrain.zip"],
            "val": [f"{_URL}myval.new.align.json", f"{_URL}myval.zip"],
            "test": [f"{_URL}mytest.new.align.json", f"{_URL}mytest.zip"],
            # "val": [f"{_URL}myval.align.ner.json", f"{_URL}myval.zip"],
        
            # "val": [f"{_URL}myval.ner.json", f"{_URL}myval.zip"],
            #
            # /home/zhanghang-s21/data/bishe/MYXFUND/mytrain.align.json
            
            # "test": [f"{_URL}{self.config.lang}.mytest.json", f"{_URL}{self.config.lang}.test.zip"],
            # "test": [f"{_MYURL}mytrain.json", f"{_MYURL}mytrain.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]
        test_files_for_many_langs = [downloaded_files["test"]]
        if self.config.additional_langs:
            additional_langs = self.config.additional_langs.split("+")
            if "all" in additional_langs:
                additional_langs = [lang for lang in _LANG if lang != self.config.lang]
            for lang in additional_langs:
                urls_to_download = {"train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]}
                additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
                train_files_for_many_langs.append(additional_downloaded_files["train"])

        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs, \
                "MODE":"train"}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs, \
                    "MODE":"val"}
            ),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs,"MODE":"val"}),
        ]

    def get_boxes(self,boxes_tmp,compute_func,ratio):
        groups = []
        group = []
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
                    if compute_func(group_box,bbox) > ratio and \
                        compute_func(bbox,group_box) > ratio:
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
        return groups
    def get_groups_from_boxes(self,boxes_tmp):
        # print("get_groups_from_boxes")
        groups = []
        group = []
        groups = self.get_boxes(boxes_tmp,compute_y,70)
        # groups 外部排序
        groups = sorted(groups, key=lambda x: (x[0][1],x[0][0]))
            # groups 内部排序
        for j, group in enumerate(groups):
            groups[j] = (groups[j][0],sorted(groups[j][1], key=lambda x: (x[0][0],x[0][1])))

        new_groups = []
            
        j=0
        while j < len(groups):
            x_group_box = groups[j][0]
            group_tmp = deepcopy(groups[j][1])
            k=0
            while k < len(groups):
                if k != j:
                    y_group_box = groups[k][0]
                    if overlapping_rectangles(x_group_box,y_group_box)> 70 or \
                        overlapping_rectangles(y_group_box,x_group_box)> 70:
                        group_tmp.extend(groups[k][1])
                        groups.pop(k)
                        k-=1
                k+=1
            j+=1
            new_groups.append(group_tmp)
        j=0
        new_groups_src = []
        while j < len(new_groups):
            label_list = [i[1]['label'] for i in new_groups[j]]
            label_list = set(label_list)
            if len(label_list) == 1 and len(new_groups[j]) > 1 \
                and ("question" in label_list) :
                group_tmp = [(i[0][0],i[0],i[1]) for i in new_groups[j]]
                # print(1) 
                k = j+1
                flag = False
                while k < len(new_groups):
                    label_list = [i[1]['label'] for i in new_groups[k]]
                    label_list = set(label_list)
                    if len(label_list) == 1 and len(new_groups[k]) > 1 \
                        and ("answer" in label_list):
                        group_tmp.extend([(i[0][0],i[0],i[1]) for i in new_groups[k]])
                        new_groups.pop(k)
                        flag = True
                        k-=1
                        # print(2)
                    else:
                        break
                        
                    k+=1
                    # sorted(groups[j][1], key=lambda x: (x[0][0],x[0][1]))
                if flag:
                    group_tmp = self.get_boxes(group_tmp,compute_x,5)
                    group_tmp = [i[1] for i in group_tmp]
                    # group_tmp = sorted(group_tmp,key=lambda x: (x[0][0]))
                    new_groups_src.extend(group_tmp)
                else:
                    new_groups_src.append(new_groups[j])
            else:
                new_groups_src.append(new_groups[j])

            j+=1
        # print("get_groups_from_boxes end")
        return new_groups_src


    def get_groups(self, MODE, doc, tables, size):
        # print("get_groups")
        document = doc["document"]
        id = doc['id']

        # tables = ocr_data[id][0]['tables']
        boxes_src = []
        group_src = []
        ocr_width, ocr_height, image_width, image_height = get_image(MODE,id)
        doc_tp = deepcopy(document)
        table_index = {}
        group_index = {}
        group_id = 0

        for table_id, table in enumerate(tables):
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
            groups = self.get_groups_from_boxes(boxes_tmp)
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
            table_index[table_id] = len(boxes_src)
                
        insert_num = 0    
        for line in doc_tp:
            bbox = line['box']
            insert_id = 0
            for table_id, table in enumerate(tables):
                table_bbox =  normalizebbox(table['bbox'], ocr_width, ocr_height,image_width, image_height)
                if bbox[1] >= table_bbox[1]:
                    insert_id = table_id
                    break
            boxes_src.insert(table_index[insert_id],line)
            for index in group_index:
                if group_index[index] >= table_index[insert_id]:
                    group_src[index].append((bbox, line))
                    insert_num+=1
                    break
        assert len(doc_tp) == insert_num
        # print("get_groups end")
        return group_src
    
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
            if pair == ["question", "answer"]:
                kvrelations.append(
                    {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                )
            elif pair == ["answer", "question"]:
                kvrelations.append(
                    {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                )
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
    
    
    
    def get_docs(self, group_src, size):
        group_doc_src = []
        entity_id_to_index_map = {}
        id2label = {}
        tokenizer = self.tokenizer
        
        group_total_len = 0
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
                                "label": line["label"].upper(),
                            }
                    )
                for j in group_doc:
                    group_doc[j] = group_doc[j] + tokenized_inputs[j]     
            group_doc_src.append((len(group_doc["input_ids"]), group_entities, group_doc))
            group_total_len += len(group_doc["input_ids"])
            
    
        tokenized_doc_src = []
        entities_src = []
        entity_id_to_index_map_src = []
        relations_src = []

        maxsteps = 512
        import math
        if group_total_len > 512:
            j = math.ceil(group_total_len/512)
            maxsteps = min(512 // j + 50, 512)
        while len(group_doc_src) > 0:
            i = 0
            # print(len(group_doc_src))
            tokenized_doc = {"input_ids": [],"bbox": [], "labels": []}
            entity_id_to_index_map = {}
            entities = []
            relations = []
            group_id = 0
            while i < len(group_doc_src):
                group_len, group_entities, group_doc = group_doc_src[i]

                    
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
                        group_entity["index_id"] = n
                        relations.extend([tuple(sorted(l)) for l in group_entity["linking"]])
                    
                    group_id +=1
                    entities.extend(group_entities)
                    group_doc_src.pop(i)
                    i-=1

                else:
                    # print("what!")
                    group_id = 0
                    break
                i+=1
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
                group_src = self.get_groups(MODE, doc, tables, size)
                tokenized_doc_src, entities_src, entity_id_to_index_map_src, relations_src= self.get_docs(group_src, size)
                
                # relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
                
                for n, tokenized_doc in enumerate(tokenized_doc_src):
                    entities = entities_src[n]
                    entity_id_to_index_map = entity_id_to_index_map_src[n]
                    relations = relations_src[n]
                    item = {}
                    for k in tokenized_doc:
                        item[k] = tokenized_doc[k]
                    
                    item.update(
                        {
                            "id": f"{doc['id']}_{n}",
                            "len":len(item["input_ids"]), 
                            "image": image,
                            "entities": entities,
                            "relations": relations,
                        })
                    print(f"{doc['id']}_{n}",len(item['input_ids']))
                    yield f"{doc['id']}_{n}", item