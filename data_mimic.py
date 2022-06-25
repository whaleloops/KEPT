
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import os
from torch.utils.data import Dataset
from constant import DATA_DIR, MIMIC_2_DIR, MIMIC_3_DIR
import sys
import re
import pandas as pd
import numpy as np
import csv
import ujson, json
from collections import defaultdict

InputDataClass = NewType("InputDataClass", Any)


def get_headersandindex(input_str):
    input_str = input_str.lower()
    headers_to_select = ["chief complaint:", "major surgical or invasive procedure:", "procedure:", "history of present illness:", "past eedical history:", "brief hospital course:", "discharge diagnosis:", "discharge diagnoses:", "discharge condition:"]
    strs = input_str.split("\n")
    headers = []
    for str_tmp in strs:
        str_tmp = str_tmp.strip()
        if len(str_tmp)>0 and str_tmp[-1] == ':':
            headers.append(str_tmp)
    headers_pos = []
    last_index = 0
    for header in headers:
        starts = last_index + input_str[last_index:].index(header)
        last_index = starts + len(header)
        headers_pos.append((header,starts))
    headers_pos += [("end:", len(input_str))]
    counta = 0
    finals = []
    while counta < len(headers_pos)-1:
        (header,starts) = headers_pos[counta]
        if header in headers_to_select:
            finals.append((header, starts, headers_pos[counta+1][1])) # (section headername, start of section, end of section)
        counta += 1
    return finals

def get_subnote(input_str, headers_pos):
    result = ""
    for (header, starts, ends) in headers_pos:
        result += input_str[starts:ends]
    return result

    
        
def create_main_code(ind2c):
    mc = list(set([c.split('.')[0] for c in set(ind2c.values())]))
    mc.sort()
    ind2mc = {ind:mc for ind, mc in enumerate(mc)}
    mc2ind = {mc:ind for ind, mc in ind2mc.items()}
    return ind2mc, mc2ind

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

def load_code_descriptions(version='mimic3'):
    # load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    if version == 'mimic2':
        with open('%s/MIMIC_ICD9_mapping' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            # header
            next(r)
            for row in r:
                desc_dict[str(row[1])] = str(row[2])
    else:
        with open("%s/D_ICD_DIAGNOSES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            # header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                desc_dict[reformat(code, True)] = desc
        with open("%s/D_ICD_PROCEDURES.csv" % (DATA_DIR), 'r') as descfile:
            r = csv.reader(descfile)
            # header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                if code not in desc_dict.keys():
                    desc_dict[reformat(code, False)] = desc
        with open('%s/ICD9_descriptions' % DATA_DIR, 'r') as labelfile:
            for _, row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = ' '.join(row[1:])
    return desc_dict

def load_full_codes(train_path, version='mimic3'):
    """
        Inputs:
            train_path: path to train dataset
            version: which (MIMIC) dataset
        Outputs:
            code lookup, description lookup
    """
    # get description lookup
    desc_dict = load_code_descriptions(version=version)
    # build code lookups from appropriate datasets
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open('%s/proc_dsums.csv' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            # header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i: c for i, c in enumerate(sorted(codes))})
    return ind2c, desc_dict



class MimicFullDataset(Dataset):
    def __init__(self, version, mode, truncate_length, tokenizer,
                 label_truncate_length=30, term_count=1):
        self.version = version
        self.mode = mode
        self.tokenizer = tokenizer

        if version == 'mimic2':
            raise NotImplementedError
        if version in ['mimic3', 'mimic3-50', 'mimic3-50l']:
            self.path = os.path.join(MIMIC_3_DIR, f"{version}_{mode}.json")

        if version in ['mimic3']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_full.csv")
        if version in ['mimic3-50']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_50.csv")
        if version in ['mimic3-50l']:
            self.train_path = os.path.join(MIMIC_3_DIR, "train_50l.csv")

        with open(self.path, "r") as f:
            self.df = ujson.load(f)

        self.ind2c, desc_dict = load_full_codes(self.train_path, version=version)
        # self.part_icd_codes = list(self.ind2c.values())
        self.c2ind = {c: ind for ind, c in self.ind2c.items()}
        self.code_count = len(self.ind2c)
        if mode == "train":
            print(f'Code count: {self.code_count}')
        
        self.ind2mc, self.mc2ind = create_main_code(self.ind2c)
        self.main_code_count = len(self.ind2mc)
        if mode == "train":
            print(f'Main code count: {self.main_code_count}')

        self.len = len(self.df)
        self.truncate_length = truncate_length
        
        # prep prompt
        if version == "mimic3-50": #TODO: remove unique sorted ICD_50_RANK for mimic3-50
            desc_list = []
            ICD_50_RANK = [   ('401.9', (715.0, 778.0, 0.9190231362467867, 25.0)),
                ('272.4', (485.0, 548.0, 0.885036496350365, 0.0)),
                ('427.31', (436.0, 470.0, 0.9276595744680851, 1.0)),
                ('414.01', (376.0, 435.0, 0.864367816091954, 0.0)),
                ('428.0', (355.0, 422.0, 0.8412322274881516, 0.0)),
                ('38.93', (224.0, 402.0, 0.5572139303482587, 0.0)),
                ('584.9', (267.0, 362.0, 0.7375690607734806, 5.0)),
                ('250.00', (290.0, 340.0, 0.8529411764705882, 0.0)),
                ('530.81', (227.0, 266.0, 0.8533834586466166, 0.0)),
                ('96.71', (183.0, 258.0, 0.7093023255813954, 0.0)),
                ('518.81', (183.0, 255.0, 0.7176470588235294, 1.0)),
                ('599.0', (200.0, 251.0, 0.796812749003984, 1.0)),
                ('285.9', (54.0, 247.0, 0.21862348178137653, 13.0)),
                ('96.04', (169.0, 233.0, 0.7253218884120172, 2.0)),
                ('96.6', (189.0, 228.0, 0.8289473684210527, 0.0)),
                ('39.61', (225.0, 226.0, 0.995575221238938, 0.0)),
                ('403.90', (168.0, 215.0, 0.7813953488372093, 0.0)),
                ('244.9', (198.0, 210.0, 0.9428571428571428, 1.0)),
                ('285.1', (121.0, 203.0, 0.5960591133004927, 0.0)),
                ('311', (145.0, 196.0, 0.7397959183673469, 0.0)),
                ('276.2', (72.0, 195.0, 0.36923076923076925, 5.0)),
                ('V15.82', (43.0, 187.0, 0.22994652406417113, 13.0)),
                ('96.72', (136.0, 187.0, 0.7272727272727273, 0.0)),
                ('V58.61', (150.0, 181.0, 0.8287292817679558, 0.0)),
                ('305.1', (115.0, 181.0, 0.6353591160220995, 2.0)),
                ('486', (125.0, 174.0, 0.7183908045977011, 1.0)),
                ('585.9', (112.0, 172.0, 0.6511627906976745, 6.0)),
                ('995.92', (114.0, 165.0, 0.6909090909090909, 1.0)),
                ('496', (125.0, 160.0, 0.78125, 0.0)),
                ('88.56', (136.0, 157.0, 0.8662420382165605, 0.0)),
                ('272.0', (99.0, 155.0, 0.6387096774193548, 0.0)),
                ('276.1', (73.0, 155.0, 0.47096774193548385, 0.0)),
                ('038.9', (99.0, 149.0, 0.6644295302013423, 0.0)),
                ('38.91', (38.0, 148.0, 0.25675675675675674, 0.0)),
                ('287.5', (60.0, 138.0, 0.43478260869565216, 7.0)),
                ('36.15', (134.0, 135.0, 0.9925925925925926, 0.0)),
                ('33.24', (87.0, 127.0, 0.6850393700787402, 0.0)),
                ('412', (90.0, 121.0, 0.743801652892562, 1.0)),
                ('V45.81', (97.0, 108.0, 0.8981481481481481, 0.0)),
                ('424.0', (59.0, 104.0, 0.5673076923076923, 0.0)),
                ('507.0', (66.0, 102.0, 0.6470588235294118, 0.0)),
                ('37.22', (70.0, 100.0, 0.7, 4.0)),
                ('511.9', (39.0, 95.0, 0.4105263157894737, 27.0)),
                ('410.71', (64.0, 91.0, 0.7032967032967034, 0.0)),
                ('99.15', (64.0, 91.0, 0.7032967032967034, 0.0)),
                ('45.13', (73.0, 87.0, 0.8390804597701149, 0.0)),
                ('88.72', (26.0, 84.0, 0.30952380952380953, 7.0)),
                ('39.95', (69.0, 72.0, 0.9583333333333334, 0.0)),
                ('37.23', (30.0, 60.0, 0.5, 0.0)),
                ('99.04', (28.0, 51.0, 0.5490196078431373, 0.0))]
            assert len(ICD_50_RANK) == len(self.ind2c)
            for icd9, info in ICD_50_RANK:
                desc_list.append(desc_dict[icd9].lower().split(",")[0])
        else:
            desc_list = []
            ICD_50_RANK = [(v,0) for k,v in self.ind2c.items()]
            for icd9, info in ICD_50_RANK:
                desc_list.append(desc_dict[icd9].lower().split(",")[0])
        
        if term_count == 1:
            c_desc_list = desc_list
        else:
            c_desc_list = []
            with open(f'./icd_mimic3_random_sort.json', 'r') as f: #TODO: change path
                icd_syn = ujson.load(f)
            for (code, info), tmp_desc in zip(ICD_50_RANK,desc_list):
                tmp_desc = [tmp_desc]
                new_terms = icd_syn.get(code, [])
                if len(new_terms) >= term_count - 1:
                    tmp_desc.extend(new_terms[0:term_count - 1])
                else:
                    tmp_desc.extend(new_terms)
                    repeat_count = int (term_count / len(tmp_desc)) + 1
                    tmp_desc = (tmp_desc * repeat_count)[0:term_count]
                c_desc_list.append(tmp_desc)

        descriptions = " " + " <mask>, ".join(desc_list) + " <mask>. "
        tmp = self.tokenizer.tokenize(descriptions)
        self.global_window = len(tmp) + 1
        assert self.global_window < 501 # only for gpu memory efficiency

        self.label_yes = self.tokenizer("yes")['input_ids'][1] # 10932
        self.label_no  = self.tokenizer("no")['input_ids'][1]  # 2362

        # num_raw_token = []
        # num_pro_token = []
        # for index in range(self.len):
        #     text = self.df[index]['TEXT']
        #     text = re.sub(r'\[\*\*[^\]]*\*\*\]', '', text)  # remove any mimic special token like [**2120-2-28**] or [**Hospital1 3278**]
        #     text = re.sub(r'  +', ' ', text)
        #     label = str(self.df[index]['LABELS']).split(';') 
        #     # self.process(text, label)
        #     tmp = self.tokenizer.tokenize(text)
        #     # num_raw_token.append(len(self.tokenizer.convert_tokens_to_string(tmp[:self.truncate_length]).split()))
        #     num_pro_token.append(len(tmp))
        # num_pro_token = np.array(num_pro_token)
        # print(f'Num of examples exceed max length {self.truncate_length}: {(num_pro_token > self.truncate_length).sum()} / {len(num_pro_token)}')
        # print(f'Avg text length: {num_pro_token.mean()}')
        # print(f'Std text length: {np.std(num_pro_token)}')
        # print(f'Med text length: {np.median(num_pro_token)}')
        # print(f'Max text length: {num_pro_token.max()}')
        # print(f'Min text length: {num_pro_token.min()}')

        num_pro_token = []
        to_sav = []
        countb = 0
        for index in range(self.len):
            text = self.df[index]['TEXT']
            text = re.sub(r'\[\*\*[^\]]*\*\*\]', '', text)  # remove any mimic special token like [**2120-2-28**] or [**Hospital1 3278**]
            tmp = self.tokenizer.tokenize(descriptions + re.sub(r'  +', ' ', text.lower().replace("\n"," ")))
            if len(tmp) <= self.truncate_length:
                num_pro_token.append(len(tmp))
                self.df[index]['TEXT'] = descriptions + re.sub(r'  +', ' ', text.lower().replace("\n"," "))
            else:
                headers_pos = get_headersandindex(text)
                if len(headers_pos) > 1:
                    new_text = get_subnote(text, headers_pos)
                    tmp = self.tokenizer.tokenize(descriptions + re.sub(r'  +', ' ', new_text.lower().replace("\n"," ")))
                    countb += 1
                    text = new_text
                # else:
                #     to_sav.append((str(self.df[index]['LABELS']),text))
                num_pro_token.append(len(tmp))
                self.df[index]['TEXT'] = descriptions + re.sub(r'  +', ' ', text.lower().replace("\n"," "))
        num_pro_token = np.array(num_pro_token)
        print(f'Num of examples exceed max length {self.truncate_length}: {(num_pro_token > self.truncate_length).sum()} / {len(num_pro_token)}')
        print(f'Avg text length: {num_pro_token.mean()}')
        print(f'Std text length: {np.std(num_pro_token)}')
        print(f'Med text length: {np.median(num_pro_token)}')
        print(f'Max text length: {num_pro_token.max()}')
        print(f'Min text length: {num_pro_token.min()}')

        # with open('abc3.txt', 'w') as f:
        #     for a,b in to_sav:
        #         f.write(f"xxx\n{a}\n{b}\n")


    def __len__(self):
        return self.len

    def process(self, text, label):
        input_word = self.tokenizer(text, padding='max_length', truncation='longest_first', max_length=self.truncate_length, 
            return_token_type_ids=True, return_attention_mask=True
            )
        binary_label = [self.label_no] * self.code_count
        for l in label:
            if l in self.c2ind:
                binary_label[self.c2ind[l]] = self.label_yes
                
        # main_label = [0] * self.main_code_count
        # for l in label:
        #     if l.split('.')[0] in self.mc2ind:
        #         main_label[self.mc2ind[l.split('.')[0]]] = 1
        
        input_word["label_ids"] = torch.tensor(binary_label, dtype=torch.long)
        return input_word


    def __getitem__(self, index):
        # proc label 
        label = str(self.df[index]['LABELS']).split(';')
        # proc input
        text = self.df[index]['TEXT']
        processed = self.process(text, label)
        return processed



@dataclass
class DataCollatorForMimic:
    global_attention_mask_size: int

    def __call__(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        global_attention_mask = torch.zeros_like(batch["input_ids"])
        # global attention on cls token
        global_attention_mask[:,0:self.global_attention_mask_size] = 1 # TODO change 408 to self.global_window
        batch["global_attention_mask"] = global_attention_mask

        return batch

def my_collate_fn(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    return 0

def my_collate_fn_led(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k != "token_type_ids":
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

    eos_mask = batch["input_ids"].eq(2) # majic number: 2 is config.eos_token_id for led
    global_attention_mask = torch.zeros_like(batch["input_ids"])
    # global attention on cls token
    global_attention_mask[eos_mask] = 1
    batch["global_attention_mask"] = global_attention_mask
    batch["decoder_input_ids"] = torch.tensor([[2]]*batch["input_ids"].shape[0])

    return batch

def modify_rule(ys, preds, examples, ind2c, c2ind, tokenizer):
    preds = np.copy(preds)
    with open('/home/zhichaoyang/mimic3/ICD-MSMN/embedding/icd_mimic3_random_sort.json', 'r') as f:
        icd2des = json.load(f)
    assert len(ys) == len(preds)
    assert len(ys) == len(examples)
    add_rules = ["511.9", "285.9", "287.5", "401.9", "584.9", "530.81", "276.2", "585.9"]
    counta = 0
    count_change= 0
    for y, pred, example in zip(ys, preds, examples):
        input_str = tokenizer.decode(example["input_ids"]).lower()
        trut = []
        for indexx, label in enumerate(y):
            if label > 0:
                trut += [ind2c[indexx]]
        to_adds = []
        for a in trut:
            if a in add_rules:
                to_adds.append(a)
        for a in set(to_adds):
            for uni_str in icd2des[a][:20]:
                if uni_str in input_str:
                    if pred[c2ind[a]] < 0:
                        count_change += 1
                        preds[counta][c2ind[a]] = 0.5
        counta += 1
    # print(count_change)
    return preds


