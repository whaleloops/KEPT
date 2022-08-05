
import os
import torch
import json
from collections import defaultdict
import pickle
import json, ujson
import copy
from collections import defaultdict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    set_seed,
    LEDConfig,
    TrainingArguments
)
import numpy as np
from data_mimic import MimicFullDataset
from evaluation import all_metrics, stagfinal_eval

def printresult(metrics):
    print("------")
    sort_orders  = sorted(metrics.items(), key=lambda x: x[0], reverse=True)
    for k,v in sort_orders:
        print(k+":"+str(v))


output_dir = '/home/zhichaoyang/mimic3/KEPT/saved_models/longformer-base-clinical_xoutput_todel'
output_prediction_file = '/home/zhichaoyang/mimic3/KEPT/saved_models/longformer-base-clinical_xoutput_todel/mimic3_test_predtop50.txt'

preds = torch.load(os.path.join(os.path.join(output_dir, "tmptodel/"), "test_preds.pt"))
y = torch.load(os.path.join(os.path.join(output_dir, "tmptodel/"), "test_y.pt"))
icd9s = torch.load(os.path.join(os.path.join(output_dir, "tmptodel/"), "test_icd9s.pt"))

with open('/home/zhichaoyang/mimic3/KEPTGEN/data/icd2ind.json', 'r') as fp:
    icd2ind = json.load(fp)
ind2icd = defaultdict(list)
for k, v in icd2ind.items():
    ind2icd[v].append(k)
with open('/home/zhichaoyang/mimic3/KEPTGEN/data/code2des.json', 'r') as fp:
    code2des = json.load(fp)
des2code = defaultdict(list)
for k, v in code2des.items():
    des2code[v].append(k)

tokenizer = AutoTokenizer.from_pretrained("/home/zhichaoyang/mimic3/KEPTGEN/saved_models/icdgen-clinicva_todel")
eval_dataset = MimicFullDataset("mimic3", "test", 8192, tokenizer, 30, 4) 

predsa, ysa = stagfinal_eval(eval_dataset, preds, y, icd9s)
threshold = -0.545
metrics = all_metrics(ysa, predsa, k=[5,8,15,50], threshold=threshold)
printresult(metrics)


preds_new = []
for a, b in zip(preds, icd9s):
    tmp = []
    for aa, bb in zip(a,b):
        if aa > threshold:
            tmp.append(bb)
    preds_new.append(tmp)

tmp = [len(a) for a in preds_new]
tmp = np.array(tmp)

with open(output_prediction_file, "w") as writer:
    for a in preds_new:
        writer.write(";".join(a)+"\n")

print("Done")
