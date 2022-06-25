
import numpy as np 
import pandas as pd 
from transformers import AutoTokenizer
from collections import defaultdict
from data_mimic3 import MimicFullDataset, my_collate_fn, my_collate_fn_led
import json




version = "mimic3-50"
max_seq_length = 8192
# test_pred_path = "/home/zhichaoyang/mimic3/ICD-MSMN/error_analysis/predict_results_mimic3_msmn.txt"
test_pred_path = "/home/zhichaoyang/mimic3/longform/saved_models/longformer-base-clinical_xoutput/error_analysis_longformer-original-clinical-prompt2alpha-checkpoint-20165/predict_results_mimic3.txt"
# test_pred_path = "/home/zhichaoyang/mimic3/longform/saved_models/longformer-base-clinical_xoutput/predict_results_mimic3_v2.txt"
# test_pred_path = "/home/zhichaoyang/mimic3/longform/saved_models/longformer-base-clinical_xoutput/predict_results_mimic3_cheat.txt"
# test_pred_path = "/home/zhichaoyang/mimic3/longform/saved_models/longformer-base-clinical/predict_results_mimic3.txt"
tokenizer_path = "/home/zhichaoyang/mimic3/longform/saved_models/longformer-original-clinical-prompt2"

# load data
with open('/home/zhichaoyang/mimic3/ICD-MSMN/embedding/icd_mimic3_random_sort.json', 'r') as f:
    icd2des = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
train_dataset = MimicFullDataset(version, "train", max_seq_length, tokenizer)
eval_dataset  = MimicFullDataset(version, "test", max_seq_length, tokenizer)
predictions = pd.read_csv(test_pred_path, delimiter="\t")


# aa=0
# ab=0
# ba=0
# bb=0
# for prediction, example in zip(predictions.iterrows(), eval_dataset):
#     input_str = tokenizer.decode(example["input_ids"]).lower()
#     trut = prediction[1]["true2"]
#     pred = prediction[1]["prediction"]
#     trut = set(trut.split(";")) if type(trut)==str else set()
#     pred = set(pred.split(";")) if type(pred)==str else set()
#     if '285.1' in trut and '285.9' in pred:
#         ab += 1
#     if '285.9' in trut and '285.1' in pred:
#         ba += 1
#     if '285.9' in trut and '285.9' in pred:
#         bb += 1
#     if '285.1' in trut and '285.1' in pred:
#         aa += 1


# proc data file
assert len(predictions) == len(eval_dataset)
errors = []
precs = defaultdict(lambda: [0.0,0.0,0.0,0.0]) # TP, PP, PPV, FP_in_text
recas = defaultdict(lambda: [0.0,0.0,0.0,0.0]) # TP, P , TPR, FN_in_text
for prediction, example in zip(predictions.iterrows(), eval_dataset):
    input_str = tokenizer.decode(example["input_ids"]).lower()
    trut = prediction[1]["true2"]
    pred = prediction[1]["prediction"]
    trut = set(trut.split(";")) if type(trut)==str else set()
    pred = set(pred.split(";")) if type(pred)==str else set()
    fp = (pred - trut)
    fn = (trut - pred)
    for a in fp:
        for uni_str in icd2des[a][:20]:
            if uni_str in input_str:
                precs[a][3] += 1
                break
    for a in fn:
        for uni_str in icd2des[a][:20]:
            if uni_str in input_str:
                recas[a][3] += 1
                break
    for a in pred:
        precs[a][1] += 1
    for a in trut:
        recas[a][1] += 1
    for a in trut.intersection(pred):
        precs[a][0] += 1
        recas[a][0] += 1
    errors.append((prediction[1]["true1"],";".join(fp),";".join(fn)))
precs_all = [0.0,0.0]
for k,v in precs.items():
    precs[k] = (v[0],v[1],v[0]/v[1], v[3] )
    precs_all[0] += v[0]
    precs_all[1] += v[1]
precs_all = precs_all[0]/precs_all[1]
recas_all = [0.0,0.0]
for k,v in recas.items():
    recas[k] = (v[0],v[1],v[0]/v[1], v[3] )
    recas_all[0] += v[0]
    recas_all[1] += v[1]
recas_all = recas_all[0]/recas_all[1]

precs_sorted = sorted(precs.items(), key=lambda x: x[1][1]-x[1][0])
recas_sorted = sorted(recas.items(), key=lambda x: x[1][1]-x[1][0])
precs_sorted = sorted(precs.items(), key=lambda x: x[1][1])
recas_sorted = sorted(recas.items(), key=lambda x: x[1][1])

# np.array([b[3] for a,b in precs_sorted]).sum()

with open('errors_fp.txt', 'w') as f:
    for a,b,c in errors:
        f.write(f"{a}\t{b}\n")

with open('errors_fn.txt', 'w') as f:
    for a,b,c in errors:
        f.write(f"{a}\t{c}\n")

# check error
counta = 0
with open('abc.txt', 'w') as f:
    for prediction, example in zip(predictions.iterrows(), eval_dataset):
        input_str = tokenizer.decode(example["input_ids"]).lower()
        trut = prediction[1]["true2"]
        trut1 = set(trut.split(";")) if type(trut)==str else set()
        if "287.5" in trut1:
            tmp = input_str.replace('\n', ' ')
            f.write(f"{trut}\t{tmp}\n")
            if "thrombocytopenia" in input_str:
                counta += 1
countb = 0
countc = 0
label_list = train_dataset.ind2c
for example in train_dataset:
    input_str = tokenizer.decode(example["input_ids"]).lower()
    trut1 = []
    for indexx, label in enumerate(example['label_ids']):
        if label > 0:
            trut1 += [label_list[indexx]]
    if "287.5" in trut1:
        countb += 1
        if "thrombo" in input_str:
            countc += 1

print("Done")
