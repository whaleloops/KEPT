import numpy as np
import os
import torch
from evaluation import all_metrics
from data_mimic import MimicFullDataset, my_collate_fn, my_collate_fn_led, DataCollatorForMimic, modify_rule
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import ujson, json
from collections import defaultdict
from evaluation import all_metrics, stagfinal_eval

def printresult(metrics):
    print("------")
    sort_orders  = sorted(metrics.items(), key=lambda x: x[0], reverse=True)
    for k,v in sort_orders:
        print(k+":"+str(v))

def find_threshold_micro(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]
    # print(threshold)
    return threshold

def calc_oravle2(eval_dataset):
    num_icds = eval_dataset.code_count
    ysa = np.zeros((eval_dataset.len, num_icds), dtype=int)
    predsa = np.zeros((eval_dataset.len, num_icds), dtype=float) - 1
    with open(eval_dataset.path, "r") as f:
        df = ujson.load(f)
    for index in range(eval_dataset.len):
        labels = str(df[index]['LABELS']).split(';') 
        for label in labels:
            ysa[index,eval_dataset.c2ind[label]] = 1
        for a in set(eval_dataset.df[index]['ICD9s']).intersection(set(labels)):
            predsa[index,eval_dataset.c2ind[a]] = 1.0
    metricsa = all_metrics(ysa, predsa, k=[5, 8, 15, 50], threshold=0)
    return metricsa

def main():
    threshold = 0
    # dev_preds = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/oracle/dev_preds.pt")
    # dev_ys = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/oracle/dev_y.pt")
    # dev_icd9s = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/oracle/dev_icd9s.pt")
    # test_preds = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/oracle/test_preds.pt")
    # test_ys = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/oracle/test_y.pt")
    # test_icd9s = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/oracle/test_icd9s.pt")
    dev_preds = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/msmn-epoch4/dev_preds.pt")
    dev_ys = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/msmn-epoch4/dev_y.pt")
    dev_icd9s = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/msmn-epoch4/dev_icd9s.pt")
    test_preds = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/msmn-epoch4/test_preds.pt")
    test_ys = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/msmn-epoch4/test_y.pt")
    test_icd9s = torch.load("/home/zhichaoyang/mimic3/KEPT/tmptodel/msmn-epoch4/test_icd9s.pt")
    # metrics = all_metrics(ys, preds, k=[5, 8, 15], threshold=threshold)

    # predsa = np.concatenate((preds, np.zeros((3372,50), dtype=float)-1.0), axis=1)
    # ysa = np.concatenate((ys, np.zeros((3372,50), dtype=int)), axis=1)

    tokenizer = AutoTokenizer.from_pretrained(
        '/home/zhichaoyang/mimic3/KEPT/saved_models/longformer-original-clinical-promptfull-oracle',
        use_fast=True,
        revision='main',
        use_auth_token=None,
    )
    dev_dataset   = MimicFullDataset("mimic3", "dev", 8192, tokenizer, 30, 4)
    eval_dataset  = MimicFullDataset("mimic3", "test", 8192, tokenizer, 30, 4)
    # calc_oravle2(eval_dataset)


    # # update new predsa and ya
    # num_icds = eval_dataset.code_count
    # ysa = np.zeros((eval_dataset.len, num_icds), dtype=int)
    # predsa = np.zeros((eval_dataset.len, num_icds), dtype=float) - 1
    # for index in range(eval_dataset.len):
    #     c2y = defaultdict(lambda: 0)
    #     c2p = defaultdict(lambda: -5.0)
    #     for y, pred, icd9 in zip(ys[index], preds[index], icd9s[index]):
    #         c2y[icd9] = y
    #         c2p[icd9] = pred
    #     text   = df[index]['TEXT']
    #     labels = str(df[index]['LABELS']).split(';') 
    #     for label in labels:
    #         ysa[index,eval_dataset.c2ind[label]] = c2y[label]
    #         predsa[index,eval_dataset.c2ind[label]] = c2p[label]
    predsa, ysa = stagfinal_eval(dev_dataset, dev_preds, dev_ys, dev_icd9s)
    threshold = find_threshold_micro(predsa, ysa)
    metricsa = all_metrics(ysa, predsa, k=[5, 8, 15, 50], threshold=threshold)
    printresult(metricsa)
    predsa, ysa = stagfinal_eval(eval_dataset, test_preds, test_ys, test_icd9s)

    # metricsa = all_metrics(ysa, predsa, k=[5, 8, 15], threshold=threshold)
    metricsa = all_metrics(ysa, predsa, k=[5, 8, 15, 50], threshold=threshold)
    printresult(metricsa)

    print("Done")


if __name__ == "__main__":
    main()