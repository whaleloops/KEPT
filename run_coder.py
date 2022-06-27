#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import os
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from data_mimic import MimicFullDataset, my_collate_fn, my_collate_fn_led, modify_rule
from tqdm import tqdm
import json
import sys
import numpy as np
from evaluation import all_metrics
# from train_parser import generate_parser, print_metrics
# from train_utils import generate_output_folder_name, generate_model
# from find_threshold import find_threshold_micro

import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

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
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from model import LongformerForMaskedLM

import wandb
logger = logging.getLogger(__name__)

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

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    version: Optional[str] = field(
        default=None, metadata={"help": "mimic version"}
    )



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

def main(): 
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.broadcast_buffers = False
        
    # Setup logging
    if is_main_process(training_args.local_rank):
        wandb.init(project="mimic_coder", entity="whaleloops")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    # accelerator = Accelerator(kwargs_handlers=kwargs_handlers)

    # word_embedding_path = training_args.word_embedding_path         
    # logger.info(f"Use word embedding from {word_embedding_path}")
    
    train_dataset = MimicFullDataset(data_args.version, "train", data_args.max_seq_length, tokenizer, 30, 4) # TODO delete 30 and 8
    dev_dataset   = MimicFullDataset(data_args.version, "dev", data_args.max_seq_length, tokenizer, 30, 4)
    eval_dataset  = MimicFullDataset(data_args.version, "test", data_args.max_seq_length, tokenizer, 30, 4)

    num_labels = train_dataset.code_count 
    # load config, model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = LongformerForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if config.model_type == "longformer": 
        data_collator = my_collate_fn
    elif config.model_type == "led":
        data_collator = my_collate_fn_led
        model.use_cache=False
        model.gradient_checkpointing=True 
    else:
        data_collator = default_data_collator


    # Get the metric function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        y = p.label_ids==10932
        
        aaa = y[np.newaxis,:,:].reshape((-1,4,50))
        predsa = preds[np.newaxis,:,:].reshape((-1,4,50))
        predsb=predsa.max(axis=1)
        y = aaa[:,0,:]
        preds = predsb

        result = all_metrics(y, preds, k=[5, 8, 15])
        return result

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = ['mimic3']
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = ['mimic3']
        predict_datasets = [eval_dataset]

        label_list = train_dataset.ind2c

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            p = trainer.predict(dev_dataset, metric_key_prefix="dev")
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

            aaa = y[np.newaxis,:,:].reshape((-1,4,50))
            predsa = preds[np.newaxis,:,:].reshape((-1,4,50))
            predsb=predsa.max(axis=1)
            y = aaa[:,0,:]
            preds = predsb
            y = p.label_ids==10932
            threshold = find_threshold_micro(preds, y)



            p = trainer.predict(predict_dataset, metric_key_prefix="predict")
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            y = p.label_ids==10932
            torch.save(preds, "./tmptodel/preds.pt")
            torch.save(y, "./tmptodel/y.pt")
            preds = torch.load("./tmptodel/preds.pt")
            y = torch.load("./tmptodel/y.pt")

            aaa = y[np.newaxis,:,:].reshape((-1,4,50))
            predsa = preds[np.newaxis,:,:].reshape((-1,4,50))
            predsb=predsa.max(axis=1)
            y = aaa[:,0,:]
            preds = predsb

            result = all_metrics(y, preds, k=[5, 8, 15], threshold=threshold)
            preds_new = modify_rule(y, preds, predict_dataset, train_dataset.ind2c, train_dataset.c2ind, tokenizer)
            result = all_metrics(y, preds_new, k=[5, 8, 15], threshold=threshold)
            preds = preds_new

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}_cheat.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\ttrue1\ttrue2\tprediction\n")
                    index = 0
                    assert len(predict_dataset.df) == len(preds)
                    for orig, proc, item in zip(predict_dataset.df, predict_dataset, preds):
                        itwm_to_write_row1 = str(orig['LABELS'])
                        tmp = itwm_to_write_row1.split(';')
                        itwm_to_write_row2 = []
                        for indexx, label in enumerate(proc['label_ids']):
                            if label > 0:
                                itwm_to_write_row2 += [label_list[indexx]]
                        itwm_to_write_row2 = ";".join(itwm_to_write_row2) 
                        itwm_to_write_row3 = []
                        for indexx, prob in enumerate(item):
                            if prob > 0:
                                itwm_to_write_row3 += [label_list[indexx]]
                        itwm_to_write_row3 = ";".join(itwm_to_write_row3)  
                        writer.write(f"{index}\t{itwm_to_write_row1}\t{itwm_to_write_row2}\t{itwm_to_write_row3}\n")
                        index += 1
        logger.info("*** Done for Predict ***")



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()