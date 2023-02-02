

# KEPT

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/knowledge-injected-prompt-based-fine-tuning/medical-code-prediction-on-mimic-iii)](https://paperswithcode.com/sota/medical-code-prediction-on-mimic-iii?p=knowledge-injected-prompt-based-fine-tuning)

This repository contains the implementation of our [KEPT](https://arxiv.org/abs/2210.03304) model on the auto icd coding task presented in EMNLP. This branch only contain code to experiment MIMIC-III-50 and MIMIC-III-rare50 in the paper. For MIMIC-III-full experiments, see the [rerank300 branch](https://github.com/whaleloops/KEPT/tree/rerank300).


Thanks to Zheng Yuan for opensourcing [MSMN](https://github.com/GanjinZero/ICD-MSMN) project, our evaluation code and data preprocsing step is heavily based on their repo. 


## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/) (currently tested on version 1.9.0+cu111)
* [Transformers](https://github.com/huggingface/transformers) (currently tested on version 4.16.2)
* tqdm==4.62.2
* ujson==5.3.0

Full environment setting is lised [here](conda-environment.yaml) and can be installed through:

```
conda env create -f conda-environment.yaml
conda activate ctorch191
```

## Download / preprocess data
One need to obtain licences to download MIMIC-III dataset. Once you obtain the MIMIC-III dataset, please follow [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to preprocess the dataset. You should obtain train_full.csv, test_full.csv, dev_full.csv, train_50.csv, test_50.csv, dev_50.csv after preprocessing. Please put them under sample_data/mimic3. Then you should use preprocess/generate_data_new.ipynb for generating json format dataset for train/dev/test. A new data will be saved in ./sample_data/mimic3.


## Modify constant
Modify constant.py : change DATA_DIR to where your preprocessed data located.

To enable wandb, modify wandb.init(project="PROJECTNAME", entity="WANDBACCOUNT") in run_coder.py.

## Generate MIMIC-III-rare50 data
Run command below and rare50 data will be created like mimic3-50l_xxx.json and xxx_50l.csv. The ./sample_data/mimic3 folder will look something like [this](data_files.PNG), without xxx_preds folders (they are prediction result for 2 stage reranker, see the correpdong branch to create those folders). 

```
python collectrare50data.py
```


## Hyperparameter and config setting to reproduce
If needed, see [wandb](https://wandb.ai/whaleloops/mimic_coder/runs/198spced/).


## Train and Eval

UMLS knowledge enhaced longformer is avail [here](https://huggingface.co/whaleloops/keptlongformer). 

To Train MIMIC-III 50 (2 A100 GPU):

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 57666 run_coder.py \
                --ddp_find_unused_parameters False \
                --disable_tqdm True \
                --version mimic3-50 --model_name_or_path whaleloops/keptlongformer \
                --do_train --do_eval --max_seq_length 8192 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1.5e-5 --weight_decay 1e-3 --adam_epsilon 1e-7 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step --global_attention_strides 1 \
                --output_dir ./saved_models/longformer-original-clinical-prompt2alpha
```

To Train MIMIC-III 50 (2 V100 GPU), with slightly less accuracy from less global attention in the prompt as global_attention_strides=3. More details could be found in [Table A.5](https://arxiv.org/abs/2210.03304):
```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 57666 run_coder.py \
                --ddp_find_unused_parameters False \
                --disable_tqdm True \
                --version mimic3-50 --model_name_or_path whaleloops/keptlongformer \
                --do_train --do_eval --max_seq_length 8192 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1.5e-5 --weight_decay 1e-3 --adam_epsilon 1e-7 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step --global_attention_strides 3 \
                --output_dir OUTPUT_DIR
```

A finetuned model (where global_attention_strides=1) is available [here](https://drive.google.com/file/d/1sv8cad8H1ajcKUis6qJFc7-9e9kWVeAv/view?usp=sharing). To eval MIMIC-III 50, change DOWNLOAD_MODEL_NAME_OR_PATH to the downloaded path:
```
CUDA_VISIBLE_DEVICES=0 python run_coder.py \
                --ddp_find_unused_parameters False \
                --disable_tqdm True \
                --version mimic3-50 --model_name_or_path DOWNLOAD_MODEL_NAME_OR_PATH \
                --do_eval --do_predict --max_seq_length 8192 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1.5e-5 --weight_decay 1e-3 --adam_epsilon 1e-7 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step --global_attention_strides 1 \
                --output_dir OUTPUT_DIR
```

To train and eval MIMIC-III rare50 by tuning only bias term and lm_head like [BitFit](https://aclanthology.org/2022.acl-short.1/), change DOWNLOAD_MODEL_NAME_OR_PATH to the downloaded path:
```
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 57666 run_coder.py \
                --ddp_find_unused_parameters False \
                --finetune_terms "bias;lm_head" \
                --disable_tqdm True \
                --version mimic3-50l --model_name_or_path DOWNLOAD_MODEL_NAME_OR_PATH \
                --do_train --do_eval --max_seq_length 8192 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1e-3 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --load_best_model_at_end True --metric_for_best_model eval_f1_macro --greater_is_better True \
                --logging_first_step --global_attention_strides 1 \
                --output_dir OUTPUT_DIR
```

## Citation

```
@article{Yang2022KnowledgeIP,
  title={Knowledge Injected Prompt Based Fine-tuning for Multi-label Few-shot ICD Coding},
  author={Zhichao Yang and Shufan Wang and Bhanu Pratap Singh Rawat and Avijit Mitra and Hong Yu},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.03304}
}
```


## License

See the [LICENSE](LICENSE) file for more details.

## Branches
### Supported branches
* [`main`](https://github.com/whaleloops/KEPT/tree/main): KEPTLongformer on MIMIC-III-50 and MIMIC-III-rare50.
* [`rerank300`](https://github.com/whaleloops/KEPT/tree/rerank300): KEPTLongformer as reranker on MIMIC-III-full.

### Deprecated branches
* [`cls50`](https://github.com/whaleloops/KEPT/tree/cls50): 1 global attention per label code description
* [`syn_explicit`](https://github.com/whaleloops/KEPT/tree/syn_explicit): Adding synonyms like MSMN per label code description

