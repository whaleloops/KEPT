# KEPT

This repository contains the implementation of our KEPT model on the auto icd coding task presented in xxx.

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
One need to obtain licences to download MIMIC-III dataset. Once you obtain the MIMIC-III dataset, please follow [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to preprocess the dataset. You should obtain train_full.csv, test_full.csv, dev_full.csv, train_50.csv, test_50.csv, dev_50.csv after preprocessing. Please put them under ./sample_data/mimic3/. Then you should use preprocess/generate_data_new.ipynb for generating json format dataset for train/dev/test. A new data will be saved in ./sample_data/mimic3/. 


## Modify constant
Modify constant.py : change DATA_DIR to where your preprocessed data located such as ./sample_data

## Collect 1st stage model predictions
To generate predictions from MSMN, see their repo [here](https://github.com/GanjinZero/ICD-MSMN). To save your time, we ran their code and top300 predictions are available [here](https://drive.google.com/drive/folders/1UZWn-uokPYVejY-9ljZoTlRt8t7iQhcV?usp=sharing), download the whole folder and unzip folder to ./sample_data/mimic3/.


## Eval

A finetuned model is available [here](https://drive.google.com/drive/folders/1ia0PxQ3b35q22_Wwj39Em99qley_nAwy?usp=sharing). To eval MIMIC-III-full:
```
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=disabled python run_coder.py --overwrite_output_dir --seed 47 \
--version mimic3 --model_name_or_path PATH_TO_MSMN_MODEL \
--do_eval --max_seq_length 8192 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
--evaluation_strategy epoch --save_strategy no --logging_first_step --eval_steps 10 --save_steps 100000 \
--rerank_pred_folder1 ./sample_data/mimic3/msmn_300_preds --global_attention_strides 3 \
--output_dir ./saved_models/reranker_xoutput_msmn300
```
With the following reuslt:
| Metric  | Score |
| ------------- | ------------- |
|rec_micro| =0.5729403619819988|
|rec_macro| =0.11342156911120573|
|rec_at_8| =0.4094837705486378|
|rec_at_75| =0.8470734920535119|
|rec_at_50| =0.8005338782352|
|rec_at_5| =0.2891628170355805|
|rec_at_15| =0.5768778119750537|
|prec_micro| =0.6411968713105065|
|prec_macro| =0.12227610414493029|
|prec_at_8| =0.7760972716488731|
|prec_at_75| =0.197504942665085|
|prec_at_50| =0.2768090154211151|
|prec_at_5| =0.8483392645314354|
|prec_at_15| =0.6178529062870699|
|f1_micro| =0.6051499904242899|
|f1_macro| =0.11768251595637802|
|f1_at_8| =0.536107150495997|
|f1_at_75| =0.32032290907137506|
|f1_at_50| =0.411373195944102|
|f1_at_5| =0.43131028155283435|
|f1_at_15| =0.5966627077602488|
|auc_micro| =0.9651754312635265|
|auc_macro| =0.8566590059725866|
|acc_micro| =0.43384592341105344|
|acc_macro| =0.08639139221100567|



## Train

UMLS knowledge enhaced longformer is avail [here](https://huggingface.co/whaleloops/keptlongformer). 

You could also train with below command on 2 A100 (40GB) GPU with 80 hour+. 
```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --master_port 56666 run_coder.py \
                --ddp_find_unused_parameters False --seed 47 \
                --disable_tqdm True \
                --version mimic3 --model_name_or_path whaleloops/keptlongformer \
                --do_train --do_eval --max_seq_length 8192 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1.5e-5 --weight_decay 1e-4 --adam_epsilon 1e-7 --num_train_epochs 4 --warmup_ratio 0.1 \
                --evaluation_strategy epoch --save_strategy epoch \
                --load_best_model_at_end True --metric_for_best_model eval_f1_macro --greater_is_better True \
                --rerank_pred_folder1 ./sample_data/mimic3/msmn_300_preds \
                --logging_first_step --gradient_accumulation_steps 6 --global_attention_strides 3 \
                --output_dir ./saved_models/longformer-original-clinical-promptfull-oracle-300-rand
```

## Citation

Please cite the following if you find this repo useful.


## License

See the [LICENSE](LICENSE) file for more details.

