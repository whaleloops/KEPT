# KEPT

This repository contains the implementation of our KEPT reranker model on the auto icd coding task presented in AAAI.

## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/) (currently tested on version 1.9.0+cu111)
* [Transformers](https://github.com/huggingface/transformers) (currently tested on version 4.16.2)
* tqdm==4.62.2
* ujson==5.3.0

Full environment setting is lised [here](conda-environment.yaml).

## Download / preprocess data
One need to obtain licences to download MIMIC-III dataset. Once you obtain the MIMIC-III dataset, please follow [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to preprocess the dataset. You should obtain train_full.csv, test_full.csv, dev_full.csv, train_50.csv, test_50.csv, dev_50.csv after preprocessing. Please put them under ./sample_data/mimic3/. Then you should use preprocess/generate_data_new.ipynb for generating json format dataset for train/dev/test. A new data will be saved in ./sample_data/mimic3/. 


## Modify constant
Modify constant.py : change DATA_DIR to where your preprocessed data located such as ./sample_data

## Collect 1st stage model predictions
To generate predictions from MSMN, see their repo [here](https://github.com/GanjinZero/ICD-MSMN). To save your time, we ran their code and predictions are available [here](https://drive.google.com/drive/folders/1XbnPwD2FNzEgnYoF3-3ruruItsG3De_e?usp=sharing), download and unzip folder to ./sample_data/mimic3/.

To generate predictions from AGMHT, see their repo [here](https://github.com/csong27/gzsl_text). To save your time, we ran their code and predictions are available [here](https://drive.google.com/drive/folders/1Z7J3W2JvnPB8TrKbSyBXuiVvGdAoeloM?usp=sharing), download and unzip folder to ./sample_data/mimic3/.

To generate predictions from GP SOAP, see their repo here. To save your time, we ran their code and predictions are available [here](https://drive.google.com/drive/folders/1jJGCNCV2E5UGM1GYYU4oI0K_B5vsxsiv?usp=sharing), download and unzip folder to ./sample_data/mimic3/.

All predictions order follows the same train/dev/test .json order that you just did in preprocessing.

## Eval

A finetuned MSMN model is available [here](https://drive.google.com/drive/folders/1ylqyuP06CgHQN1KPJ0QTTptmmBFx8O9P?usp=sharing). To eval MIMIC-III on MSMN + GP SOAP:
```
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=disabled python run_coder.py --overwrite_output_dir --seed 47 \
--version mimic3 --model_name_or_path PATH_TO_MSMN_MODEL \
--do_eval --max_seq_length 8192 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
--evaluation_strategy epoch --save_strategy no --logging_first_step --eval_steps 10 --save_steps 100000 \
--rerank_pred_folder1 ./sample_data/mimic3/msmn_preds/ \
--rerank_pred_folder2 ./sample_data/mimic3/keptgen_preds/ \
--output_dir ./saved_models/reranker_xoutput_msmn_gp1
```
With the following reuslt:
| Metric  | Score |
| ------------- | ------------- |
| acc_macro   | = 0.08786680886727369|
| acc_micro   | =  0.4265015806111691|
| auc_macro   | =  0.7271945707693832|
| auc_micro   | =  0.8752969149719587|
| eval_samples| =                3372|
| f1_at_15    | =  0.5934809738405847|
| f1_at_5     | =  0.4311171236707107|
| f1_at_50    | =  0.4017349062524513|
| f1_at_8     | =  0.5338691886573289|
| f1_macro    | = 0.12057606836332291|
| f1_micro    | =  0.5979686057248373|
| prec_at_15  | =  0.6136812969553183|
| prec_at_5   | =  0.8456109134045078|
| prec_at_50  | =  0.2699584816132859|
| prec_at_8   | =  0.7712781731909846|
| prec_macro  | = 0.12525060454031595|
| prec_micro  | =  0.6408980376632912|
| rec_at_15   | =  0.5745681247305229|
| rec_at_5    | =  0.2893071852073718|
| rec_at_50   | =   0.784846800502318|
| rec_at_8    | = 0.40821542080441947|
| rec_macro   | = 0.11623789954909551|
| rec_micro   | =  0.5604292354861033|

Similarly, to eval MIMIC-III on MSMN + AGMHT:
```
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=disabled python run_coder.py --overwrite_output_dir --seed 47 \
--version mimic3 --model_name_or_path PATH_TO_MSMN_MODEL \
--do_eval --max_seq_length 8192 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
--evaluation_strategy epoch --save_strategy no --logging_first_step --eval_steps 10 --save_steps 100000 \
--rerank_pred_folder1 ./sample_data/mimic3/msmn_preds/ \
--rerank_pred_folder2 ./sample_data/mimic3/agmht_preds/ \
--output_dir ./saved_models/reranker_xoutput_msmn_agmht1
```

A finetuned oracle model is available [here](https://drive.google.com/drive/folders/1pWo4Ikb3UBy3x1vLW47OOgbu1hk7uMt7?usp=sharing). To eval MIMIC-III on oracle setting:
```
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=disabled python run_coder.py --overwrite_output_dir --seed 47 \
--do_oracle \
--version mimic3 --model_name_or_path PATH_TO_ORACLE_MODEL \
--do_eval --max_seq_length 8192 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
--evaluation_strategy epoch --save_strategy no --logging_first_step --eval_steps 10 --save_steps 100000 \
--rerank_pred_folder1 ./sample_data/mimic3/msmn_preds/ \
--rerank_pred_folder2 ./sample_data/mimic3/keptgen_preds/ \
--output_dir ./saved_models/reranker_xoutput_oracle
```
With the following reuslt:
| Metric  | Score |
| ------------- | ------------- |
| acc_macro    |=  0.2149537519309995|
| acc_micro    |=   0.521996814092953|
| auc_macro    |=  0.9977499779299266|
| auc_micro    |=  0.9989342029714507|
| eval_samples |=                3372|
| f1_at_15     |=  0.6162406569171164|
| f1_at_5      |=  0.4284793553573903|
| f1_at_50     |=  0.5288116452272519|
| f1_at_8      |=  0.5323252154109755|
| f1_macro     |=  0.2676004250199075|
| f1_micro     |=  0.6859368025734548|
| prec_at_15   |=   0.642487149070779|
| prec_at_5    |=  0.8518386714116252|
| prec_at_50   |=  0.3594839857651245|
| prec_at_8    |=  0.7789145907473309|
| prec_macro   |=  0.2669427394690533|
| prec_micro   |=  0.6433108147121649|
| rec_at_15    |=  0.5920544120963408|
| rec_at_5     |= 0.28622636514686367|
| rec_at_50    |=  0.9997002026763976|
| rec_at_8     |=  0.4043240114319444|
| rec_macro    |=  0.2682613593470444|
| rec_micro    |=  0.7346124682688808|

## Citation

Please cite the following if you find this repo useful.


## License

See the [LICENSE](LICENSE) file for more details.

