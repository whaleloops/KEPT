# KEPT

This repository contains the implementation of our KEPT reranker model on the auto icd coding task presented in AAAI.

## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/) (currently tested on version 1.9.0+cu111)
* [Transformers](https://github.com/huggingface/transformers) (currently tested on version 4.16.2)

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

A finetuned MSMN70 model is available [here](https://drive.google.com/drive/folders/1ylqyuP06CgHQN1KPJ0QTTptmmBFx8O9P?usp=sharing). To eval MIMIC-III on MSMN + GP SOAP:
```
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=disabled python run_coder.py --overwrite_output_dir --seed 47 \
--version mimic3 --model_name_or_path PATH_TO_MSMN70_MODEL \
--do_eval --max_seq_length 8192 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 \
--evaluation_strategy epoch --save_strategy no --logging_first_step --eval_steps 10 --save_steps 100000 \
--rerank_pred_folder1 ./sample_data/mimic3/msmn_preds/ \
--rerank_pred_folder2 ./sample_data/mimic3/keptgen_preds/ \
--output_dir ./saved_models/reranker_xoutput_msmn_gp1
```
With the following reuslt:
| Metric  | Score |
| ------------- | ------------- |
 |acc_micro| =0.41833426240884586|
 |acc_macro| =0.10366394048932563|
| eval_samples| =                3372|
 |rec_micro| =0.6061385289948231|
 |rec_macro| =0.15267296785350526|
 |rec_at_8| =0.40104061434674254|
 |rec_at_75| =0.8158778848621171|
 |rec_at_50| =0.8026150205285459|
 |rec_at_5| =0.2830237353017691|
 |rec_at_15| =0.5695721694625864|
 |prec_micro| =0.5744996640992382|
 |prec_macro| =0.14223086129915702|
 |prec_at_8| =0.755523428232503|
 |prec_at_75| =0.1903914590747331|
 |prec_at_50| =0.2782621589561092|
 |prec_at_5| =0.8254448398576513|
 |prec_at_15| =0.6064650059311981|
 |f1_micro| =0.5898951657536106|
 |f1_macro| =0.14726704483488753|
 |f1_at_8| =0.523958153040988|
 |f1_at_75| =0.30873678476178223|
 |f1_at_50| =0.4132521116402119|
 |f1_at_5| =0.42151935940715995|
 |f1_at_15| =0.5874399149256664|

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

