
# KEPT

This repository contains the implementation of our KEPT model on the auto icd coding task presented in xxx.

## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/) (currently tested on version 1.9.0+cu111)
* [Transformers](https://github.com/huggingface/transformers) (currently tested on version 4.16.2)
* tqdm==4.62.2
* ujson==5.3.0

Full environment setting is lised [here](conda-environment.yaml).

## Download / preprocess data
One need to obtain licences to download MIMIC-III dataset. Once you obtain the MIMIC-III dataset, please follow [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to preprocess the dataset. You should obtain train_full.csv, test_full.csv, dev_full.csv, train_50.csv, test_50.csv, dev_50.csv after preprocessing. Please put them under sample_data/mimic3. Then you should use preprocess/generate_data_new.ipynb for generating json format dataset for train/dev/test. A new data will be saved in ./sample_data/mimic3.


## Modify constant
Modify constant.py : change DATA_DIR to where your preprocessed data located.

Modify wandb logging by changing run_coder.py this line wandb.init(project="mimic_coder", entity="whaleloops")

## Generate MIMIC-III-rare50 data
Run command below and rare50 data will be created like mimic3-50l_xxx.json and xxx_50l.csv. The ./sample_data/mimic3 folder will look something like [this](data_files.PNG), without xxx_preds folders (they are prediction result for 2 stage reranker, see the correpdong branch to create those folders). 

```
python collectrare50data.py
```

## Train and Eval

Train MIMIC-III 50 (2 A100 GPU):

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 57666 run_coder.py \
                --ddp_find_unused_parameters False \
                --disable_tqdm True \
                --version mimic3-50 --model_name_or_path MODEL_NAME_OR_PATH \
                --do_train --do_eval --max_seq_length 8192 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1.5e-5 --weight_decay 1e-3 --adam_epsilon 1e-7 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step \
                --output_dir ./saved_models/longformer-original-clinical-prompt2alpha
```
A finetuned model is available [here](https://drive.google.com/file/d/1sv8cad8H1ajcKUis6qJFc7-9e9kWVeAv/view?usp=sharing). To eval MIMIC-III 50:
```
CUDA_VISIBLE_DEVICES=0 python run_coder.py \
                --ddp_find_unused_parameters False \
                --disable_tqdm True \
                --version mimic3-50 --model_name_or_path DOWNLOAD_MODEL_NAME_OR_PATH \
                --do_eval --do_predict --max_seq_length 8192 \
                --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
                --learning_rate 1.5e-5 --weight_decay 1e-3 --adam_epsilon 1e-7 --num_train_epochs 8 \
                --evaluation_strategy epoch --save_strategy epoch \
                --logging_first_step \
                --output_dir OUTPUT_DIR
```

## Citation

Please cite the following if you find this repo useful.


## License

See the [LICENSE](LICENSE) file for more details.

