# CLEVE: Contrastive Pre-training for Event Extraction

Source code for ACL 2021 paper "CLEVE: Contrastive Pre-training for Event Extraction"

## Requirements

- transformers == 2.5.0

- pytorch == 1.0.0

## Pre-training

### On Golden Data

Here we take pre-training on golden ACE 2005 English dataset as the first running example, which corresponds to `CLEVE on ACE (golden)` model in the paper.

At first, obtain the [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06) dataset and preprocess it as the format specified in [our previous work HMEAE](https://github.com/thunlp/HMEAE). You can directly use the script in [the repo](https://github.com/thunlp/HMEAE).

And then rush the below script to do pre-training. You can also modify some hyperparameters by yourself.

```bash
python3 run_ee.py \
    --data_dir path_to_preprocessed_ACE_data \
    --model_type  roberta\
    --model_name_or_path roberta-large \
    --task_name ace \
    --output_dir path_to_output_checkpoint \
    --max_seq_length 128 \
    --do_lower_case \
    --per_gpu_train_batch_size 42 \
    --per_gpu_eval_batch_size 42 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --save_steps 500 \
    --logging_steps 50 \
    --seed 233333 \
    --do_pretrain \
    --max_contrast_entity_per_sentence 10 \
```

### On AMR Parsing results

The basic pipeline to do pre-training on top of AMR parsing results is similar. We will soon update the AMR parsing example codes after cleansing.

## Downstream Usage

### Supervised Fine-tuning

```bash
python3 run_ee.py \
    --data_dir path_to_preprocessed_ACE_data \
    --model_type roberta \
    --model_name_or_path path_to_pretrained_checkpoints \
    --task_name ace \
    --output_dir path_to_output_results \
    --max_seq_length 128 \
    --do_lower_case \
    --per_gpu_train_batch_size 42 \
    --per_gpu_eval_batch_size 42 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --save_steps 500 \
    --logging_steps 50 \
    --seed 233333 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
```

### Unsupervised EE

Coming soon.

## Citation

If these codes help you, please cite our paper:

```bibtex
@inproceedings{wang-etal-2021-cleve,
    title = "{CLEVE}: {C}ontrastive {P}re-training for {E}vent {E}xtraction",
    author = "Wang, Ziqi  and Wang, Xiaozhi  and Han, Xu  and Lin, Yankai  and Hou, Lei  and Liu, Zhiyuan  and Li, Peng  and Li, Juanzi  and Zhou, Jie",
    booktitle = "Proceedings of ACL-IJCNLP",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.491",
    doi = "10.18653/v1/2021.acl-long.491",
    pages = "6283--6297",
}
```
