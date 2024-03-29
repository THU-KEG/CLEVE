# CLEVE: Contrastive Pre-training for Event Extraction

Source code for ACL 2021 paper "CLEVE: Contrastive Pre-training for Event Extraction"

## Requirements

- transformers == 2.5.0
- pytorch == 1.2.0
- nltk
- tqdm

Note: Test with CUDA 10.0. Higher torch version may occur bugs.

## Overview

Our pipeline contains four parts.

- NYT preprocessing
- AMR Parsing
- Pre-training
- Downstream Usage

If you don't want to pre-train by yourself, you can use our pre-trained checkpoint (based on roberta-large) and skip to ```Downstream Usage```. You can download the checkpoint from [Google Drive](https://drive.google.com/file/d/1i2R2_XyD47TphVavqU2rKQB0pjNRL3l4/view?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1_fVg0Eeyigoxq72BohDxdA) (Extraction Code: d5b2) and [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c92baf57fce24508ab73/?dl=1).

## NYT Preprocessing

### Get dataset

Due to the license limitation, we cannot release the New York Times Annotated Corpus used in our pre-training or provide the preprocessed files here. Please download the dataset from [here](https://catalog.ldc.upenn.edu/LDC2008T19). We use ```${NYT_HOME}``` to denote the path to the downloaded original NYT corpus.

### Preprocess

First, we need to prepare a Python 2.7 environment. Then:

```bash
git clone https://github.com/notnews/nytimes-corpus-extractor.git
cd nytimes-corpus-extractor
pip install -r requirements.txt
python nytextract.py ${NYT_HOME}/data
```

Then we will get full texts of the NYT corpus in `.txt` format in ```nytimes-corpus-extractor/text/nyt_corpus/data```. We use ```${NYT_TEXT_HOME}``` to denote this folder in later sections.

### Merge

```${NYT_TEXT_HOME}``` has plenty of folders and each folder has many `.txt` files, which is not convinient for later operations. Use

```bash
python ${CLEVE_HOME}/AMR/sent_tokenize.py --data_dir ${NYT_TEXT_HOME} --num {NUM}
```

(This command needs Python 3.6)

```${NUM}``` is the number of sentences in NYT we actully use in our pre-training. Here we use ```30000``` for example, but more sentences will make your pretraining better (but it will increse preprocessing time). We can not gurantee that using ```30000``` will definitely create a good pre-trained model, but this is a time-friendly option for you to get familiar to our pipelines (We used almost the whole NYT dataset in our experiments and the preprocessing lasts for more than a month!). This command will take about 4 hours. Then we will get a file ```nyt_sent_limit.txt```. It contains one sentence per line. We use ```[input_sentence_file]``` to denote this file.



## AMR Parsing

In this section, we will use [CAMR](https://github.com/c-amr/camr)  to parse the file ```[input_sentence_file]``` and  [JAMR](https://github.com/jflanigan/jamr) to do alignment. Our goal is to get an AMR file in the format like the following example:

```
# ::id 1
# ::snt It's as if Carl Lewis were an actor instead of an athlete.
# ::tok It 's as if Carl Lewis were an actor instead of an athlete .
# ::alignments 4-6|0.1.0+0.1.0.0+0.1.0.0.0+0.1.0.0.1 0-1|0.0 8-9|0.1 12-13|0 ::annotator Aligner v.03 ::date 2021-08-29T03:10:23.763
# ::node	0	athletes	12-13
# ::node	0.0	it	0-1
# ::node	0.1	actor	8-9
# ::node	0.1.0	newspaper	4-6
# ::node	0.1.0.0	name	4-6
# ::node	0.1.0.0.0	"Carl"	4-6
# ::node	0.1.0.0.1	"Lewis"	4-6
# ::root	0	athletes
# ::edge	actor	ARG0	newspaper	0.1	0.1.0	
# ::edge	athletes	domain	actor	0	0.1	
# ::edge	athletes	domain	it	0	0.0	
# ::edge	name	op1	"Carl"	0.1.0.0	0.1.0.0.0	
# ::edge	name	op2	"Lewis"	0.1.0.0	0.1.0.0.1	
# ::edge	newspaper	name	name	0.1.0	0.1.0.0	
(x13 / athletes
	:domain (x1 / it)
	:domain (x9 / actor
		:ARG0 (x5 / newspaper
			:name (n / name
				:op1 "Carl"
				:op2 "Lewis"))))

(Other instances....)
```

If you want to use another AMR parser to get this file, you can skip this section but keep the final file in the same format. We denote this file as ```[nyt_parsed_file]```.

### CAMR

We still need to use Python 2.7 to run CAMR.

```bash
git clone https://github.com/c-amr/camr.git
pip install nltk==3.4.5
cd camr
bash ./scripts/config.sh
```

nltk version should be not higher than `3.4.5` since `3.4.5` is the latest version supporting Python 2.7. Then please add ```ssplit.eolonly=true``` to  ```${CAMR_HOME}/stanfordnlp/default.properties``` (Otherwise a bug will occur) and set ```VERBOSE``` to ```False``` in ```${CAMR_HOME}/stanfordnlp/default.properties``` (Otherwise the speed will be much lower).

CAMR requires JDK 1.8. You can download JDK 1.8 from Oracle and add JDK to you environment variable ```$PATH```.

```bash
python amr_parsing.py -m preprocess [input_sentence_file]
```

For `30000` sentences, this script will execute for about `8` hours. Now we get tokenized sentences (`.tok`), POS tags and name entities (`.prp`) and dependency structures (`.charniak.parse.dep`). Then download model file and uncompress it:

```bash
wget http://www.cs.brandeis.edu/~cwang24/files/amr-anno-1.0.train.m.tar.gz
tar zxvf amr-anno-1.0.train.m.tar.gz
```

Now we can do parsing:

```bash
python amr_parsing.py -m parse --model [model_file] [input_sentence_file] 2>log/error.log
```

Now we get parsed AMR file (`.parsed`) (denote as```[input_amr_file]```). Before we do alignment, we need to add tokens to AMR files.

```bash
python amr_parsing.py -m preprocess --amrfmt amr [input_amr_file]
```

Now we get a tokenized AMR file (`.amr.tok`) (denote as```[input_amr_tok_file]```). It should be like:

```
# ::id 1
# ::snt It's as if Carl Lewis were an actor instead of an athlete.
# ::tok It 's as if Carl Lewis were an actor instead of an athlete .
(x13 / athletes
	:domain (x1 / it)
	:domain (x9 / actor
		:ARG0 (x5 / newspaper
			:name (n / name
				:op1 "Carl"
				:op2 "Lewis"))))

(Other instances....)
```

> Sometimes stanford corenlp will throw exceptions when processing some specific sentences, if you encounter sunch situations, simply delete that sentence and repeat steps above. This is a bug in stanford corenlp and we don't know how to fix it.

### JAMR

We still need Python 2.7 to run JAMR. To set up JAMR:

```bash
git clone https://github.com/jflanigan/jamr.git
git checkout Semeval-2016
```

JAMR requires `sbt == 0.13.18`. If you do not have it, you need to install it via:

```bash
wget https://github.com/sbt/sbt/releases/download/v0.13.18/sbt-0.13.18.tgz
tar zxvf sbt-0.13.18.tgz
```

And then add it to your ```$PATH``` and use ```sbt about``` to check if it is available. Next you could run following commands to set up JAMR:

```bash
bash ./setup
bash scripts/config.sh
./compile
```

Use this command to do alignment:

```bash
${JAMR_HOME}/run Aligner -v 0 --print-nodes-and-edges < [input_amr_tok_file] > [nyt_parsed_file]
```



## Pre-training

### Dataset

Please preprocess ACE 2005 to format same as  [this repo](https://github.com/thunlp/HMEAE). Processed data should be stored in ```${ACE_HOME}```.

### Pre-training

Now switch to Python 3.6. To get contrastive pre-training data, use:

```bash
python ${CLEVE_HOME}/AMR/load_AMR.py --amr_file [nyt_parsed_file]
```

You will get a file ```contrast_examples.pkl``` that contains pretraining data. Put it into ```${ACE_HOME}```. Then use following command to pre-train model:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python run_ee.py \
    --data_dir  ${ACE_HOME}\
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name ace \
    --output_dir ${MODEL_DUMP_HOME} \
    --max_seq_length 128 \
    --do_lower_case \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --per_gpu_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --save_steps 50 \
    --logging_steps 50 \
    --seed 233333 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --max_contrast_entity_per_sentence 20 \
    --do_pretrain \
```

You will get pretained model in ```${MODEL_DUMP_HOME}```. Please change ```${BATCH_SIZE}``` according to your GPU cards.



## Downstream Usage

To run event detection:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python run_ee.py \
    --data_dir ${ACE_HOME} \
    --model_type roberta \
    --model_name_or_path ${MODEL_DUMP_HOME}/checkpoint-XX \
    --task_name ace \
    --output_dir ${ED_MODEL_DUMP_HOME} \
    --max_seq_length 128 \
    --do_lower_case \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --per_gpu_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 5 \
    --learning_rate 1e-5 \
    --num_train_epochs 50 \
    --save_steps 100 \
    --logging_steps 100 \
    --seed 233333 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --max_contrast_entity_per_sentence 20 \
```

Please change ```${BATCH_SIZE}``` according to your GPU memory.

After event detection, you will get a ```pred.json``` file in ```${ED_MODEL_DUMP_HOME}```. To run event argument extraction, put this file to ```${ACE_HOME}``` and run:

```bash
cd EAE
CUDA_VISIBLE_DEVICES=${GPU_ID} python run_ee.py \
    --data_dir ${ACE_HOME} \
    --model_type roberta \
    --model_name_or_path ${MODEL_DUMP_HOME}/checkpoint-XX \
    --task_name ace_eae \
    --output_dir ${EAE_MODEL_DUMP_HOME} \
    --max_seq_length 128 \
    --do_lower_case \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --per_gpu_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --save_steps 200 \
    --logging_steps 200 \
    --seed 110 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
```

The parameters are similar with the event detection part.


## Structure Pre-training
Please see the ```struct``` folder.

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
