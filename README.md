# CLEVE: Contrastive Pre-training for Event Extraction

Source code for ACL 2021 paper "CLEVE: Contrastive Pre-training for Event Extraction"

## Requirements

- transformers == 2.5.0
- pytorch == 1.2.0

## Overview

Our pipeline contains four parts.

- NYT preprocessing
- AMR Parsing
- Pre-training
- Downstream Usage

## NYT Preprocessing

### Get dataset

Due to license limitation, we can't release NYT corpus. Please get dataset from [here](https://catalog.ldc.upenn.edu/LDC2008T19). We use ```${NYT_HOME}``` to denote the folder of NYT corpus.

### Preprocess

First, we need to prepare a environment with Python 2.7. Then

```shell
git clone https://github.com/notnews/nytimes-corpus-extractor.git
pip install -r requirements.txt
cd nytimes-corpus-extractor
python nytextract.py ${NYT_HOME}/data
```

Then we will get full text of NYT corpus in txt format in ```nytimes-corpus-extractor/text/nyt_corpus/data```. We use ```${NYT_TEXT_HOME}``` to denote this folder in the later description.

### Merge

```${NYT_TEXT_HOME}``` has plenty of folders and each folder has many txt files, which is not convinient for later operations. Use

```shell
python ${CLEVE_HOME}/AMR/sent_tokenize.py --data_dir ${NYT_TEXT_HOME} --num {NUM}
```

(This command needs Python 3.6)

```{NUM}``` means how many sentences in NYT corpus we actully use in our pretraining. ```30000``` would be enough for our task. This command will execute for about 4 hours. Now we have a file ```nyt_sent_limit.txt```. It contains one sentence per line. We use ```[input_sentence_file]``` to denote this file.



## AMR Parsing

In this section, we will use [CAMR](https://github.com/c-amr/camr)  to parse file ```[input_sentence_file]``` and use and [JAMR](https://github.com/jflanigan/jamr) to do alignment. Our goal is to get an AMR file with following format:

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

If you can use your own AMR parser to output this file, you can skip this section. We denote this file as ```[nyt_parsed_file]```.

### CAMR

We still need to use python 2.7 to run CAMR.

```shell
git clone https://github.com/c-amr/camr.git
pip install nltk==3.4.5
cd camr
./scripts/config.sh
```

nltk version should not be higher 3.4.5 since 3.4.5 is the latest version that support python2.7. Then please add ```ssplit.eolonly=true``` to  ```${CAMR_HOME}/stanfordnlp/default.properties``` (Otherwise a bug will occur)and set ```VERBOSE``` to ```False``` in ```${CAMR_HOME}/stanfordnlp/default.properties``` (Otherwise the speed will be much lower).

CAMR requires JDK 1.8. If you do not have JDK or your JDK version is not 1.8, you may not run CAMR successfully. You can download JDK 1.8 from Oracle and add JDK to you ```$PATH```.

```shell
python amr_parsing.py -m preprocess [input_sentence_file]
```

For 30000 sentences, this script will execute for about 8 hours. Now we get tokenized sentences(.tok), pos tag and name entity (.prp) and dependency structure (.charniak.parse.dep). Then Download model file and uncompress it:

```shell
wget http://www.cs.brandeis.edu/~cwang24/files/amr-anno-1.0.train.m.tar.gz
tar zxvf amr-anno-1.0.train.m.tar.gz
```

Now we can do parsing:

```shell
python amr_parsing.py -m parse --model [model_file] [input_sentence_file] 2>log/error.log
```

Now we get parsed AMR file (.parsed). Before we do alignment, we need to add tokens to AMR files.

```shell
python amr_parsing.py -m preprocess --amrfmt amr [input_amr_file]
```

Now we get a tokenized AMR file (.amr.tok). It should be like:

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



### JAMR

We still need Python 2.7 to run JAMR. To set up JAMR:

```
git clone https://github.com/jflanigan/jamr.git
git checkout Semeval-2016
```

JAMR requires sbt 0.13.18. If you do not have it (you can use ```sbt about``` to check), you need to install it via:

```
wget https://github.com/sbt/sbt/releases/download/v0.13.18/sbt-0.13.18.tgz
tar zxvf sbt-0.13.18.tgz
```

and then add it to your ```$PATH```. Next you could run following commands to set up JAMR:

```shell
bash ./setup
bash scripts/config.sh
./compile
```

Use this command to do alignment:

```shell
${JAMR_HOME}/run Aligner -v 0 --print-nodes-and-edges < [input_amr_tok_file] > [nyt_parsed_file]
```



## Pre-training

### Dataset

If you are running with ACE 2005, please preprocess format same as  [this repo](https://github.com/thunlp/HMEAE). If you are running with MAVEN, nothing needs to be done. Processed data should be stored in ```${DATA_HOME}``` (```${ACE_HOME}``` or ```${MAVEN_HOME}```)

### Pre-training

Now switch to Python 3.6. To get contrastive pretraining data, use:

```shell
python ${CLEVE_HOME}/AMR/load_AMR.py --amr_file [nyt_parsed_file]
```

You will get a file ```contrast_examples.pkl``` that contains pretraining data. Put it into ```${ACE_HOME}```. Then use following command to pretrain model:

```shell
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
    --num_train_epochs 100 \
    --save_steps 50 \
    --logging_steps 50 \
    --seed 233333 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --max_contrast_entity_per_sentence 10 \
    --do_pretrain \
```

You will get pretained model in ```${MODEL_DUMP_HOME}```. Please change ```${BATCH_SIZE}``` according to your GPU cards.



## Downstream Usage

### Supervised EE

To run Event Detection:

```bash
CUDA_VISIBLE_DEVICES=${GPU_ID} python run_ee.py \
    --data_dir ${DATA_HOME} \
    --model_type roberta \
    --model_name_or_path ${MODEL_DUMP_HOME}/checkpoint-XX \
    --task_name ${TASK_NAME} \
    --output_dir ${ED_MODEL_DUMP_HOME} \
    --max_seq_length 128 \
    --do_lower_case \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --per_gpu_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 50 \
    --save_steps 500 \
    --logging_steps 50 \
    --seed 233333 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
```

```${TASK_NAME}``` could be ```ace``` or ```maven```. Please change ```${BATCH_SIZE}``` according to your GPU cards.

After ED, you will get ```pred.json``` in ```${ED_MODEL_DUMP_HOME}```. To run Event Argument Extraction,  simply put this file to ```${DATA_HOME}``` and run:

```shell
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
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --save_steps 100 \
    --logging_steps 100 \
    --seed 11 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training
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
