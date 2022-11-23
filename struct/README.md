# CLEVE Structure Pre-training

This folder contains structure pre-training codes. Codes here are copied from [GCC](https://github.com/THUDM/GCC) with minor modifications, which will be illustrated below. As stated in our paper, this part is running [GCC](https://github.com/THUDM/GCC) with our inputs.

## Prerequisites

Same as [GCC](https://github.com/THUDM/GCC). Please ensure you can run pre-training codes in [GCC](https://github.com/THUDM/GCC).

## Pre-training
* Follow the README in the main folder and get ```[nyt_parsed_file]```.
* Put ```glove.6B.300d.txt``` to this folder.
* Convert AMR file to the DGL format ```python load_AMR.py --amr_file [nyt_parsed_file]```, and the converted file will in ```GCC/data```
* Pre-training: ```bash scripts/pretrain.sh <gpu> --moco --nce-k 16384```

## Down-stream Usage
Dumped parameters can be loaded into the model ```GCC/gcc/models/graph_encoder.py```. The output ```x``` is graph representations, ```node_reps``` is node representations. The pre-trained model can be used to AMR-knowledge-related tasks. More details for down-stream usage can be found in [GCC](https://github.com/THUDM/GCC).

## Main difference between the GCC and this codebase
* input files are different. Our DGL file contains token semantic information and edge information. Core codes: ```load_AMR.py```.
* Node embedding and edge embedding. We use semantic embedding (Glove embedding) for nodes and edges (random initialization). GCC ignores edge information and semantic embedding for nodes. Core codes: ```GCC/gcc/models/graph_encoder.py, GCC/train.py```. Note: Of course, glove embedding can be replaced by contextual representations such as BERT and RoBERTa. However, BERT and RoBERTa will require you have much more GPU memory. Due to the hardware limitation, we use glove embedding here.
  
## Cite
Please cite GCC and our paper if you find this code helpful.

```
@article{qiu2020gcc,
  title={GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training},
  author={Qiu, Jiezhong and Chen, Qibin and Dong, Yuxiao and Zhang, Jing and Yang, Hongxia and Ding, Ming and Wang, Kuansan and Tang, Jie},
  journal={arXiv preprint arXiv:2006.09963},
  year={2020}
}

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