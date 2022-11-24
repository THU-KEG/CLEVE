# CLEVE Structure Pre-training

This folder contains structure pre-training codes. Codes here are copied from [GCC](https://github.com/THUDM/GCC) with minor modifications, which will be illustrated below. As stated in our paper, this part is running [GCC](https://github.com/THUDM/GCC) with our inputs.

## Prerequisites

Same as [GCC](https://github.com/THUDM/GCC). Please ensure you can run pre-training codes in [GCC](https://github.com/THUDM/GCC).

## Pre-training
* Follow the README in the main folder and get ```[nyt_parsed_file]```.
* Put ```glove.6B.300d.txt``` to this folder.
* Convert AMR file to the DGL format ```python load_AMR.py --amr_file [nyt_parsed_file]```, and the converted file will be in ```GCC/data```

> There are two load_AMR.py in this repo, one is in the main folder, and one is in this folder. Here we mean the load_AMR.py in this folder.

* Go to ```GCC``` folder, and run pre-training: ```bash scripts/pretrain.sh <gpu> --moco --nce-k 10800```

> You may want to adjust parameters (defined in ```train.py```) based on the concrete data. For example, the ```nce-k``` parameter is the queue size of MoCo, which needs to be small compared to the **data size (totale node numbers)**, but in the meanwhile be as large as possible to maintain a good contrastive training performance. ```num-samples``` is similar. 6% is a good portion. Reasons can be found [here](https://github.com/facebookresearch/moco/issues/24#issuecomment-631233654). ```rw-hops``` implicitly controls the size of subgraphs, which should be small compared to the average **graph size (average node numbers)**.

## Down-stream Usage
Dumped parameters can be loaded into the model ```GCC/gcc/models/graph_encoder.py```. The output ```x``` is graph representations, ```node_reps``` is node representations. The pre-trained model can be used to AMR-knowledge-related tasks. More details for down-stream usage can be found in [GCC](https://github.com/THUDM/GCC).

## Main difference between the GCC and this codebase
* Our input files contain more information. Our DGL file contains token semantic information and edge type information. Core codes: ```load_AMR.py```.
* Nodes initialization and edges initialiazation. We add semantic embedding (glove embedding) for node features initialization. We also use random initialization based on edge types to initialize edge features. Original GCC codebase ignores the semantic meaning of nodes, and does not implement edge features. Core codes: ```GCC/gcc/models/graph_encoder.py, GCC/train.py```. 

> Note 1: Glove embeddings can be replaced by contextual representations such as BERT and RoBERTa.

> Note 2: Models in the codebase (e.g., GIN) may not use edge features. However, we implement edge features in case other models may need them.
  
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