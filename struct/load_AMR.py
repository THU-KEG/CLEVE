import dgl
from dgl.data.utils import save_graphs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re
from collections import defaultdict
import argparse
import pickle
import numpy as np
from tqdm import tqdm

class Graph():
    def __init__(self,raw_nodes,raw_edges,tokens):
        self.node_id2idx,self.abandon_wordidx = self.process_nodes(raw_nodes) #{id:(st,ed)}
        self.edges = self.process_edges(raw_edges)       #{(st,ed):[(type,(st,ed)),....]}
        self.tokens = tokens
        self.process_graph(re.compile(r'^op\d+'))
        self.process_graph(re.compile(r'name'))

    def process_nodes(self,nodes):
        node_id2idx = {}
        abandon_wordidx = []
        proper_noun = re.compile(r'"(\S+)"')
        for node in nodes:
            node_id2idx[node[0]] = (int(node[2]),int(node[3]))
            if node[1] in ['and','or','not'] or (proper_noun.match(node[1]) is not None):   # Remove concat words and proper nouns from pretraining
                abandon_wordidx.append((int(node[2]),int(node[3])))
        return node_id2idx,abandon_wordidx
    
    def process_edges(self,raw_edges):
        edges = defaultdict(list)
        for raw_edge in raw_edges:
            node1,node2 = raw_edge[3],raw_edge[4]
            node1idx = self.node_id2idx[node1] if node1 in self.node_id2idx else node1
            node2idx = self.node_id2idx[node2] if node2 in self.node_id2idx else node2
            if node1idx==node2idx:    #remove self circle. This will also remove edges like:   country ---:name---> name ---:op---> U.S. if they are same word
                continue
            if raw_edge[1].endswith('-of'):   # reverse edge
                edges[node2idx].append((raw_edge[1][:-3],node1idx))
            else:
                edges[node1idx].append((raw_edge[1],node2idx))
        return edges
    
    def process_graph(self,rule):
        edges = self.edges
        edges_tuple = [(k,v[0],v[1]) for k,vs in edges.items() for v in vs]
        new_op_edges_tuple = []
        for edge_tuple in edges_tuple:
            node1idx = edge_tuple[0]
            edge_rel = edge_tuple[1]
            node2idx = edge_tuple[2]
            if rule.match(edge_rel) is not None:                # Merge all op/name nodes to its parent nodes
                new_edges_tuple = [(e_t[0],e_t[1],node2idx) for e_t in edges_tuple if e_t[2]==node1idx]
                new_op_edges_tuple.extend(new_edges_tuple)
        
        for new_edges_tuple in new_op_edges_tuple:
            n1,rel,n2 = new_edges_tuple
            self.edges[n1].append((rel,n2))


def load_amr(filename):
    with open(filename,'r') as f:
        amrs = f.read()
    
    amrs = amrs.split('\n\n')
    amrs = amrs[:-1] if amrs[-1]=="" else amrs
    node_format = re.compile(r'# ::node\t(\S+)\t(\S+)\t(\d+)-(\d+)')
    edge_format = re.compile(r'# ::edge\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)')
    gs = []
    for amr in amrs:
        tokens = amr.split('\n')[2]
        assert tokens.startswith('# ::tok ')
        tokens = tokens[len("# ::tok "):].split(' ')
        nodes = node_format.findall(amr)
        edges = edge_format.findall(amr)
        graph = Graph(nodes,edges,tokens)
        gs.append(graph)
    return gs


def minus_list(list1,list2):
    rax = []
    for a in list1:
        if a not in list2:
            rax.append(a)
    return rax


def all_edge_types(gs):
    rs = []
    for g in gs:
        for u, r_vs in g.edges.items():
            for r, v in r_vs:
                rs.append(r)
    rs = list(set(rs))
    return rs


def load_embedding(glove_path, embedding_dim):
    word2idx = {}
    wordemb = []
    with open(glove_path,'r',encoding='utf-8') as f:
        for line in f:
            splt = line.split()
            assert len(splt)==embedding_dim + 1
            vector = list(map(float, splt[-embedding_dim:]))
            word = splt[0].lower()
            word2idx[word] = len(word2idx)+2
            wordemb.append(vector)
    return word2idx,np.asarray(wordemb, np.float32)


def process_to_dgl(g, word2idx, edge_types2id):
    _node_collection = list(set(list(g.node_id2idx.values())))
    # Remove nodes that have no edges
    node_collection = []
    for node in _node_collection:
        if node in g.edges and isinstance(node, tuple):
            num_edge = sum([1 for e in g.edges[node] if isinstance(e[1],tuple)])
            if num_edge>0:
                node_collection.append(node)
    node2idx = {node:idx for idx, node in enumerate(node_collection)}
    if len(node_collection) == 0:
        return None, None
    dgl_g = dgl.DGLGraph()
    have_empty_node = False
    # Init nodes
    dgl_g.add_nodes(len(node_collection))
    # Add edges
    for u, r_vs in g.edges.items():
        if u not in node2idx:
            have_empty_node = True
            continue
        uid = node2idx[u]
        for r, v in r_vs:
            if v not in node2idx:
                have_empty_node = True
                continue
            vid = node2idx[v]
            dgl_g.add_edges([uid], [vid], {'type': torch.tensor([edge_types2id[r]])})
            dgl_g.add_edges([vid], [uid], {'type': torch.tensor([edge_types2id[r]])})
    # Node token id
    node_token_ids = []
    max_token_len = 30
    for node in node_collection:
        assert isinstance(node,tuple)
        node_st, node_ed = node
        _token_ids = [word2idx[g.tokens[node_posi].lower()] if g.tokens[node_posi].lower() in word2idx else 1 for node_posi in range(node_st, node_ed)]
        if len(_token_ids)<max_token_len:
            _token_ids += [0] * (max_token_len-len(_token_ids))
        else:
            _token_ids = _token_ids[:max_token_len]
        node_token_ids.append(_token_ids)
    dgl_g.ndata['token_ids'] = torch.tensor(node_token_ids, dtype=torch.int32)
    return dgl_g, have_empty_node


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amr_file",
        default=None,
        type=str,
        required=True,
        help="The input amr file.",
    )
    args = parser.parse_args()

    print("Loading AMR file....")
    gs = load_amr(
        args.amr_file
    )

    print("Loading Glove....")
    word2idx, wordemb = load_embedding('./glove.6B.300d.txt', 300)

    print("Processing graphs...")
    edge_types = all_edge_types(gs)
    print("Total edge types:", len(edge_types))
    edge_types2id = {e:i for i,e in enumerate(edge_types)}
    sizes = []
    dgl_gs = []
    num_graphs_w_empty_nodes = 0
    tot = 0
    for g in tqdm(gs):
        dgl_g, have_empty_node = process_to_dgl(g, word2idx, edge_types2id)
        if dgl_g is None:
            continue
        size = dgl_g.number_of_nodes()
        dgl_gs.append(dgl_g)
        sizes.append(size)
        tot += 1
        if have_empty_node:
            num_graphs_w_empty_nodes += 1
    print("{} of {} graphs have empty nodes".format(num_graphs_w_empty_nodes, tot))

    print("---Sample----")
    print("#Nodes:", dgl_gs[0].number_of_nodes())
    print("Nodes:", dgl_gs[0].nodes())
    print("#Edges:",dgl_gs[0].number_of_edges())
    print("Edges:",dgl_gs[0].edges())
    print("Nodes Data:", dgl_gs[0].ndata)
    print("Edge Data:", dgl_gs[0].edata)

    print("Dumping graphs...")
    save_graphs("./GCC/data/small.bin", dgl_gs, {"graph_sizes": torch.tensor(sizes)})