import re
from collections import defaultdict
import argparse
import pickle

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


def process_to_pretrain(gs):
    examples = []
    for g in gs:
        example = {}
        tokens = g.tokens
        edges = g.edges
        example['tokens'] = tokens

        nodes = list(set(g.node_id2idx.values()))
        example['positive_edges'] = {}
        example['negative_edges'] = {}
        useful_nodes = []
        for k,vs in edges.items():
            if type(k)!=tuple or k in g.abandon_wordidx:
                continue
            positive_nodes = [v[1] for v in vs if ((v[0].lower().startswith('arg') or v[0].lower() in ['time','year','duration','decade','weekday','location','path','destination']) and type(v[1])==tuple and (v[1] not in g.abandon_wordidx))]
            if len(positive_nodes)==0:
                continue
            example['positive_edges'][k] = positive_nodes
            useful_nodes.extend(positive_nodes)
        useful_nodes = list(set(useful_nodes))
        for k in edges:
            try:
                example['negative_edges'][k] = minus_list(useful_nodes,example['positive_edges'][k]+[k] + g.abandon_wordidx)
            except:
                pass
        if len(example['positive_edges'])>1:
            examples.append(example)

    return examples


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


    gs = load_amr(
        args.amr_file
    )
    examples = process_to_pretrain(gs)
    
    with open('contrast_examples.pkl','wb') as f:
        pickle.dump(examples,f)