import re
from collections import defaultdict
import argparse
import pickle

class Graph():
    def __init__(self,raw_nodes,raw_edges,tokens):
        self.node_id2idx = self.process_nodes(raw_nodes) #{id:(st,ed)}
        self.edges = self.process_edges(raw_edges)       #{(st,ed):[(type,(st,ed)),....]}
        self.tokens = tokens

    def process_nodes(self,nodes):
        node_id2idx = {}
        for node in nodes:
            node_id2idx[node[0]] = (node[2],node[3])
        return node_id2idx
    
    def process_edges(self,raw_edges):
        edges = defaultdict(list)
        for raw_edge in raw_edges:
            node1,node2 = raw_edge[3],raw_edge[4]
            node1idx,node2idx =  self.node_id2idx[node1],self.node_id2idx[node2]
            if node1idx==node2idx:    #remove self circle
                continue
            if raw_edge[1].endswith('-of'):
                edges[node2idx].append((raw_edge[1][:-3],node1idx))
            else:
                edges[node1idx].append((raw_edge[1],node2idx))
        return edges

def load_amr(filename):
    with open(filename,'r') as f:
        amrs = f.read()
    
    amrs = amrs.split('\n\n')
    amrs = amrs[:-1] if amrs[-1]=="" else amrs
    node_format = re.compile(r'# ::node\t(\S+)\t(\S+)\t(\d+)-(\d+)')
    edge_format = re.compile(r'# ::edge\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)')
    gs = []
    for amr in amrs:
        tokens = amr.split('\n')[1]
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
        for k,vs in edges.items():
            positive_nodes = [v[1] for v in vs if v[0].lower().startswith('arg') or v[0].lower() in ['time','location']]
            if len(positive_nodes)==0:
                continue
            example['positive_edges'][k] = positive_nodes
            example['negative_edges'][k] = minus_list(nodes,positive_nodes+[k])
        if len(example['positive_edges'])!=0:
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