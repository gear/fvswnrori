import os
import torch
import numpy as np
import pickle as pkl


def load_adj_data(config):
    """Load and return sparse adjacency matrix representation for a dataset. 
    Other properties such as features are the same as other load methods.
    """
    dataset_name = config['dataset']
    dataset_root = config['dataset_root']
    device = torch.device(config['device'])

    graph_file = os.path.join(dataset_root, dataset_name)
    with open(graph_file+'.graph', 'rb') as f:
        nx_graph = pkl.load(f)
    with open(graph_file+'.X', 'rb') as f:
        features = pkl.load(f)
    with open(graph_file+'.y', 'rb') as f:
        labels = pkl.load(f)
    with open(graph_file+'.split', 'rb') as f:
        splits = pkl.load(f)

    edge_list = list(nx_graph.edges())
    if not nx_graph.is_directed():
        edge_list.extend([(j,i) for (i,j) in el])

    data = torch.ones(len(edge_list))
    idx = torch.LongTensor([*zip(*edge_list)])
    shape = (nx_graph.number_of_nodes(), nx_graph.number_of_nodes())
    adj = torch.sparse.FloatTensor(idx, data, torch.Size(shape))

