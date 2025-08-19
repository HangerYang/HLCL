import torch
import numpy as np
from torch_geometric import transforms as T
from torch_geometric.utils import to_undirected, add_self_loops
import scipy as sp
import networkx as nx
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, WikiCS, HeterophilousGraphDataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data
import sys
import scipy
import gdown
import os.path as osp
from typing import Callable, Optional
from torch_geometric.data import Data, InMemoryDataset, download_url
from sklearn.preprocessing import label_binarize
from os import path
import pandas as pd

DATAPATH = "./data/"

class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None
    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
    
dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}


def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
            output=f'{DATAPATH}pokec.mat', quiet=False)
    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')
    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset

def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'data/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset

def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features

def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
            output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
            output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)
    
    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset
def load_fb100(filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    mat = scipy.io.loadmat(DATAPATH + 'facebook100/' + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata

def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    metadata = metadata.astype(int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset
def load_nc_dataset(dataname, sub_dataname=''):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == 'Penn94':
        dataset = load_fb100_dataset(dataname)
    elif dataname == 'pokec':
        dataset = load_pokec_mat()
    # elif dataname in ('ogbn-arxiv', 'ogbn-products'):
    #     dataset = load_ogb_dataset(dataname)
    elif dataname == "genius":
        dataset = load_genius()
    elif dataname == "twitch-gamer":
        dataset = load_twitch_gamer_dataset() 
    else:
        raise ValueError('Invalid dataname')
    print('mission complete')
    return dataset

def dataset_split(file_loc = './data/', dataset_name = 'cora', train_ratio=0.1, val_ratio=0.1, test_ratio=0.8):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=file_loc+dataset_name, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['cornell', 'texas', 'wisconsin']: 
        dataset = WebKB(root=file_loc+dataset_name, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=file_loc+dataset_name, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['actor']:
        dataset = Actor(root=file_loc+dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['WikiCS']:
        dataset = WikiCS(root =file_loc+dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
        dataset = HeterophilousGraphDataset(root =file_loc+dataset_name, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ["Penn94", "twitch-gamer", "pokec", "genius"]:
        dataset = load_nc_dataset(dataset_name)
        data = Data(x=dataset.graph["node_feat"], edge_index=dataset.graph["edge_index"], y=dataset.label, num_features = dataset.graph["node_feat"].shape[1])
        data.edge_index = to_undirected(data.edge_index)
    elif dataset_name in ["chameleon_filtered", "squirrel_filtered"]:
        print("use")
        a = np.load("./data/{}.npz".format(dataset_name))
        data = Data(x = torch.tensor(a["node_features"]), edge_index = torch.tensor(a["edges"].T), y = torch.tensor(a["node_labels"]))
        # data.edge_index = to_undirected(data.edge_index)
        data.train_mask = torch.tensor(a["train_masks"]).t()
        data.val_mask = torch.tensor(a["val_masks"]).t()
        data.test_mask = torch.tensor(a["test_masks"]).t()
    else:
        raise Exception('dataset not available...')
    if dataset_name not in ["Penn94","chameleon_filtered", "squirrel_filtered", "twitch-gamer", "pokec", "genius"] :
        data = dataset[0]
        data.num_classes = dataset.num_classes
    if dataset_name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
        data.edge_index = to_undirected(data.edge_index)
    # data.edge_index, _ = add_self_loops(data.edge_index)
    return data

# def build_graph(dataset,train_ratio=0.1):
#     val_ratio = 0.1
#     test_ratio = 1.0 - val_ratio - train_ratio
#     data = dataset_split(dataset_name= dataset,train_ratio=train_ratio, val_ratio=0.1, test_ratio=test_ratio)
#     data.edge_index, _ = add_self_loops(data.edge_index)
#     g = to_networkx(data, to_undirected=True)
#     Lsym = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(g)
#     Anorm = sp.sparse.identity(np.shape(Lsym)[0]) - Lsym
#     adj = nx.adjacency_matrix(g)

#     data.adj =csr_to_sparse(adj)
#     data.lsym = csr_to_sparse(Lsym)
#     data.anorm = csr_to_sparse(Anorm)
#     return data

# def build_graph_simplified(dataset,train_ratio=0.1):
#     val_ratio = 0.1
#     test_ratio = 1.0 - val_ratio - train_ratio
#     data = dataset_split(dataset_name= dataset,train_ratio=train_ratio, val_ratio=0.1, test_ratio=test_ratio)
#     data.edge_index, _ = add_self_loops(data.edge_index)
#     return data

# def get_attribute(data):
#     g = to_networkx(data, to_undirected=True)
#     Lsym = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(g)
#     Anorm = sp.sparse.identity(np.shape(Lsym)[0]) - Lsym
#     adj = nx.adjacency_matrix(g)

#     data.adj =csr_to_sparse(adj)
#     data.lsym = csr_to_sparse(Lsym)
#     data.anorm = csr_to_sparse(Anorm)
#     return data

# def adj_lap(edge_index, num_nodes, device):
#     edge_index_adj, adj_weight= gcn_norm(edge_index, None, num_nodes, add_self_loops=False)
#     edge_index_lap, lap_weight= get_laplacian(edge_index, None, "sym")
#     shape = num_nodes
#     adj_length = adj_weight.size()[0]
#     lap_length = lap_weight.size()[0]
#     adj = torch.zeros(shape, shape)
#     lap = torch.zeros(shape, shape)

#     for i in range(adj_length):
#         x1 = edge_index_adj[0][i]
#         y1 = edge_index_adj[1][i]
#         adj[x1][y1] = adj_weight[i]
#     for i in range(lap_length): 
#         x2 = edge_index_lap[0][i]
#         y2 = edge_index_lap[1][i]
#         lap[x2][y2] = lap_weight[i]
#     lap = lap.to(device)
#     adj = adj.to(device)
#     return lap, adj

    
# def train_test_split_nodes(data, train_ratio=0.1, val_ratio=0.2, test_ratio=0.2, class_balance=True):
#     r"""Splits nodes into train, val, test masks
#     """
#     n_nodes = data.num_nodes
#     train_mask, ul_train_mask, val_mask, test_mask = torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)
#     total_train_mask, total_val_mask, total_test_mask = [], [], []
#     n_tr = round(n_nodes * train_ratio)
#     n_val = round(n_nodes * val_ratio)
#     n_test = round(n_nodes * test_ratio)

#     train_samples, rest = [], []
#     for i in range(10):
#         if class_balance:
#             unique_cls = list(set(data.y.numpy()))
#             n_cls = len(unique_cls)
#             cls_samples = [n_tr // n_cls + (1 if x < n_tr % n_cls else 0) for x in range(n_cls)]

#             for cls, n_s in zip(unique_cls, cls_samples):
#                 cls_ss = (data.y == cls).nonzero().T.numpy()[0]
#                 cls_ss = np.random.choice(cls_ss, len(cls_ss), replace=False)
#                 train_samples.extend(cls_ss[:n_s])
#                 rest.extend(cls_ss[n_s:])

#             train_mask[train_samples] = 1
#             # assert (sorted(train_samples) == list(train_mask.nonzero().T[0].numpy()))
#             rand_indx = np.random.choice(rest, len(rest), replace=False)
#             # train yet unlabeled
#             ul_train_mask[rand_indx[n_val + n_test:]] = 1

#         else:
#             rand_indx = np.random.choice(np.arange(n_nodes), n_nodes, replace=False)
#             train_mask[rand_indx[n_val + n_test:n_val + n_test + n_tr]] = 1
#             # train yet unlabeled
#             ul_train_mask[rand_indx[n_val + n_test + n_tr:]] = 1

#         val_mask[rand_indx[:n_val]] = 1
#         test_mask[rand_indx[n_val:n_val + n_test]] = 1
#         total_train_mask.append(train_mask.to(torch.bool))
#         total_val_mask.append(val_mask.to(torch.bool))
#         total_test_mask.append(test_mask.to(torch.bool))

#     data.ul_train_mask = ul_train_mask.to(torch.bool)
#     data.train_mask = total_train_mask
#     data.test_mask = total_test_mask
#     data.val_mask = total_val_mask
#     return data