import torch
import numpy as np
from torch_geometric import transforms as T
from torch_geometric.utils import to_networkx, add_self_loops, get_laplacian
import scipy as sp
import networkx as nx
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, WikiCS
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data
import sys

import os.path as osp
from typing import Callable, Optional
from torch_geometric.data import Data, InMemoryDataset, download_url


sys.path.append("/home/hyang/HL_Contrast/Non-Homophily-Large-Scale/")
from dataset import load_nc_dataset
def csr_to_sparse(csr):
    csr = csr.tocoo().astype(np.float32)
    indices = torch.from_numpy(
    np.vstack(
        (csr.row, csr.col)).astype(np.int64)
    )
    values = torch.from_numpy(csr.data)
    shape = torch.Size(csr.shape)
    sparse = torch.sparse.FloatTensor(indices, values, shape)
    return sparse

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
    elif dataset_name == "penn94":
        dataset = load_nc_dataset('fb100', "Penn94")
        data = Data(x=dataset.graph["node_feat"], edge_index=dataset.graph["edge_index"], y=dataset.label, num_features = dataset.graph["node_feat"].shape[1])
    elif dataset_name in ["chameleon_filtered", "squirrel_filtered"]:
        a = np.load("/home/hyang/HL_Contrast/data/{}.npz".format(dataset_name))
        data = Data(x = torch.tensor(a["node_features"]), edge_index = torch.tensor(a["edges"].T), y = torch.tensor(a["node_labels"]))
        data.train_mask = torch.tensor(a["train_masks"]).t()
        data.val_mask = torch.tensor(a["val_masks"]).t()
        data.test_mask = torch.tensor(a["test_masks"]).t()
    else:
        raise Exception('dataset not available...')
    if dataset_name not in ["penn94","chameleon_filtered", "squirrel_filtered"] :
        data = dataset[0]
    # if dataset_name in ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'actor', 'WikiCS']:
    #     data.train_mask = torch.swapaxes(data.train_mask, 0, 1)
    #     data.val_mask = torch.swapaxes(data.val_mask, 0, 1)
    #     try:
    #         data.test_mask = torch.swapaxes(data.test_mask, 0, 1)
    #     except:
    #         data.test_mask = np.repeat(data.test_mask[np.newaxis], 10, axis = 0)
    # else:
        # data = train_test_split_nodes(data, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        data.num_classes = dataset.num_classes
    return data

def build_graph(dataset,train_ratio=0.1):
    val_ratio = 0.1
    test_ratio = 1.0 - val_ratio - train_ratio
    data = dataset_split(dataset_name= dataset,train_ratio=train_ratio, val_ratio=0.1, test_ratio=test_ratio)
    data.edge_index, _ = add_self_loops(data.edge_index)
    g = to_networkx(data, to_undirected=True)
    Lsym = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(g)
    Anorm = sp.sparse.identity(np.shape(Lsym)[0]) - Lsym
    adj = nx.adjacency_matrix(g)

    data.adj =csr_to_sparse(adj)
    data.lsym = csr_to_sparse(Lsym)
    data.anorm = csr_to_sparse(Anorm)
    return data

def build_graph_simplified(dataset,train_ratio=0.1):
    val_ratio = 0.1
    test_ratio = 1.0 - val_ratio - train_ratio
    data = dataset_split(dataset_name= dataset,train_ratio=train_ratio, val_ratio=0.1, test_ratio=test_ratio)
    data.edge_index, _ = add_self_loops(data.edge_index)
    return data

def get_attribute(data):
    g = to_networkx(data, to_undirected=True)
    Lsym = nx.linalg.laplacianmatrix.normalized_laplacian_matrix(g)
    Anorm = sp.sparse.identity(np.shape(Lsym)[0]) - Lsym
    adj = nx.adjacency_matrix(g)

    data.adj =csr_to_sparse(adj)
    data.lsym = csr_to_sparse(Lsym)
    data.anorm = csr_to_sparse(Anorm)
    return data

def adj_lap(edge_index, num_nodes, device):
    edge_index_adj, adj_weight= gcn_norm(edge_index, None, num_nodes, add_self_loops=False)
    edge_index_lap, lap_weight= get_laplacian(edge_index, None, "sym")
    shape = num_nodes
    adj_length = adj_weight.size()[0]
    lap_length = lap_weight.size()[0]
    adj = torch.zeros(shape, shape)
    lap = torch.zeros(shape, shape)

    for i in range(adj_length):
        x1 = edge_index_adj[0][i]
        y1 = edge_index_adj[1][i]
        adj[x1][y1] = adj_weight[i]
    for i in range(lap_length): 
        x2 = edge_index_lap[0][i]
        y2 = edge_index_lap[1][i]
        lap[x2][y2] = lap_weight[i]
    lap = lap.to(device)
    adj = adj.to(device)
    return lap, adj

    
def train_test_split_nodes(data, train_ratio=0.1, val_ratio=0.2, test_ratio=0.2, class_balance=True):
    r"""Splits nodes into train, val, test masks
    """
    n_nodes = data.num_nodes
    train_mask, ul_train_mask, val_mask, test_mask = torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)
    total_train_mask, total_val_mask, total_test_mask = [], [], []
    n_tr = round(n_nodes * train_ratio)
    n_val = round(n_nodes * val_ratio)
    n_test = round(n_nodes * test_ratio)

    train_samples, rest = [], []
    for i in range(10):
        if class_balance:
            unique_cls = list(set(data.y.numpy()))
            n_cls = len(unique_cls)
            cls_samples = [n_tr // n_cls + (1 if x < n_tr % n_cls else 0) for x in range(n_cls)]

            for cls, n_s in zip(unique_cls, cls_samples):
                cls_ss = (data.y == cls).nonzero().T.numpy()[0]
                cls_ss = np.random.choice(cls_ss, len(cls_ss), replace=False)
                train_samples.extend(cls_ss[:n_s])
                rest.extend(cls_ss[n_s:])

            train_mask[train_samples] = 1
            # assert (sorted(train_samples) == list(train_mask.nonzero().T[0].numpy()))
            rand_indx = np.random.choice(rest, len(rest), replace=False)
            # train yet unlabeled
            ul_train_mask[rand_indx[n_val + n_test:]] = 1

        else:
            rand_indx = np.random.choice(np.arange(n_nodes), n_nodes, replace=False)
            train_mask[rand_indx[n_val + n_test:n_val + n_test + n_tr]] = 1
            # train yet unlabeled
            ul_train_mask[rand_indx[n_val + n_test + n_tr:]] = 1

        val_mask[rand_indx[:n_val]] = 1
        test_mask[rand_indx[n_val:n_val + n_test]] = 1
        total_train_mask.append(train_mask.to(torch.bool))
        total_val_mask.append(val_mask.to(torch.bool))
        total_test_mask.append(test_mask.to(torch.bool))

    data.ul_train_mask = ul_train_mask.to(torch.bool)
    data.train_mask = total_train_mask
    data.test_mask = total_test_mask
    data.val_mask = total_val_mask
    return data




class HeterophilousGraphDataset(InMemoryDataset):
    url = ('https://github.com/yandex-research/heterophilous-graphs/raw/'
           'main/data')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower().replace('-', '_')
        assert self.name in [
            'roman_empire',
            'amazon_ratings',
            'minesweeper',
            'tolokers',
            'questions',
        ]

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)

    def process(self):
        raw = np.load(self.raw_paths[0], 'r')
        x = torch.from_numpy(raw['node_features'])
        y = torch.from_numpy(raw['node_labels'])
        edge_index = torch.from_numpy(raw['edges']).t().contiguous()
        train_mask = torch.from_numpy(raw['train_masks']).t().contiguous()
        val_mask = torch.from_numpy(raw['val_masks']).t().contiguous()
        test_mask = torch.from_numpy(raw['test_masks']).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'