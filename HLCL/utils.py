import random
import numpy as np
import torch
import os
from torch_geometric.utils import (
    coalesce,
    to_dense_adj,dense_to_sparse,to_torch_csr_tensor,to_edge_index,remove_self_loops
)
import torch.nn.functional as F
import argparse
from torch_geometric.loader import ClusterData, ClusterLoader, RandomNodeSampler
import GCL.augmentors as A
from HLCL.augmentation import PPRDiffusion, EdgeAdding

def seed_everything(seed=1234):                                                                                                          
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False
    random.seed(seed) 

def intersect(a,b):
    a = a.t()
    b = b.t()
    a = set((tuple(i) for i in a.cpu().numpy()))
    b = set((tuple(i) for i in b.cpu().numpy()))
    intersect = torch.tensor(list(a.intersection(b)))
    return intersect.t()
def union(a,b):
    un = torch.cat((a,b), dim=1)
    return coalesce(un)

def soft_union(a,b,adj):
    low = F.normalize(a+b)
    high = F.normalize(adj * 2 - low)
    low_edge_index, low_edge_weight = dense_to_sparse(low)
    high_edge_index, high_edge_weight = dense_to_sparse(high)
    return (low_edge_index, high_edge_index), (low_edge_weight, high_edge_weight)
def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask], mask.nonzero().t()[0]



def create_neg_mask(args, datas, device):
    new_datas = []
    for data in datas:
        data.neg_mask = -1
        new_datas.append(data)
    return new_datas

def edge_create(args, data, device):
    with torch.no_grad():
        graph = data.x @ data.x.t()
        # graph.fill_diagonal_(1)
        threshold = min(torch.diagonal(graph)).cpu()
        edge_idx = data.edge_index
        edge_weight = []
        edge_weight = graph[edge_idx[0], edge_idx[1]].tolist()
        node_lst = {}
        weight_lst = {}
        k = len(edge_idx.t())
        for i in range(k):
            if edge_idx[0][i].item() in node_lst:
                node_lst[edge_idx[0][i].item()].append(edge_idx[1][i])
                weight_lst[edge_idx[0][i].item()].append(edge_weight[i])
            else:
                node_lst[edge_idx[0][i].item()] = []
                node_lst[edge_idx[0][i].item()].append(edge_idx[1][i])
                weight_lst[edge_idx[0][i].item()] = []
                weight_lst[edge_idx[0][i].item()].append(edge_weight[i]) 
        low_graph = []
        low_weight = []
        high_graph = []
        high_weight = []
        for node, wgt in weight_lst.items():
            if args.edge == "ratio":
                t_len = len(wgt)
                hk = np.ceil(t_len * args.high_k).astype(int)
                lk = np.ceil(t_len * args.low_k).astype(int)
                wgt = torch.tensor(wgt)
                epsilon = torch.arange(wgt.size(0), device=wgt.device) * 1e-8
                low_edge_wgt, low_edge_idx = torch.topk(wgt+epsilon, lk)
                remained_wgts, remained_idx = th_delete(wgt, low_edge_idx)
                if len(low_edge_wgt) == 0:
                    low_edge_wgt, low_edge_idx = torch.tensor(1), torch.tensor([])
                if len(remained_wgts) > hk:
                    epsilon = torch.arange(remained_wgts.size(0), device=remained_wgts.device) * 1e-8
                    high_edge_wgt, high_edge_idx = torch.topk(remained_wgts+epsilon, hk,largest=False)
                    high_edge_idx = remained_idx[high_edge_idx]
                elif len(remained_wgts) > 0:
                    high_edge_wgt, high_edge_idx = remained_wgts, remained_idx
                else:
                    high_edge_wgt, high_edge_idx = torch.tensor(1), torch.tensor([])
            # else:
            #     wgt = torch.tensor(wgt)
            #     low_edge_idx = torch.nonzero(wgt > threshold * args.threshold).flatten()
            #     low_edge_wgt = wgt[low_edge_idx]
            #     high_edge_wgt, high_edge_idx = th_delete(wgt, low_edge_idx)
            
            for idx, neigh_node in enumerate(high_edge_idx):
                high_graph.append(torch.tensor([node, node_lst[node][neigh_node]]))
                high_graph.append(torch.tensor([node_lst[node][neigh_node], node]))
                high_weight.append(high_edge_wgt[idx])
                high_weight.append(high_edge_wgt[idx])
            for idx, neigh_node in enumerate(low_edge_idx):
                low_graph.append(torch.tensor([node, node_lst[node][neigh_node]]))
                low_graph.append(torch.tensor([node_lst[node][neigh_node], node]))
                low_weight.append(low_edge_wgt[idx])
                low_weight.append(low_edge_wgt[idx])
            if len(high_edge_idx) == 0:
                random_edge = random.choice(low_edge_idx)
                high_graph.append(torch.tensor([node, node_lst[node][random_edge]]))
                high_graph.append(torch.tensor([node_lst[node][random_edge], node]))
                high_weight.append(high_edge_wgt)
                high_weight.append(high_edge_wgt)
        data.high_edge_index = torch.stack(high_graph).t().to(device)
        data.low_edge_index = torch.stack(low_graph).t().to(device)
        data.high_edge_weight = torch.ones(data.high_edge_index.shape[1]).to(device)
        data.low_edge_weight = torch.ones(data.low_edge_index.shape[1]).to(device)
    return data

def edge_create_(args, x, edge_idx, device=None):
    with torch.no_grad():
        graph = x @ x.t()
        graph.fill_diagonal_(1)
        threshold = min(torch.diagonal(graph)).cpu()
        edge_weight = []
        edge_weight = graph[edge_idx[0], edge_idx[1]].tolist()
        node_lst = {}
        weight_lst = {}
        k = len(edge_idx.t())
        for i in range(k):
            if edge_idx[0][i].item() in node_lst:
                node_lst[edge_idx[0][i].item()].append(edge_idx[1][i])
                weight_lst[edge_idx[0][i].item()].append(edge_weight[i])
            else:
                node_lst[edge_idx[0][i].item()] = []
                node_lst[edge_idx[0][i].item()].append(edge_idx[1][i])
                weight_lst[edge_idx[0][i].item()] = []
                weight_lst[edge_idx[0][i].item()].append(edge_weight[i]) 
        low_graph = []
        low_weight = []
        high_graph = []
        high_weight = []
        for node, wgt in weight_lst.items():
            if args.edge == "ratio":
                t_len = len(wgt)
                hk = max(1, np.ceil(t_len * args.high_k).astype(int))
                lk = max(1, np.ceil(t_len * args.low_k).astype(int))
                wgt = torch.tensor(wgt)
                epsilon = torch.arange(wgt.size(0), device=wgt.device) * 1e-8
                low_edge_wgt, low_edge_idx = torch.topk(wgt+epsilon, lk)
                remained_wgts, remained_idx = th_delete(wgt, low_edge_idx)
                if len(remained_wgts) > hk:
                    epsilon = torch.arange(remained_wgts.size(0), device=remained_wgts.device) * 1e-8
                    high_edge_wgt, high_edge_idx = torch.topk(remained_wgts+epsilon, hk,largest=False)
                    high_edge_idx = remained_idx[high_edge_idx]
                elif len(remained_wgts) > 0:
                    high_edge_wgt, high_edge_idx = remained_wgts, remained_idx
                else:
                    high_edge_wgt, high_edge_idx = torch.tensor(1), torch.tensor([])
                
            for idx, neigh_node in enumerate(high_edge_idx):
                high_graph.append(torch.tensor([node, node_lst[node][neigh_node]]))
                high_graph.append(torch.tensor([node_lst[node][neigh_node], node]))
                high_weight.append(high_edge_wgt[idx])
                high_weight.append(high_edge_wgt[idx])
            for idx, neigh_node in enumerate(low_edge_idx):
                low_graph.append(torch.tensor([node, node_lst[node][neigh_node]]))
                low_graph.append(torch.tensor([node_lst[node][neigh_node], node]))
                low_weight.append(low_edge_wgt[idx])
                low_weight.append(low_edge_wgt[idx])
            if len(high_edge_idx) == 0:
                random_edge = random.choice(low_edge_idx)
                high_graph.append(torch.tensor([node, node_lst[node][random_edge]]))
                high_graph.append(torch.tensor([node_lst[node][random_edge], node]))
                high_weight.append(high_edge_wgt)
                high_weight.append(high_edge_wgt)
        
        high_edge_index = torch.stack(high_graph).t().to(device)
        low_edge_index = torch.stack(low_graph).t().to(device)
        low_edge_weight = torch.ones(low_edge_index.shape[1]).to(device)
        high_edge_weight = torch.ones(high_edge_index.shape[1]).to(device)
    return low_edge_index, high_edge_index, low_edge_weight, high_edge_weight
    
def two_hop(data):
    N = data.num_nodes
    adj = to_torch_csr_tensor(data.edge_index, size=(N, N))
    edge_index2, _ = to_edge_index(adj @ adj)
    edge_index2, _ = remove_self_loops(edge_index2)
    edge_index = torch.cat([data.edge_index, edge_index2], dim=1)
    data.edge_index = edge_index
    return data

def new_edges(args, x):
    num_nodes, _ = x.shape
    low_graph = []
    low_weight = []
    for i in range(num_nodes):
        low_edge_wgt, low_edge_idx = torch.topk(x[i], args.add_edge)
        for idx, j in enumerate(low_edge_idx):
            edge_pair = torch.tensor([i,j])
            low_graph.append(edge_pair)
            low_weight.append(low_edge_wgt[idx])
    return low_graph, low_edge_wgt

def representation_combine_supervised(args, z1, z2, zs1, zs2):
    if args.combine_x:
        return torch.cat((z1,z2),dim=1), z1, z2, (zs1+zs2)
    else:
        return z1, z1, z2, (zs1+zs2)
    
def res_combine_supervised(args, device, edge_index, low_k=None, high_k=None, low_x=None, high_x=None, low_zs=None, high_zs=None):
    with torch.no_grad():
        ret = representation_combine_supervised(args, low_x, high_x, low_zs, high_zs)
        low_edges_0,low_edges_1, low_edge_weight_0, low_edge_weight_1 = edge_create(args, low_x, edge_index, high_k, low_k, device)
        high_edges_0,high_edges_1, high_edge_weight_0, high_edge_weight_1 = edge_create(args, high_x, edge_index, high_k, low_k, device)
        low_pass_edges = intersect(low_edges_0, high_edges_0)
        high_pass_edges = intersect(low_edges_1, high_edges_1)
        low_edge_weight = torch.ones(low_pass_edges.shape[1])
        high_edge_weight = torch.ones(high_pass_edges.shape[1])
    return ret, low_pass_edges, high_pass_edges, low_edge_weight, high_edge_weight

def res_combine(args, device, edge_index, low_k=None, high_k=None, low_x=None, high_x=None):
    with torch.no_grad():
        ret = representation_combine(args, low_x, high_x)
        low_edges_0,low_edges_1, low_edge_weight_0, low_edge_weight_1 = edge_create_(args, low_x, edge_index, device)
        high_edges_0,high_edges_1, high_edge_weight_0, high_edge_weight_1 = edge_create_(args, high_x, edge_index, device)
        low_pass_edges = intersect(low_edges_0, high_edges_0)
        high_pass_edges = intersect(low_edges_1, high_edges_1)
        low_edge_weight = torch.ones(low_pass_edges.shape[1])
        high_edge_weight = torch.ones(high_pass_edges.shape[1])
    return ret, low_pass_edges, high_pass_edges, low_edge_weight, high_edge_weight

        

def representation_combine(args, z1, z2, z3, z4):
    if args.combine_x:
        return torch.cat((z1,z3),dim=1), z1, z2, z3, z4
    else:
        return z1, z1, z2, z3, z4
    


def split_edges(edge_index, split):
    unknown_edges = []
    known_edges = []
    for i in edge_index.T:
        if i[0] not in split["train"] or i[1] not in split["train"]:
            unknown_edges.append(i)
        elif i[0] == i[1]:
            unknown_edges.append(i)
            known_edges.append(i)
        else:
            known_edges.append(i)
    return torch.stack(unknown_edges), torch.stack(known_edges)



def make_loader(args, dataset, idx, mini_batch=True, device=torch.device('cpu'), test=False):
    if not mini_batch:
        loader = RandomNodeSampler(dataset, num_parts=1, shuffle=True, num_workers=0)
        return loader
        
    if args.train_batch == 'cluster':
        cluster_data = ClusterData(dataset, num_parts=args.num_parts)
        loader = ClusterLoader(cluster_data, batch_size=args.cluster_batch_size, shuffle=True, num_workers=0)
def aug_choice(args):
    if args.augmentation == "EdgeRemoving":
        return (A.EdgeRemoving(pe=args.haug1), A.EdgeRemoving(pe=args.haug1))
    elif args.augmentation == "FeatureMasking":
        return (A.FeatureMasking(pf=args.haug1), A.FeatureMasking(pf=args.haug1))
    elif args.augmentation == "NodeDropping":
        return (A.NodeDropping(pn=args.haug1), A.NodeDropping(pn=args.haug1))
    elif args.augmentation == "EdgeAdding":
        return (EdgeAdding(pe=args.haug1), EdgeAdding(pe=args.haug1))
    elif args.augmentation == "PPRDiffusion":
        return (PPRDiffusion(), PPRDiffusion())
    else:
        return (A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)]), A.Compose([A.NodeDropping(pn=args.haug1), A.FeatureMasking(pf=args.haug2)]))

def get_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = "cora", help='benchmark dataset : cora, citeseer, pubmed')
    parser.add_argument('--pre_eval', type=int, default = 50, help='number per epoch to evaluate GCL')
    parser.add_argument('--hidden', type=int, default = 512, help='gcn, gat, fbgcn')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--preepochs' ,type=int, default = 500, help='pretraining epoch')
    parser.add_argument('--per_epoch' ,type=int, default = 50, help='graph update frequency')
    parser.add_argument('--pre_learning_rate',type=float, default = 0.001, help='pre training learning rate')
    parser.add_argument('--aug1', type=float, default = 0.2, help='aug parameter')
    parser.add_argument('--aug2', type=float, default = 0.2, help='aug parameter')
    parser.add_argument('--augmentation', type=str, default = "simple")
    parser.add_argument('--runs', type=int, default=10, help='number of distinct runs')
    parser.add_argument('--num_layer', type=int, default=2, help='number of layers')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--low_k', type=float, default=0.2)
    parser.add_argument('--high_k', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--combine_x', action='store_true')
    parser.add_argument('--infer_combine_x', action='store_false')
    parser.add_argument('--split', type=str, default = "simple")
    parser.add_argument('--cluster_batch_size', type=int, default = 1)
    parser.add_argument('--num_parts', type=int, default = 1)
    parser.add_argument('--eval', type=str, default = "acc")
    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.8)


    args, unknown = parser.parse_known_args()
    return args

