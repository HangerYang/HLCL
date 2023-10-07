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

def create_neg_mask(args, device, low_graph=None, high_graph=None, x=None):
    if args.neg == "inverse_low":
        a = to_dense_adj(low_graph)
        a[a==0] = -1
        a[a>0] = 0
        a = a * -1
        a = a[0].cpu()
    elif args.neg == "high":
        a = to_dense_adj(high_graph)
        a = a[0].cpu()
    elif args.neg == "simple":
        a = None
    elif args.neg == "global":
        graph = x @ x.t()
        graph[graph==0] = 10
        indices = graph.topk(k=300, largest=False)[1]
        a = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
        a = a.cpu()
    return a

def create_neg_mask_cuda(args, device, low_graph=None, high_graph=None, x=None):
    if args.neg == "inverse_low":
        a = to_dense_adj(low_graph)
        a[a==0] = -1
        a[a>0] = 0
        a = a * -1
        a = a[0].to(device)
    elif args.neg == "high":
        a = to_dense_adj(high_graph)
        a = a[0].to(device)
    elif args.neg == "simple":
        a = None
    elif args.neg == "global":
        graph = x @ x.t()
        graph[graph==0] = 10
        indices = graph.topk(k=300, largest=False)[1]
        a = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
        a = a.to(device)
    return a

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
def add_edge(edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)

    return coalesce(edge_index)


def create_neg_mask_cuda_updated(args, datas, device):
    new_datas = []
    for data in datas:
        if args.neg == "inverse_low":
            a = to_dense_adj(data.low_edge_index)
            a[a==0] = -1
            a[a>0] = 0
            a = a * -1
            data.neg_mask = a[0].to(device)
        elif args.neg == "high":
            a = to_dense_adj(data.high_edge_index)
            data.neg_mask = a[0].to(device)
        elif args.neg == "simple":
            data.neg_mask = -1
        elif args.neg == "global":
            graph = data.x @ data.x.t()
            graph[graph==0] = 10
            indices = graph.topk(k=300, largest=False)[1]
            data.neg_mask = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
            data.neg_mask = data.neg_mask.to(device)
        new_datas.append(data)
    return new_datas
def create_neg_mask_updated(args, datas, device):
    new_datas = []
    for data in datas:
        if args.neg == "inverse_low":
            a = to_dense_adj(data.low_edge_index)
            a[a==0] = -1
            a[a>0] = 0
            a = a * -1
            data.neg_mask = a[0]
        elif args.neg == "high":
            a = to_dense_adj(data.high_edge_index)
            data.neg_mask = a[0]
        elif args.neg == "simple":
            data.neg_mask = -1
        elif args.neg == "global":
            graph = data.x @ data.x.t()
            graph[graph==0] = 10
            indices = graph.topk(k=300, largest=False)[1]
            data.neg_mask = torch.zeros(graph.shape[0], graph.shape[0]).scatter_(1, indices, 1)
            data.neg_mask = data.neg_mask
        new_datas.append(data)
    return new_datas

def edge_create_updated(args, data, device):
    with torch.no_grad():
        graph = data.x @ data.x.t()
        adj_idx = to_dense_adj(data.edge_index)[0]
        graph = graph*adj_idx
        edge_idx, edge_weight = dense_to_sparse(graph)
        node_lst = {}
        weight_lst = {}
        k = len(edge_idx.t())
        for i in range(k):
            if i > 0 and edge_idx[0][i] == edge_idx[0][i-1]:
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
            t_len = len(wgt)
            hk = np.ceil(t_len * args.high_k).astype(int)
            lk = np.ceil(t_len * args.low_k).astype(int)
            wgt = torch.tensor(wgt)
            low_edge_wgt, low_edge_idx = torch.topk(wgt, lk)
            remained_wgts, remained_idx = th_delete(wgt, low_edge_idx)
            
            if len(remained_wgts) > hk:
                high_edge_wgt, high_edge_idx = torch.topk(remained_wgts, hk,largest=False)
                high_edge_idx = remained_idx[high_edge_idx]
            else:
                high_edge_wgt, high_edge_idx = remained_wgts, remained_idx
            for idx, neigh_node in enumerate(high_edge_idx):
                edge_pair = torch.tensor([node, node_lst[node][neigh_node]])
                high_graph.append(edge_pair)
                high_weight.append(high_edge_wgt[idx])
            for idx, neigh_node in enumerate(low_edge_idx):
                edge_pair = torch.tensor([node, node_lst[node][neigh_node]])
                low_graph.append(edge_pair)
                low_weight.append(low_edge_wgt[idx])
        data.high_edge_index = torch.stack(high_graph).t().to(device)
        data.low_edge_index = torch.stack(low_graph).t().to(device)
        data.high_edge_weight = torch.ones(data.high_edge_index.shape[1]).to(device)
        data.low_edge_weight = torch.ones(data.low_edge_index.shape[1]).to(device)
    return data
                # if args.edge == "hard_num":
        #     graph = x @ x.t()
        #     adj_idx = to_dense_adj(edge_index)[0]
        #     graph = graph*adj_idx
        #     indices = graph.topk(k=int(low_k))[1]
        #     indices = [indices[i][graph[i][indices[i]] > 0] if (graph[i][indices[i]] > 0).sum() > 0 else torch.tensor([i]) for i in range(indices.shape[0])]
        #     max_length = max([len(l) for l in indices])
        #     indices = [torch.cat((l, l[-1].repeat(max_length - len(l)))).to(device) for l in indices]
        #     indices = torch.stack(indices)
        #     low_edge_index = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
        #     low_edge_index, low_edge_weight = dense_to_sparse(low_edge_index)
            
        #     graph[graph==0] = 10
        #     indices = graph.topk(k=int(high_k), largest=False)[1]
        #     indices = [indices[i][graph[i][indices[i]] < 10] if (graph[i][indices[i]] < 10).sum() > 0 else torch.tensor([i]) for i in range(indices.shape[0]) ]
        #     max_length = max([len(l) for l in indices])
        #     indices = [torch.cat((l, l[-1].repeat(max_length - len(l)))).to(device) for l in indices]
        #     indices = torch.stack(indices)
        #     high_edge_index = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
        #     high_edge_index, high_edge_weight = dense_to_sparse(high_edge_index)
        # elif args.edge == "hard_ratio":
            # graph = x @ x.t()
            # adj_idx = to_dense_adj(edge_index)[0]
            # num_edges = [i.sum() for i in adj_idx]
            # graph = graph*adj_idx
            # indices = []
            # for i in range(len(num_edges)):
            #     num = num_edges[i].item()
            #     ratio = int(num * low_k)
            #     indices.append(graph[i].topk(k=ratio)[1])
            # indices = [indices[i] if indices[i].shape[0] > 0 else torch.tensor([i]) for i in range(len(indices))]
            # max_length = max([len(l) for l in indices])
            # indices = [torch.cat((l, l[-1].repeat(max_length - len(l)))).to(device) for l in indices]
            # indices = torch.stack(indices)
            # low_edge_index = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
            # low_edge_index, low_edge_weight = dense_to_sparse(low_edge_index)

            # graph[graph==0] = 10
            # indices = []
            # for i in range(len(num_edges)):
            #     num = num_edges[i].item()
            #     ratio = int(num * high_k)
            #     indices.append(graph[i].topk(k=ratio, largest=False)[1])
            # indices = [indices[i] if indices[i].shape[0] > 0 else torch.tensor([i]) for i in range(len(indices))]
            # max_length = max([len(l) for l in indices])
            # indices = [torch.cat((l, l[-1].repeat(max_length - len(l)))).to(device) for l in indices]
            # indices = torch.stack(indices)
            # high_edge_index = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
            # high_edge_index, high_edge_weight = dense_to_sparse(high_edge_index)
        # elif args.edge == "soft":
        #     graph = x @ x.t()
        #     adj_idx = to_dense_adj(edge_index)[0]
        #     graph = graph * adj_idx
        #     return graph, adj_idx


    

def edge_create(args, x, edge_index, high_k=0, low_k=0, device=None):
    with torch.no_grad():
        if args.edge == "hard_num":
            graph = x @ x.t()
            adj_idx = to_dense_adj(edge_index)[0]
            graph = graph*adj_idx
            indices = graph.topk(k=int(low_k))[1]
            indices = [indices[i][graph[i][indices[i]] > 0] if (graph[i][indices[i]] > 0).sum() > 0 else torch.tensor([i]) for i in range(indices.shape[0])]
            max_length = max([len(l) for l in indices])
            indices = [torch.cat((l, l[-1].repeat(max_length - len(l)))).to(device) for l in indices]
            indices = torch.stack(indices)
            low_edge_index = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
            low_edge_index, low_edge_weight = dense_to_sparse(low_edge_index)
            
            graph[graph==0] = 10
            indices = graph.topk(k=int(high_k), largest=False)[1]
            indices = [indices[i][graph[i][indices[i]] < 10] if (graph[i][indices[i]] < 10).sum() > 0 else torch.tensor([i]) for i in range(indices.shape[0]) ]
            max_length = max([len(l) for l in indices])
            indices = [torch.cat((l, l[-1].repeat(max_length - len(l)))).to(device) for l in indices]
            indices = torch.stack(indices)
            high_edge_index = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
            high_edge_index, high_edge_weight = dense_to_sparse(high_edge_index)
        elif args.edge == "hard_ratio":
            graph = x @ x.t()
            adj_idx = to_dense_adj(edge_index)[0]
            graph = graph*adj_idx
            edge_idx, edge_weight = dense_to_sparse(graph)
            node_lst = {}
            weight_lst = {}
            k = len(edge_idx.t())
            for i in range(k):
                if i > 0 and edge_idx[0][i] == edge_idx[0][i-1]:
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
                t_len = len(wgt)
                hk = np.ceil(t_len * high_k).astype(int)
                lk = np.ceil(t_len * low_k).astype(int)
                wgt = torch.tensor(wgt)
                low_edge_wgt, low_edge_idx = torch.topk(wgt, lk)
                remained_wgts, remained_idx = th_delete(wgt, low_edge_idx)
                if len(remained_wgts) > hk:
                    high_edge_wgt, high_edge_idx = torch.topk(remained_wgts, hk,largest=False)
                    high_edge_idx = remained_idx[high_edge_idx]
                else:
                    high_edge_wgt, high_edge_idx = remained_wgts, remained_idx
                for idx, neigh_node in enumerate(high_edge_idx):
                    edge_pair = torch.tensor([node, node_lst[node][neigh_node]])
                    high_graph.append(edge_pair)
                    high_weight.append(high_edge_wgt[idx])
                for idx, neigh_node in enumerate(low_edge_idx):
                    edge_pair = torch.tensor([node, node_lst[node][neigh_node]])
                    low_graph.append(edge_pair)
                    low_weight.append(low_edge_wgt[idx])
            high_edge_index = torch.stack(high_graph).t().to(device)
            low_edge_index = torch.stack(low_graph).t().to(device)
            low_edge_weight = torch.ones(low_edge_index.shape[1]).to(device)
            high_edge_weight = torch.ones(high_edge_index.shape[1]).to(device)
    
            # graph = x @ x.t()
            # adj_idx = to_dense_adj(edge_index)[0]
            # num_edges = [i.sum() for i in adj_idx]
            # graph = graph*adj_idx
            # indices = []
            # for i in range(len(num_edges)):
            #     num = num_edges[i].item()
            #     ratio = int(num * low_k)
            #     indices.append(graph[i].topk(k=ratio)[1])
            # indices = [indices[i] if indices[i].shape[0] > 0 else torch.tensor([i]) for i in range(len(indices))]
            # max_length = max([len(l) for l in indices])
            # indices = [torch.cat((l, l[-1].repeat(max_length - len(l)))).to(device) for l in indices]
            # indices = torch.stack(indices)
            # low_edge_index = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
            # low_edge_index, low_edge_weight = dense_to_sparse(low_edge_index)

            # graph[graph==0] = 10
            # indices = []
            # for i in range(len(num_edges)):
            #     num = num_edges[i].item()
            #     ratio = int(num * high_k)
            #     indices.append(graph[i].topk(k=ratio, largest=False)[1])
            # indices = [indices[i] if indices[i].shape[0] > 0 else torch.tensor([i]) for i in range(len(indices))]
            # max_length = max([len(l) for l in indices])
            # indices = [torch.cat((l, l[-1].repeat(max_length - len(l)))).to(device) for l in indices]
            # indices = torch.stack(indices)
            # high_edge_index = torch.zeros(graph.shape[0], graph.shape[0]).to(device).scatter_(1, indices, 1)
            # high_edge_index, high_edge_weight = dense_to_sparse(high_edge_index)
        elif args.edge == "soft":
            graph = x @ x.t()
            adj_idx = to_dense_adj(edge_index)[0]
            graph = graph * adj_idx
            return graph, adj_idx


    return low_edge_index, high_edge_index, low_edge_weight, high_edge_weight

def two_hop(data):
    N = data.num_nodes
    adj = to_torch_csr_tensor(data.edge_index, size=(N, N))
    edge_index2, _ = to_edge_index(adj @ adj)
    edge_index2, _ = remove_self_loops(edge_index2)
    edge_index = torch.cat([data.edge_index, edge_index2], dim=1)
    data.edge_index = edge_index
    return data

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
        low_pass_edges = union(low_edges_0, high_edges_0)
        high_pass_edges = union(low_edges_1, high_edges_1)
        low_edge_weight = torch.ones(low_pass_edges.shape[1])
        high_edge_weight = torch.ones(high_pass_edges.shape[1])
    return ret, low_pass_edges, high_pass_edges, low_edge_weight, high_edge_weight

def res_combine(args, device, edge_index, low_k=None, high_k=None, low_x=None, high_x=None):
    with torch.no_grad():
        ret = representation_combine(args, low_x, high_x)
        # if args.edge != "soft":
        # if args.md == "union":
        low_edges_0,low_edges_1, low_edge_weight_0, low_edge_weight_1 = edge_create(args, low_x, edge_index, high_k, low_k, device)
        high_edges_0,high_edges_1, high_edge_weight_0, high_edge_weight_1 = edge_create(args, high_x, edge_index, high_k, low_k, device)
        low_pass_edges = union(low_edges_0, high_edges_0)
        high_pass_edges = union(low_edges_1, high_edges_1)
        low_edge_weight = torch.ones(low_pass_edges.shape[1])
        high_edge_weight = torch.ones(high_pass_edges.shape[1])
    return ret, low_pass_edges, high_pass_edges, low_edge_weight, high_edge_weight
        # elif args.md == "intersect":
        #     low_edges_0,low_edges_1, low_edge_weight_0, low_edge_weight_1 = edge_create(args, low_x, edge_index, high_k, low_k, device)
        #     high_edges_0,high_edges_1, high_edge_weight_0, high_edge_weight_1 = edge_create(args, high_x, edge_index, high_k, low_k, device)
        #     low_pass_edges = intersect(low_edges_0, high_edges_0)
        #     high_pass_edges = intersect(low_edges_1, high_edges_1)
        #     low_edge_weight = torch.ones(low_pass_edges.shape[1])
        #     high_edge_weight = torch.ones(high_pass_edges.shape[1])
        # elif args.md == "low":
        #     low_edges_0,low_edges_1, low_edge_weight_0, low_edge_weight_1 = edge_create(args, low_x, edge_index, high_k, low_k, device)
        #     low_pass_edges = low_edges_0
        #     high_pass_edges = low_edges_1
        #     low_edge_weight = torch.ones(low_pass_edges.shape[1])
        #     high_edge_weight = torch.ones(high_pass_edges.shape[1])
        # elif args.md == "high":
        #     high_edges_0,high_edges_1, high_edge_weight_0, high_edge_weight_1 = edge_create(args, high_x, edge_index, high_k, low_k, device)
        #     low_pass_edges = high_edges_0
        #     high_pass_edges = high_edges_1
        #     low_edge_weight = torch.ones(low_pass_edges.shape[1])
        #     high_edge_weight = torch.ones(high_pass_edges.shape[1])
    # else:
    #     if args.md == "union":
    #         low_graph, adj_idx = edge_create(args, low_x, edge_index, high_k, low_k, device)
    #         high_graph, adj_idx = edge_create(args, high_x, edge_index, high_k, low_k, device)
    #         edge, weight = soft_union(low_graph, high_graph, adj_idx)
    #         low_pass_edges = edge[0]
    #         high_pass_edges = edge[1]

    #         low_edge_weight = weight[0]
    #         high_edge_weight = weight[1]

    #     elif args.md == "low":
    #         graph, adj_idx = edge_create(args,low_x, edge_index)
    #         low_graph = F.normalize(graph)
    #         high_graph = F.normalize(adj_idx - low_graph)
    #         low_pass_edges, low_edge_weight = dense_to_sparse(low_graph)
    #         high_pass_edges, high_edge_weight = dense_to_sparse(high_graph)
    #     elif args.md == "high":
    #         graph, adj_idx = edge_create(args,high_x, edge_index)
    #         low_graph = F.normalize(graph)
    #         high_graph = F.normalize(adj_idx - low_graph)
    #         low_pass_edges, low_edge_weight = dense_to_sparse(low_graph)
    #         high_pass_edges, high_edge_weight = dense_to_sparse(high_graph)
    return ret, low_pass_edges, high_pass_edges, low_edge_weight, high_edge_weight

        

def representation_combine(args, z1, z2):
    if args.combine_x:
        return torch.cat((z1,z2),dim=1), z1, z2
    else:
        return z1, z1, z2
    


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
        

def get_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = "cora", help='benchmark dataset : cora, citeseer, pubmed')
    parser.add_argument('--pre_eval', type=int, default = 50, help='number per epoch to evaluate GCL')
    parser.add_argument('--hidden', type=int, default = 512, help='gcn, gat, fbgcn')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--intraview_negs' ,default="none") # delete
    parser.add_argument('--preepochs' ,type=int, default = 500, help='pretraining epoch')
    parser.add_argument('--per_epoch' ,type=int, default = 50, help='graph update frequency')
    parser.add_argument('--pre_learning_rate',type=float, default = 0.001, help='pre training learning rate')
    parser.add_argument('--aug1', type=float, default = 0.2, help='aug parameter')
    parser.add_argument('--aug2', type=float, default = 0.2, help='aug parameter')
    parser.add_argument('--runs', type=int, default=10, help='number of distinct runs')
    parser.add_argument('--num_layer', type=int, default=2, help='number of layers')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--edge', type=str, default="hard_ratio") # delete
    parser.add_argument('--low_k', type=float, default=0.2)
    parser.add_argument('--high_k', type=float, default=0.2)
    parser.add_argument('--infer_edges', action='store_true') # delete
    parser.add_argument('--combine_x', action='store_true')
    parser.add_argument('--md', type=str, default = "union") # delete
    parser.add_argument('--mode', type=str, default = "known") # delete
    parser.add_argument('--neg', type=str, default = "simple")
    parser.add_argument('--split', type=str, default = "simple")
    parser.add_argument('--cluster_batch_size', type=int, default = 1)
    parser.add_argument('--num_parts', type=int, default = 200)
    parser.add_argument('--eval', type=str, default = "acc")
    args, unknown = parser.parse_known_args()
    return args

