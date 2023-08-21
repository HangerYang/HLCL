import torch
from torch_geometric.utils import add_self_loops
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from Evaluator import LREvaluator
import numpy as np

from utility.data import dataset_split
import datetime
from HLCL.utils import get_arguments, split_edges,seed_everything, edge_create_updated, create_neg_mask_cuda_updated, create_neg_mask_updated
import os
from HLCL.models import Encoder_updated, HLCLConv, DualBranchContrast
from torch_geometric.utils import homophily
from torch_geometric.loader import ClusterData, ClusterLoader

def train(args, subgraphs, encoder_model, contrast_model, optimizer, device, edges):
    encoder_model.train()
    new_subgraphs = []
    total_loss = 0
    for subgraph in subgraphs:
        optimizer.zero_grad()
        if edges and args.infer_edges:
            (z, z1, z2), new_graph = encoder_model(args, subgraph, edges=True, device = device)
            new_graph.low_edge_index = torch.cat((new_graph.known_low_edge_index, new_graph.low_edge_index), dim=1)
            new_graph.high_edge_index = torch.cat((new_graph.known_high_edge_index, new_graph.high_edge_index), dim=1)
            new_graph.low_edge_weight = torch.cat((new_graph.known_low_edge_weight, new_graph.low_edge_weight))
            new_graph.high_edge_weight = torch.cat((new_graph.known_high_edge_weight, new_graph.high_edge_weight))
            new_subgraphs.append(new_graph)
        elif edges:
            (z, z1, z2), new_graph = encoder_model(args, subgraph, edges=True, device = device)
            new_subgraphs.append(new_graph)
        else:
            (z, z1, z2) = encoder_model(args, subgraph, edges=False)
        h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
        loss = contrast_model(h1, h2, neg_mask=subgraph.neg_mask)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss
    
    if edges:
        return total_loss.item(), new_subgraphs
    else:
        return total_loss.item()

def test(args, subgraphs, encoder_model, splits):
    encoder_model.eval()
    results = []
    for i, subgraph in enumerate(subgraphs):
        subgraph.y = torch.flatten(subgraph.y)
        right_idx = torch.where(subgraph.y>-1)[0]
        z, _, _= encoder_model(args, subgraph, edges=False)
        z = z[right_idx]
        subgraph.y = subgraph.y[right_idx]
        # split = get_split(num_samples=z.size()[0], train_ratio=0.5, test_ratio=0.25)
        result = LREvaluator()(z, subgraph.y, splits[i], args.eval)
        results.append(result["accuracy"])
    result = np.mean(results)
    return {
            'accuracy': result
        }



args = get_arguments()
seed_everything(args.seed)
device = args.device
data = dataset_split(dataset_name = args.dataset)
hidden_dim=args.hidden
proj_dim=256
total_result = []
device = torch.device("cuda:{}".format(device))
pre_learning_rate = args.pre_learning_rate
preepochs=args.preepochs
epoch = 0
low_k = args.low_k
high_k = args.high_k
def get_split_given(data, run, device):
    split = {}
    split["train"] = torch.where(data.train_mask.t()[run])[0].to(device)
    split["valid"] = torch.where(data.val_mask.t()[run])[0].to(device)
    split["test"] = torch.where(data.test_mask.t()[run])[0].to(device)
    return split

for run in range(args.runs):
    aug1 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    gconv = HLCLConv(input_dim=data.num_features, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=args.num_layer).to(device)
    encoder_model = Encoder_updated(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim=hidden_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=args.intraview_negs).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=pre_learning_rate)
    cluster_data = ClusterData(data, num_parts=args.num_parts)
    train_loader = ClusterLoader(cluster_data, batch_size=args.cluster_batch_size, shuffle=True, num_workers=8)
    subgraphs = []
    splits = []
    for subgraph in train_loader:
        low_pass_graph = []
        high_pass_graph = []
        if args.split == "simple":
            split = get_split(subgraph.x.size()[0], train_ratio=0.5, test_ratio=0.25)
            split["train"] = split["train"].to(device)
        else:
            split = get_split_given(subgraph, run, device)
        splits.append(split)
        unknown_edges, known_edges = split_edges(subgraph.edge_index, split)
        for i in known_edges:
            if (subgraph.y[i[0]] == subgraph.y[i[1]]) :
                low_pass_graph.append(i)
            else:
                high_pass_graph.append(i)
        if args.infer_edges:
            subgraph.known_low_edge_index = torch.stack(low_pass_graph).T.to(device)
            subgraph.known_high_edge_index = torch.stack(high_pass_graph).T.to(device)
            subgraph.known_high_edge_weight = torch.ones(subgraph.known_high_edge_index.shape[1]).to(device)
            subgraph.known_low_edge_weight = torch.ones(subgraph.known_low_edge_index.shape[1]).to(device)
            subgraph = edge_create_updated(args, subgraph, device)
            subgraph.low_edge_index = torch.cat((subgraph.known_low_edge_index, subgraph.low_edge_index), dim=1)
            subgraph.high_edge_index = torch.cat((subgraph.known_high_edge_index, subgraph.high_edge_index), dim=1)
            subgraph.low_edge_weight = torch.cat((subgraph.known_low_edge_weight, subgraph.low_edge_weight))
            subgraph.high_edge_weight = torch.cat((subgraph.known_high_edge_weight, subgraph.high_edge_weight))
            # print(homophily(data.low_edge_index,data.y))
            # print(homophily(data.high_edge_index,data.y))
        else:
            subgraph.low_edge_index = torch.stack(low_pass_graph).T.to(device)
            subgraph.high_edge_index = torch.stack(high_pass_graph).T.to(device)
            subgraph.high_edge_weight = torch.ones(subgraph.high_edge_index.shape[1]).to(device)
            subgraph.low_edge_weight = torch.ones(subgraph.low_edge_index.shape[1]).to(device)
        subgraph = create_neg_mask_cuda_updated(args, [subgraph],device)[0]
        subgraph = subgraph.to(device)
        neg_masks = []
        pos_masks = []
        neg_sample=[]  
        pos_sample=[]  
        for i in range(torch.unique(data.y).shape[0]):
            nongroup = torch.where(torch.logical_and(subgraph.y!=i, torch.tensor([k in split["train"] for k in torch.tensor(range(subgraph.num_nodes))]).to(device)))[0]
            yes_group = torch.where(torch.logical_and(subgraph.y==i, torch.tensor([k in split["train"] for k in torch.tensor(range(subgraph.num_nodes))]).to(device)))[0]
            neg_mask = torch.zeros(subgraph.x.shape[0])
            neg_mask[nongroup] = 1.
            neg_masks.append(neg_mask.to(device))

            pos_mask = torch.zeros(subgraph.x.shape[0])
            pos_mask[yes_group] = 1.
            pos_masks.append(pos_mask.to(device))
        for j in range(len(subgraph.y)):
            if j in split["train"]:
                neg_sample.append(neg_masks[subgraph.y[j]])
                pos_sample.append(pos_mask(subgraph.y[j]))
            else:
                pos_sample.append(torch.zeros(subgraph.x.shape[0]).scatter_(0 ,torch.tensor(j), 1).to(device))
                if args.neg == "simple":
                    neg_sample.append(torch.ones(subgraph.x.shape[0]).scatter_(0 ,torch.tensor(j), 0).to(device))
                elif args.infer_edges:
                    neg_sample.append(subgraph.neg_mask[j])
        subgraph.neg_mask = torch.stack(neg_sample).to(device)
        subgraphs.append(subgraph)
    
    
    
    with tqdm(total=preepochs, desc='(T)') as pbar:
        for epoch in range(preepochs):
            if epoch % args.per_epoch == 0 and epoch >= args.per_epoch:
                loss, subgraphs = train(args, subgraphs, encoder_model, contrast_model, optimizer, device, edges=True)
                # print(homophily(data.low_edge_index,data.y))
                # print(homophily(data.high_edge_index,data.y))

                # if args.neg != "simple":
                #     data = create_neg_mask_cuda_updated(args, [data],device)[0]
                #     for j in range(len(data.y)):
                #         if j in split["train"]:
                #             data.neg_mask = neg_masks[data.y[j]]
                #         else:
                #             neg_sample.append(data.neg_mask[j])
            else:
                loss = train(args, subgraphs, encoder_model, contrast_model, optimizer, device, edges=False)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch % args.pre_eval == 0 and epoch >= args.pre_eval:
                test_result = test(args, subgraphs, encoder_model, splits)
                total_result.append((run, epoch, test_result["accuracy"]))
test_result = test(args, subgraphs, encoder_model, splits)
total_result.append((run, epoch, test_result["accuracy"]))
# print(total_result)
performance = {"epoch": [], "acc":[], "std":[]}
total_result = np.asarray(total_result)
for epoch in np.unique(total_result[:,1]):
    performance['acc'].append(np.mean(total_result[np.where(total_result[:,1] == epoch), 2]))
    performance['std'].append(np.std(total_result[np.where(total_result[:,1] == epoch), 2]))
    performance['epoch'].append(epoch)
best_epoch = performance['epoch'][np.argmax(performance['acc'])]
best_std = performance['std'][np.argmax(performance['std'])]
best_acc = np.max(performance['acc'])
with open('./new_result/{}_HLCL_supervised_{}.csv'.format(args.dataset, args.edge), 'a+') as file:
    file.write('\n')
    file.write('Time: {}\n'.format(datetime.datetime.now()))
    file.write('pre_learning_rate = {}\n'.format(pre_learning_rate))
    file.write('Graph Update Frequency = {}\n'.format(args.per_epoch))
    file.write('Low K = {}\n'.format(args.low_k))
    file.write('High K = {}\n'.format(args.high_k))
    file.write('Edge Creation = {}\n'.format(args.md))
    file.write('Combine X = {}\n'.format(args.combine_x))
    file.write('EdgeRemoving Aug = {}\n'.format(args.aug1))
    file.write('FeatMasking Aug = {}\n'.format(args.aug2))
    file.write('Infer Edges: {}\n'.format(args.infer_edges))
    file.write('Num of Layers = {}\n'.format(args.num_layer))
    file.write('Split = {}\n'.format(args.split))
    file.write('(E):Mean Accuracy: {}, with Std: {}, at Epoch {}'.format(best_acc, best_std, best_epoch))