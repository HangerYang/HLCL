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
from HLCL.models import Encoder_updated,HLCLConv, DualBranchContrast
from torch_geometric.utils import dense_to_sparse, homophily

def train(args, graph, encoder_model, contrast_model, optimizer, device, edges):
    encoder_model.train()
    optimizer.zero_grad()
    if edges:
        (z, z1, z2), new_graph = encoder_model(args, graph, edges=True, device = device)
        new_graph.low_edge_index = torch.cat((new_graph.known_low_edge_index, new_graph.low_edge_index), dim=1)
        new_graph.high_edge_index = torch.cat((new_graph.known_high_edge_index, new_graph.high_edge_index), dim=1)
        new_graph.low_edge_weight = torch.cat((new_graph.known_low_edge_weight, new_graph.low_edge_weight))
        new_graph.high_edge_weight = torch.cat((new_graph.known_high_edge_weight, new_graph.high_edge_weight))
    else:
        (z, z1, z2) = encoder_model(args, graph, edges=False)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2, neg_mask=graph.neg_mask)
    loss.backward()
    optimizer.step()
    if edges:
        return loss.item(), new_graph
    else:
        return loss.item()

def test(args, graph, encoder_model, split):
    encoder_model.eval()
    # results = []
    graph.y = torch.flatten(graph.y)
    right_idx = torch.where(graph.y>-1)[0]
    z, _, _= encoder_model(args, graph, edges=False)
    z = z[right_idx]
    graph.y = graph.y[right_idx]
    # split = get_split(num_samples=z.size()[0], train_ratio=0.5, test_ratio=0.25)
    result = LREvaluator()(z, graph.y, split, args.eval)
    # results.append(result["accuracy"])
    # result = np.mean(results)
    return result



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
    low_pass_graph = []
    high_pass_graph = []
    
    if args.split == "simple":
        split = get_split(data.x.size()[0], train_ratio=0.48, test_ratio=0.2)
        split["train"] = split["train"].to(device)
    else:
        split = get_split_given(data, run, device)
    unknown_edges, known_edges = split_edges(data.edge_index, split)
    for i in known_edges:
        if (data.y[i[0]] == data.y[i[1]]) :
            low_pass_graph.append(i)
        else:
            high_pass_graph.append(i)
    data.known_low_edge_index = torch.stack(low_pass_graph).T.to(device)
    data.known_high_edge_index = torch.stack(high_pass_graph).T.to(device)
    data.known_high_edge_weight = torch.ones(data.known_high_edge_index.shape[1]).to(device)
    data.known_low_edge_weight = torch.ones(data.known_low_edge_index.shape[1]).to(device)
    if args.infer_edges:
        
        data = edge_create_updated(args, data, device)
        data.low_edge_index = torch.cat((data.known_low_edge_index, data.low_edge_index), dim=1)
        data.high_edge_index = torch.cat((data.known_high_edge_index, data.high_edge_index), dim=1)
        data.low_edge_weight = torch.cat((data.known_low_edge_weight, data.low_edge_weight))
        data.high_edge_weight = torch.cat((data.known_high_edge_weight, data.high_edge_weight))
        print(homophily(data.low_edge_index,data.y))
        print(homophily(data.high_edge_index,data.y))
        data = create_neg_mask_cuda_updated(args, [data],device)[0]
    data = data.to(device)
    neg_masks = []
    neg_sample=[]  
    for i in range(torch.unique(data.y).shape[0]):
        nongroup = torch.where(torch.logical_and(data.y!=i, torch.tensor([k in split["train"] for k in torch.tensor(range(data.num_nodes))]).to(device)))[0]
        neg_mask = torch.zeros(data.x.shape[0])
        neg_mask[nongroup] = 1.
        neg_masks.append(neg_mask.to(device))
    for j in range(len(data.y)):
        if j in split["train"]:
            neg_sample.append(neg_masks[data.y[j]])
        else:
            if args.neg == "simple":
                neg_sample.append(torch.ones(data.x.shape[0]).scatter_(0 ,torch.tensor(j), 0).to(device))
            elif args.infer_edges:
                neg_sample.append(data.neg_mask[j])
    data.neg_mask = torch.stack(neg_sample).to(device)
    with tqdm(total=preepochs, desc='(T)') as pbar:
        for epoch in range(preepochs):
            if epoch % args.per_epoch == 0 and epoch >= args.per_epoch:
                loss, data = train(args, data, encoder_model, contrast_model, optimizer, device, edges=True)
                print(homophily(data.low_edge_index,data.y))
                print(homophily(data.high_edge_index,data.y))
                if args.neg != "simple":
                    data = create_neg_mask_cuda_updated(args, [data],device)[0]
                    for j in range(len(data.y)):
                        if j in split["train"]:
                            data.neg_mask = neg_masks[data.y[j]]
                        else:
                            neg_sample.append(data.neg_mask[j])
            else:
                loss = train(args, data, encoder_model, contrast_model, optimizer, device, edges=False)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch % args.pre_eval == 0 and epoch >= args.pre_eval:
                test_result = test(args, data, encoder_model, split)
                total_result.append((run, epoch, test_result["accuracy"]))
    test_result = test(args, data, encoder_model, split)
    total_result.append((run, epoch, test_result["accuracy"]))
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
    file.write('Negative Masks: {}\n'.format(args.neg))
    file.write('Num of Layers = {}\n'.format(args.num_layer))
    file.write('Split = {}\n'.format(args.split))
    file.write('(E):Mean Accuracy: {}, with Std: {}, at Epoch {}'.format(best_acc, best_std, best_epoch))