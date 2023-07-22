from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter
import torch
from torch_geometric.utils import get_laplacian,add_self_loops, degree
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
from HLCL.utils import get_arguments, split_edges,seed_everything, edge_create, create_neg_mask,create_neg_mask_cuda
import os
from HLCL.models import Encoder,HLCLConv, DualBranchContrast
from torch_geometric.utils import dense_to_sparse
def train(args, encoder_model, contrast_model, x, lp_edge_index, hp_edge_index, lp_edge_weight, hp_edge_weight, optimizer, neg_mask = None):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(args, x, lp_edge_index, hp_edge_index,lp_edge_weight, hp_edge_weight)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    # h1, h2 = z1, z2
    loss = contrast_model(h1, h2, neg_mask = neg_mask)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_rewire(args, encoder_model, contrast_model, x, edge_index, device, lp_edge_index, hp_edge_index, lp_edge_weight, hp_edge_weight, optimizer, edges=False, known_info = None):
    encoder_model.train()
    optimizer.zero_grad()
    known_low_edge_index, known_high_edge_index, known_low_edge_weight, known_high_edge_weight = known_info
    cb_hp_edge_weight = torch.cat((known_high_edge_weight, hp_edge_weight))
    cb_lp_edge_weight = torch.cat((known_low_edge_weight, lp_edge_weight))
    cb_hp_edge_index = torch.cat((known_high_edge_index, hp_edge_index), dim=1)
    cb_lp_edge_index = torch.cat((known_low_edge_index, lp_edge_index), dim=1)
    if edges:
        (z, z1, z2), low_edge_index, high_edge_index, lp_edge_weight, hp_edge_weight = encoder_model(args, x, cb_lp_edge_index, cb_hp_edge_index ,cb_lp_edge_weight, cb_hp_edge_weight, edge_index, edges, device)
    else:
        (z, z1, z2) = encoder_model(args, x, cb_lp_edge_index, cb_hp_edge_index ,cb_lp_edge_weight, cb_hp_edge_weight)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    # h1, h2 = z1, z2
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    if edges:
        return loss.item(), low_edge_index.to(device), high_edge_index.to(device), lp_edge_weight.to(device), hp_edge_weight.to(device)
    else:
        return loss.item()


def test(args, encoder_model, x, lp_edge_index, hp_edge_index, lp_edge_weight, hp_edge_weight, y, split, known_info = None):
    encoder_model.eval()
    if args.mode == "rewire":
        known_low_edge_index, known_high_edge_index, known_low_edge_weight, known_high_edge_weight = known_info
        hp_edge_weight = torch.cat((known_high_edge_weight, hp_edge_weight))
        lp_edge_weight = torch.cat((known_low_edge_weight, lp_edge_weight))
        hp_edge_index = torch.cat((known_high_edge_index, hp_edge_index), dim=1)
        lp_edge_index = torch.cat((known_low_edge_index, lp_edge_index), dim=1)
    z, _, _ = encoder_model(args, x, lp_edge_index, hp_edge_index,lp_edge_weight, hp_edge_weight, edges=False)
    result = LREvaluator()(z, y, split)
    return result

def get_split_given(data, run, device):
    split = {}
    split["train"] = torch.where(data.train_mask.t()[run])[0].to(device)
    split["valid"] = torch.where(data.val_mask.t()[run])[0].to(device)
    split["test"] = torch.where(data.test_mask.t()[run])[0].to(device)
    return split

args = get_arguments()
seed_everything(args.seed)
device = args.device

data = dataset_split(dataset_name = args.dataset)
hidden_dim=512
proj_dim=256
total_result = []
device = torch.device("cuda:{}".format(device))
data = data.to(device)
pre_learning_rate = args.pre_learning_rate
preepochs=args.preepochs
neg_masks = []
epoch = 0
data.edge_index = add_self_loops(data.edge_index)[0]
for run in range(args.runs):
    split = get_split(data.x.size()[0], train_ratio=0.6, test_ratio=0.2)
    split["train"] = split["train"].to(device)
    low_pass_graph = []
    high_pass_graph = []
    # split = get_split_given(data, run, device)
    unknown_edges, known_edges = split_edges(data.edge_index, split)
    unknown_edges = unknown_edges.t()
    for i in known_edges:
        if (data.y[i[0]] == data.y[i[1]]) :
            low_pass_graph.append(i)
        else:
            high_pass_graph.append(i)
    known_low_edge_index = torch.stack(low_pass_graph).T
    known_high_edge_index = torch.stack(high_pass_graph).T
    known_low_edge_weight = torch.ones(known_low_edge_index.shape[1]).to(device)
    known_high_edge_weight = torch.ones(known_high_edge_index.shape[1]).to(device)
    known_info = known_low_edge_index, known_high_edge_index, known_low_edge_weight, known_high_edge_weight 
    if args.edge == "soft":
        graph, adj_idx = edge_create(args, data.x, unknown_edges)
        low_graph = F.normalize(graph)
        high_graph = F.normalize(adj_idx - low_graph)
        unknown_low_edge_index, unknown_low_edge_weight = dense_to_sparse(low_graph)
        unknown_high_edge_index, unknown_high_edge_weight = dense_to_sparse(high_graph)
    elif args.edge == "hard_ratio" or args.edge == "hard_num":
        unknown_low_edge_index, unknown_high_edge_index, unknown_low_edge_weight, unknown_high_edge_weight = edge_create(args, data.x, unknown_edges, args.high_k, args.low_k, device)
    neg_sample=[]  
    for i in range(torch.unique(data.y).shape[0]):
        nongroup = torch.where(torch.logical_and(data.y!=i, torch.tensor([k in split["train"] for k in torch.tensor(range(data.num_nodes))]).to(device)))[0]
        neg_mask = torch.zeros(data.x.shape[0])
        neg_mask[nongroup] = 1.
        neg_masks.append(neg_mask)
    unknown_neg_masks = create_neg_mask(args, device, unknown_low_edge_index, unknown_high_edge_index, data.x)
    for j in range(len(data.y)):
        if j in split["train"]:
            neg_sample.append(neg_masks[data.y[j]])
        else:
            if args.neg == "simple":
                neg_sample.append(torch.ones(data.x.shape[0]).scatter_(0 ,torch.tensor(j), 0))
            else:
                neg_sample.append(unknown_neg_masks[j])
    neg_sample = torch.stack(neg_sample).to(device)
    aug1 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    gconv = HLCLConv(input_dim=data.num_features, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=args.num_layer).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim=hidden_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=args.intraview_negs).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=pre_learning_rate)

    with tqdm(total=preepochs, desc='(T)') as pbar:
        if args.mode == "known":
            for epoch in range(preepochs):
                loss = train(args, encoder_model, contrast_model, data.x, known_low_edge_index, known_high_edge_index, known_low_edge_weight, known_high_edge_weight, optimizer, neg_sample)
                pbar.set_postfix({'loss': loss})
                pbar.update()
                if epoch % args.pre_eval == 0 and epoch >= args.pre_eval:
                    test_result = test(args, encoder_model, data.x, known_low_edge_index, known_high_edge_index, known_low_edge_weight, known_high_edge_weight, data.y, split)
                    total_result.append((run, epoch, test_result["accuracy"]))

            test_result = test(args, encoder_model, data.x, known_low_edge_index, known_high_edge_index, known_low_edge_weight, known_high_edge_weight, data.y, split)
            total_result.append((run, epoch, test_result["accuracy"]))
        elif args.mode == "rewire":
            for epoch in range(preepochs):
                if epoch % args.per_epoch == 0 and epoch >= args.per_epoch:
                    loss, unknown_low_edge_index, unknown_high_edge_index, unknown_low_edge_weight, unknown_high_edge_weight = train_rewire(args, encoder_model, contrast_model, data.x, unknown_edges, device, unknown_low_edge_index, unknown_high_edge_index, unknown_low_edge_weight, unknown_high_edge_weight, optimizer, True, known_info)
                    if args.neg != "simple":
                        unknown_neg_mask = create_neg_mask_cuda(args, device, unknown_low_edge_index, unknown_high_edge_index, data.x)
                        for j in range(len(data.y)):
                            if j in split["train"]:
                                pass
                            else:
                                neg_sample[j] = unknown_neg_masks[j]
                        
                else:
                    loss = train_rewire(args, encoder_model, contrast_model, data.x, unknown_edges, device, unknown_low_edge_index, unknown_high_edge_index, unknown_low_edge_weight, unknown_high_edge_weight, optimizer, False, known_info)
                pbar.set_postfix({'loss': loss})
                pbar.update()
                if epoch % args.pre_eval == 0 and epoch >= args.pre_eval:
                    test_result = test(args, encoder_model, data.x, unknown_low_edge_index, unknown_high_edge_index, unknown_low_edge_weight, unknown_high_edge_weight, data.y, split,known_info)
                    total_result.append((run, epoch, test_result["accuracy"]))

            test_result = test(args, encoder_model, data.x, unknown_low_edge_index, unknown_high_edge_index, unknown_low_edge_weight, unknown_high_edge_weight, data.y, split,known_info)
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
    file.write('Edge Mode = {}\n'.format(args.mode))
    file.write('EdgeRemoving Aug = {}\n'.format(args.aug1))
    file.write('FeatMasking Aug = {}\n'.format(args.aug2))
    file.write('Negative Masks: {}\n'.format(args.neg))
    file.write('Num of Layers = {}\n'.format(args.num_layer))
    file.write('(E):Mean Accuracy: {}, with Std: {}, at Epoch {}'.format(best_acc, best_std, best_epoch))