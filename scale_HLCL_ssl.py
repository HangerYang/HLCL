import torch
import GCL.losses as L
import GCL.augmentors as A
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from Evaluator import LREvaluator
import numpy as np
from utility.data import dataset_split
from HLCL.utils import get_arguments, seed_everything, edge_create_updated, create_neg_mask_cuda_updated
from HLCL.models import Encoder_updated, HLCLConv, DualBranchContrast
import torch.nn.functional as F
import datetime
from torch_geometric.loader import ClusterData, ClusterLoader

def train(args, subgraphs, encoder_model, contrast_model, optimizer, device, edges):
    encoder_model.train()
    new_subgraphs = []
    total_loss = 0
    for subgraph in subgraphs:
        optimizer.zero_grad()
        if edges:
            (z, z1, z2), subgraph = encoder_model(args, subgraph, edges=True, device = device)
            new_subgraphs.append(subgraph)
        else:
            (z, z1, z2) = encoder_model(args, subgraph, edges=False)
        h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
        # h1, h2 = z1, z2
        loss = contrast_model(h1, h2, neg_mask=subgraph.neg_mask)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss
    if edges:
        return total_loss.item(), new_subgraphs
    else:
        return total_loss.item()


def test(args, subgraphs, encoder_model):
    encoder_model.eval()
    results = []
    for subgraph in subgraphs:
        subgraph.y = torch.flatten(subgraph.y)
        right_idx = torch.where(subgraph.y>-1)[0]
        z, _, _= encoder_model(args, subgraph, edges=False)
        z = z[right_idx]
        subgraph.y = subgraph.y[right_idx]
        split = get_split(num_samples=z.size()[0], train_ratio=0.5, test_ratio=0.25)
        result = LREvaluator()(z, subgraph.y, split, args.eval)
        results.append(result["accuracy"])
    result = np.mean(results)
    return {
            'accuracy': result
        }

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
# if args.two_hop:
#     data = two_hop(data)
hidden_dim=args.hidden
proj_dim=256
total_result = []
device = torch.device("cuda:{}".format(device))
pre_learning_rate = args.pre_learning_rate
preepochs=args.preepochs
epoch = 0
low_k = args.low_k
high_k = args.high_k
for run in range(args.runs):
    aug1 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    gconv = HLCLConv(input_dim=data.num_features, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=args.num_layer).to(device)
    encoder_model = Encoder_updated(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim=hidden_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=args.intraview_negs).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=pre_learning_rate, weight_decay=5e-5)
    cluster_data = ClusterData(data, num_parts=args.num_parts)
    train_loader = ClusterLoader(cluster_data, batch_size=args.cluster_batch_size, shuffle=True, num_workers=8)
    subgraphs = []
    for subgraph in train_loader:
        subgraph = subgraph.to(device)
    # if args.edge == "soft":
    #     graph, adj_idx = edge_create(args, data.x, data.edge_index)
    #     low_graph = F.normalize(graph)
    #     high_graph = F.normalize(adj_idx - low_graph)
    #     low_edge_index, low_edge_weight = dense_to_sparse(low_graph)
    #     high_edge_index, high_edge_weight = dense_to_sparse(high_graph)
    # else:
        subgraph = edge_create_updated(args, subgraph, device)
        subgraphs.append(subgraph)
    subgraphs = create_neg_mask_cuda_updated(args, subgraphs,device)
    with tqdm(total=preepochs, desc='(T)') as pbar:
        for epoch in range(preepochs):
            if epoch % args.per_epoch == 0 and epoch >= args.per_epoch:
                loss, subgraphs = train(args, subgraphs, encoder_model, contrast_model, optimizer, device, edges=True)
                subgraphs = create_neg_mask_cuda_updated(args, subgraphs, device)
            else:
                loss = train(args, subgraphs, encoder_model, contrast_model, optimizer, device, edges=False)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch % args.pre_eval == 0 and epoch >= args.pre_eval:
                test_result = test(args, subgraphs, encoder_model)
                total_result.append((run, epoch, test_result["accuracy"]))

test_result = test(args, subgraphs, encoder_model)
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


with open('./new_result/{}_HLCL_ssl_{}.csv'.format(args.dataset, args.edge), 'a+') as file:
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