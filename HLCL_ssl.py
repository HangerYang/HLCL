import torch
import GCL.losses as L
import GCL.augmentors as A
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from Evaluator import LREvaluator
import numpy as np
from utility.data import dataset_split
from HLCL.utils import get_arguments, seed_everything, edge_create, two_hop, create_neg_mask
from HLCL.models import Encoder, HLCLConv, DualBranchContrast
from torch_geometric.utils import dense_to_sparse,remove_self_loops
import torch.nn.functional as F


def train(args, encoder_model, contrast_model, x, edge_index, device, lp_edge_index, hp_edge_index, low_edge_weight, high_edge_weight,optimizer, edges=False, neg_mask = None):
    encoder_model.train()
    optimizer.zero_grad()
    # hp_edge_index, high_edge_weight = remove_self_loops(hp_edge_index, high_edge_weight)
    if edges:
        (z, z1, z2), low_edge_index, high_edge_index, low_edge_weight, high_edge_weight = encoder_model(args, x, lp_edge_index, hp_edge_index ,low_edge_weight, high_edge_weight, origin_edge_index = edge_index, edges=True, device = device)
    else:
        (z, z1, z2) = encoder_model(args, x, lp_edge_index, hp_edge_index ,low_edge_weight, high_edge_weight)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    # h1, h2 = z1, z2
    loss = contrast_model(h1, h2, neg_mask=neg_mask)
    loss.backward()
    optimizer.step()
    if edges:
        if args.edge == "soft":
            return loss.item(), low_edge_index.to(device), high_edge_index.to(device), low_edge_weight.to(device), high_edge_weight.to(device)
        else:
            return loss.item(), low_edge_index.to(device), high_edge_index.to(device), None, None
    else:
        return loss.item()


def test(args, encoder_model, x, edge_index, device,lp_edge_index, hp_edge_index, low_edge_weight, high_edge_weight, y, split):
    encoder_model.eval()
    # hp_edge_index, high_edge_weight = remove_self_loops(hp_edge_index, high_edge_weight)
    z, _, _= encoder_model(args, x, lp_edge_index, hp_edge_index ,low_edge_weight, high_edge_weight)
    # split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, y, split)
    return result


args = get_arguments()
seed_everything(args.seed)
device = args.device
data = dataset_split(dataset_name = args.dataset)
if args.two_hop:
    data = two_hop(data)
hidden_dim=512
proj_dim=256
total_result = []
device = torch.device("cuda:{}".format(device))
hidden_dim = hidden_dim
pre_learning_rate = args.pre_learning_rate
preepochs=args.preepochs
epoch = 0
low_k = args.low_k
high_k = args.high_k
data = data.to(device)
for run in range(args.runs):
    split = get_split(data.x.size()[0], train_ratio=0.1, test_ratio=0.8)
    aug1 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])

    gconv = HLCLConv(input_dim=data.num_features, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=args.num_layer).to(device)
    if args.edge == "soft":
        graph, adj_idx = edge_create(args, data.x, data.edge_index)
        low_graph = F.normalize(graph)
        high_graph = F.normalize(adj_idx - low_graph)
        low_edge_index, low_edge_weight = dense_to_sparse(low_graph)
        high_edge_index, high_edge_weight = dense_to_sparse(high_graph)
    else:
        low_edge_index, high_edge_index, low_edge_weight, high_edge_weight = edge_create(args, data.x, data.edge_index, high_k, low_k, device)
    neg_mask = create_neg_mask(args, device, low_edge_index, high_edge_index, data.x)
    
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim=hidden_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=args.intraview_negs).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=pre_learning_rate)

    with tqdm(total=preepochs, desc='(T)') as pbar:
        for epoch in range(preepochs):
            if epoch % args.per_epoch == 0 and epoch >= args.per_epoch:
                loss, low_edge_index, high_edge_index, low_edge_weight, high_edge_weight= train(args = args, encoder_model = encoder_model, contrast_model = contrast_model, x = data.x, edge_index = data.edge_index, device= device, lp_edge_index = low_edge_index, hp_edge_index = high_edge_index, low_edge_weight = low_edge_weight, high_edge_weight = high_edge_weight, optimizer = optimizer,edges = True, neg_mask=neg_mask)
                neg_mask = create_neg_mask(args, device, low_edge_index, high_edge_index, data.x)
            else:
                loss = train(args = args, encoder_model = encoder_model, contrast_model = contrast_model, x = data.x, edge_index = data.edge_index, device= device, lp_edge_index = low_edge_index, hp_edge_index = high_edge_index, low_edge_weight = low_edge_weight, high_edge_weight = high_edge_weight, optimizer = optimizer,edges = False, neg_mask=neg_mask)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch % args.pre_eval == 0 and epoch >= args.pre_eval:
                test_result = test(args, encoder_model, data.x, data.edge_index, device, low_edge_index, high_edge_index, low_edge_weight, high_edge_weight, data.y, split)
                total_result.append((run, epoch, test_result["accuracy"]))

test_result = test(args, encoder_model, data.x, data.edge_index, device,low_edge_index, high_edge_index, low_edge_weight, high_edge_weight, data.y, split)
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


with open('./results/ssl_{}_{}_HLCL.csv'.format(args.dataset, args.edge), 'a+') as file:
    file.write('\n')
    file.write('Num of Layers = {}\n'.format(args.num_layer))
    file.write('Graph Update Frequency = {}\n'.format(args.per_epoch))
    file.write('Two Hop Edges = {}\n'.format(args.two_hop))
    file.write('Combine X = {}\n'.format(args.two_hop))
    file.write('Low K = {}\n'.format(args.low_k))
    file.write('High K = {}\n'.format(args.high_k))
    file.write('Edge Creation = {}\n'.format(args.md))
    file.write('Intra Negative = {}\n'.format(args.intraview_negs))
    file.write('(E): Mean Accuracy: {}, with Std: {}, at Epoch {}'.format(best_acc, best_std, best_epoch))
    file.write('\n')