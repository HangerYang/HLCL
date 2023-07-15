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
from GCL.losses import Loss
from abc import ABC, abstractmethod
from utility.data import dataset_split
import argparse
from torch_geometric.utils import (
    remove_self_loops,
    to_edge_index,
    to_torch_csr_tensor,
)
import os
class Mix_Pass(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
    def forward(self, x, edge_index, edge_weight, high_pass = False):
        if high_pass:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization="sym")
            
            x = self.lin(x)
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight, high_pass=high_pass)
            out += self.bias
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            # edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization="sym")

            # Step 2: Linearly transform node feature matrix.
            x = self.lin(x)

            # Step 3: Compute normalization.
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Step 4-5: Start propagating messages.
            out = self.propagate(edge_index, x=x, norm=norm,high_pass=high_pass)

            # Step 6: Apply a final bias vector.
            out += self.bias
        return out
    def message(self, x_j, high_pass, norm=None, edge_weight=None):
        if high_pass:
            return edge_weight.view(-1, 1) * x_j
        else:
            return norm.view(-1, 1) * x_j

def seed_everything(seed=1234):                                                                                                          
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False
    random.seed(seed) 


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = "cora", help='benchmark dataset : cora, citeseer, pubmed')
    parser.add_argument('--pre_eval', type=int, default = 50, help='number per epoch to evaluate GCL')
    # parser.add_argument('--gnn', type=str, help='gcn, gat, fbgcn')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--intraview_negs' ,type=str, default="none", help='hidden dimension in the model')
    parser.add_argument('--preepochs' ,type=int, default = 500, help='pretraining epoch')
    parser.add_argument('--pre_learning_rate',type=float, default = 0.01, help='pre training learning rate')
    parser.add_argument('--aug1', type=float, default = 0.3, help='aug parameter')
    parser.add_argument('--aug2', type=float, default = 0.3, help='aug parameter')
    parser.add_argument('--runs', type=int, default=3, help='number of distinct runs')
    parser.add_argument('--neg', type=str, default="full_neg", help='number of distinct runs')
    parser.add_argument('--num_layer', type=int, default=2, help='number of layers')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--low_aug_only', action='store_true')
    args, unknown = parser.parse_known_args()
    return args

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(Mix_Pass(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(Mix_Pass(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None, high_pass=False):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight,high_pass)
            z = self.activation(z)
        return z
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder 
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, lp_edge_index, hp_edge_index, lp_edge_weight=None, hp_edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, lp_edge_index, lp_edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, hp_edge_index, hp_edge_weight)
        z = self.encoder(x, edge_index1, edge_weight1)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        # z = self.encoder(x)
        # z1 = self.encoder(x1)
        # z2 = self.encoder(x2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
def train(encoder_model, contrast_model, x, lp_edge_index, hp_edge_index, lp_edge_weight, hp_edge_weight, optimizer, neg_mask = None):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(x, lp_edge_index, hp_edge_index,lp_edge_weight, hp_edge_weight)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    # h1, h2 = z1, z2
    loss = contrast_model(h1, h2, neg_mask = neg_mask)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, x, lp_edge_index, hp_edge_index, lp_edge_weight, hp_edge_weight, y, split):
    encoder_model.eval()
    z, _, _ = encoder_model(x, lp_edge_index, hp_edge_index,lp_edge_weight, hp_edge_weight)
    # split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, y, split)
    return result


class Sampler(ABC):
    def __init__(self, intraview_negs="none"):
        self.intraview_negs = intraview_negs

    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        # if self.intraview_negs:
        ret = self.add_intraview_negs(*ret, self.intraview_negs)
        return ret
    
    @abstractmethod
    def sample(self, anchor, sample, *args, **kwargs):
        pass

    @staticmethod
    def add_intraview_negs(anchor, sample, pos_mask, neg_mask, intraview_negs):
        if intraview_negs == "none":
            return anchor, sample, pos_mask, neg_mask
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        if intraview_negs == "simple":
            intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        elif intraview_negs == "origin":
            intraview_neg_mask = neg_mask
        new_sample = torch.cat([sample, anchor], dim=0)                     # (M+N) * K
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)     # M * (M+N)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)     # M * (M+N)
        return anchor, new_sample, new_pos_mask, new_neg_mask
    
class SameScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, neg_mask = None, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        if neg_mask is None:
            neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask
        # return anchor, sample, pos_mask
    
def get_sampler(mode: str, intraview_negs: bool) -> Sampler:
    return SameScaleSampler(intraview_negs=intraview_negs)
    
class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None, neg_mask=None):
        if self.mode == 'L2L':
            anchor1, sample1, pos_mask1, neg_mask1= self.sampler(anchor=h1, sample=h2, neg_mask = neg_mask)
            anchor2, sample2, pos_mask2, neg_mask2= self.sampler(anchor=h2, sample=h1, neg_mask = neg_mask)
        # elif self.mode == 'G2G':
        #     assert g1 is not None and g2 is not None
        #     anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
        #     anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        # else:  # global-to-local
        #     if batch is None or batch.max().item() + 1 <= 1:  # single graph
        #         assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
        #         anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
        #         anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
        #     else:  # multiple graphs
        #         assert all(v is not None for v in [h1, h2, g1, g2, batch])
        #         anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
        #         anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        # pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        # pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5
    

args = get_arguments()
seed_everything(args.seed)
device = args.device
data = dataset_split(dataset_name = args.dataset)
hidden_dim=512
proj_dim=256
total_result = []
device = torch.device("cuda:{}".format(device))
data = data.to(device)
hidden_dim = hidden_dim
pre_learning_rate = args.pre_learning_rate
preepochs=args.preepochs
neg_masks = []
# mlp_model = torch.nn.Sequential(
#     torch.nn.Linear(data.x.shape[1], hidden_dim),
#     torch.nn.ReLU(),
#     torch.nn.Linear(hidden_dim, proj_dim),
#     torch.nn.ReLU())
# neg_sample = [neg_masks[i] for i in data.y]
# neg_sample = torch.stack(neg_sample).to(device)
epoch = 0
for run in range(args.runs):
    split = get_split(data.x.size()[0], train_ratio=0.1, test_ratio=0.8)
    split["train"] = split["train"].to(device)
    neg_sample=[]
    if args.neg == "full_neg":    
        for i in range(torch.unique(data.y).shape[0]):
            nongroup = torch.where(torch.logical_and(data.y!=i, torch.tensor([k in split["train"] for k in torch.tensor(range(data.num_nodes))]).to(device)))[0]
            neg_mask = torch.zeros(data.x.shape[0])
            neg_mask[nongroup] = 1.
            neg_masks.append(neg_mask) 
        for j in range(len(data.y)):
            neg_sample.append(neg_masks[data.y[j]])
        neg_sample = torch.stack(neg_sample).to(device)
    elif args.neg == "one_hop":
        for j in range(len(data.y)):
            neighbors = torch.where(data.edge_index[0] == j)[0]
            irrelevant_neighbors = data.edge_index[1][neighbors][data.y[data.edge_index[1][neighbors]] != data.y[j]]
            irrelevant_neighbors = torch.tensor([i for i in irrelevant_neighbors if i in split["train"]])
            neg_mask = torch.zeros(data.x.shape[0])
            if len(irrelevant_neighbors) > 0:
                neg_mask[irrelevant_neighbors] = 1.
            neg_sample.append(neg_mask)
        neg_sample = torch.stack(neg_sample).to(device)
    elif args.neg == "two_hop":
        N = data.num_nodes
        adj = to_torch_csr_tensor(data.edge_index, size=(N, N))
        edge_index2, _ = to_edge_index(adj @ adj)
        edge_index2, _ = remove_self_loops(edge_index2)
        edge_index = torch.cat([data.edge_index, edge_index2], dim=1)
        for j in range(len(data.y)):
            neighbors = torch.where(edge_index[0] == j)[0]
            irrelevant_neighbors = edge_index[1][neighbors][data.y[edge_index[1][neighbors]] != data.y[j]]
            irrelevant_neighbors = torch.tensor([i for i in irrelevant_neighbors if i in split["train"]])
            neg_mask = torch.zeros(data.x.shape[0])
            if len(irrelevant_neighbors) > 0:
                neg_mask[irrelevant_neighbors] = 1.
            neg_sample.append(neg_mask)
        neg_sample = torch.stack(neg_sample).to(device)
    elif args.neg == "pos":
        neg_mask = torch.zeros(data.x.shape[0])
        for j in range(len(data.y)):
            neg_sample.append(neg_mask)
        neg_sample = torch.stack(neg_sample).to(device)
    elif args.neg == "simple":
        neg_sample=None
    
    aug1 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=args.aug1), A.FeatureMasking(pf=args.aug2)])
    if args.low_aug_only:
        aug2 = A.Compose([A.EdgeRemoving(pe=0.), A.FeatureMasking(pf=0.)])
    gconv = GConv(input_dim=data.num_features, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=args.num_layer).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim, proj_dim=hidden_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=args.intraview_negs).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=pre_learning_rate)

    with tqdm(total=preepochs, desc='(T)') as pbar:
        for epoch in range(preepochs):
            loss = train(encoder_model, contrast_model, data.x, data.edge_index, data.edge_index, data.edge_weight, data.edge_weight, optimizer, neg_sample)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch % args.pre_eval == 0 and epoch >= args.pre_eval:
                test_result = test(encoder_model, data.x, data.edge_index, data.edge_index, data.edge_weight, data.edge_weight, data.y, split)
                total_result.append((run, epoch, test_result["accuracy"]))

test_result = test(encoder_model, data.x, data.edge_index, data.edge_index, data.edge_weight, data.edge_weight, data.y, split)
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


with open('./results/nc_GRACE_{}_{}.csv'.format(args.dataset, args.neg), 'a+') as file:
    file.write('\n')
    file.write('pre_learning_rate = {}\n'.format(pre_learning_rate))
    file.write('EdgeRemoving Aug = {}\n'.format(args.aug1))
    file.write('FeatMasking Aug = {}\n'.format(args.aug2))
    if args.low_aug_only:
        file.write("Only Low-Pass Augmentation\n")
    file.write('Intra_Neg: {}\n'.format(args.intraview_negs))
    file.write('Num of Layers = {}\n'.format(args.num_layer))
    file.write('(E): GRACE Mean Accuracy: {}, with Std: {}, at Epoch {}'.format(best_acc, best_std, best_epoch))