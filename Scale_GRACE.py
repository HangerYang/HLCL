import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import get_laplacian
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from Evaluator import LREvaluator
from GCL.models import DualBranchContrast
from model.trial_pretrain import GCNConv
from HLCL.utils import get_arguments
import numpy as np
from torch_geometric.utils import to_undirected, sort_edge_index
from GCL.eval import get_split
from utility.data import dataset_split
from torch_geometric.loader import ClusterData, ClusterLoader
import datetime
# import sys
# sys.path.append("/home/hyang/HL_Contrast/Non-Homophily-Large-Scale/")
# from data_utils import eval_acc
# from batch_utils import make_loader
# from dataset import load_nc_dataset

def train(encoder_model, contrast_model, train_loader, optimizer, device):
    encoder_model.train()
    total_loss = 0
    for tg_batch in train_loader:
        # batch_dataset = torch_geo_to_nc_dataset(tg_batch, device=device)
        tg_batch = tg_batch.to(device)
        optimizer.zero_grad()
        _, z1, z2 = encoder_model(tg_batch.x, tg_batch.edge_index, tg_batch.edge_attr)
        h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
        loss = contrast_model(h1, h2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def test(encoder_model, test_loader):
    encoder_model.eval()
    for tg_batch in test_loader:
        tg_batch = tg_batch.to(device)
        tg_batch.y = torch.flatten(tg_batch.y)
        right_idx = torch.where(tg_batch.y>-1)[0]
        z,_,_ = encoder_model(tg_batch.x, tg_batch.edge_index, tg_batch.edge_attr)
        z = z[right_idx]
        tg_batch.y = tg_batch.y[right_idx]
        split = get_split(num_samples=z.size()[0], train_ratio=0.5, test_ratio=0.25)
        # print(torch.sum(torch.flatten(tg_batch.y) >= 0))
        result = LREvaluator()(z, tg_batch.y, split, args.eval)
    return result

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, second_hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None, normalize=True):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight,normalize)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


total_result = []
args = get_arguments()
print(args)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = 'cuda:{}'.format(str(args.device)) if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
data = dataset_split(dataset_name = args.dataset)
# dataset = load_nc_dataset(args.dataset, args.sub_dataset)
# if len(dataset.label.shape) == 1:
#     dataset.label = dataset.label.unsqueeze(1)
# dataset.label = dataset.label.to(device)

# split_idx = dataset.get_idx_split(train_prop=0.5, valid_prop=0.25)

# n = dataset.graph['num_nodes']
# # infer the number of classes for non one-hot and one-hot labels
# c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
# d = dataset.graph['node_feat'].shape[1]
# dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
# one_side = (args.aug_side != "both")

# train_loader, subgraph_loader = None, None
# print(f"num nodes {n} | num classes {c} | num node feats {d}")
# eval_func = eval_acc
# train_idx = split_idx['train']
# train_idx = train_idx.to(device)
# print('making train loader')
# train_loader = make_loader(args, dataset, train_idx, device=device)
# test_loader = make_loader(args, dataset, split_idx['test'], True, device=device, test=True)
for run in range(args.runs):
    cluster_data = ClusterData(data, num_parts=args.num_parts)
    train_loader = ClusterLoader(cluster_data, batch_size=args.cluster_batch_size, shuffle=True, num_workers=8)
    aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.2)])
    gconv = GConv(data.num_features, args.hidden, args.hidden, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=args.hidden, proj_dim=args.hidden).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    optimizer = Adam(encoder_model.parameters(),
                        lr=args.pre_learning_rate, weight_decay=5e-5)
    with tqdm(total=args.preepochs, desc='(T)') as pbar:
            for epoch in range(args.preepochs):
                loss = train(encoder_model, contrast_model, train_loader, optimizer, device)
                pbar.set_postfix({'loss': loss})
                pbar.update()
                if epoch % args.pre_eval == 0 and epoch >= args.pre_eval:
                    test_result = test(encoder_model, train_loader)
                    total_result.append((test_result["accuracy"]))
    test_result = test(encoder_model, train_loader)
    total_result.append(test_result["accuracy"])

with open('./new_result/{}_GRACE.csv'.format(args.dataset), 'a+') as file:
    file.write('\n')
    file.write('Time: {}\n'.format(datetime.datetime.now()))
    file.write('(E): GRACE Mean Accuracy: {}, with Std: {}'.format(np.mean(total_result), np.std(total_result)))
    file.write('\n')