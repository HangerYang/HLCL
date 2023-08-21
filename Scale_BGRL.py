import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
from utility.config import get_arguments
from torch_geometric.utils import get_laplacian
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split
from GCL.models import BootstrapContrast
from model.trial_pretrain import GCNConv
# from utility.data import build_graph
from Evaluator import LREvaluator
import numpy as np
from torch_geometric.utils import to_undirected, sort_edge_index
from torch_geometric.loader import ClusterData, ClusterLoader
from utility.data import dataset_split
import datetime
from HLCL.utils import get_arguments
# import sys
# sys.path.append("/home/hyang/HL_Contrast/Non-Homophily-Large-Scale/")
# from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, to_sparse_tensor
# from parse import parse_method, parser_add_main_args
# from batch_utils import nc_dataset_to_torch_geo, torch_geo_to_nc_dataset, AdjRowLoader, make_loader
# from dataset import load_nc_dataset, NCDataset



class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None,normalize=True):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight,normalize)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target



def train(encoder_model, contrast_model, train_loader, optimizer, device):
    encoder_model.train()
    total_loss = 0
    for tg_batch in train_loader:
        tg_batch = tg_batch.to(device)
        optimizer.zero_grad()
        _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(tg_batch.x, tg_batch.edge_index, tg_batch.edge_attr)
        loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
        loss.backward()
        optimizer.step()
        encoder_model.update_target_encoder(0.99)
        total_loss += loss.item()
    return total_loss


def test(encoder_model, test_loader):
    encoder_model.eval()
    for tg_batch in test_loader:
        tg_batch = tg_batch.to(device)
        tg_batch.y = torch.flatten(tg_batch.y)
        right_idx = torch.where(tg_batch.y>-1)[0]
        h1, h2, _, _, _, _ = encoder_model(tg_batch.x, tg_batch.edge_index, tg_batch.edge_attr)
        h1 = h1[right_idx]
        h2 = h2[right_idx]
        z = torch.cat([h1, h2], dim=1)
        tg_batch.y = tg_batch.y[right_idx]
        split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
        result = LREvaluator()(z, tg_batch.y, split, args.eval)
    return result



total_result = []
args = get_arguments()
print(args)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = 'cuda:{}'.format(str(args.device)) if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# dataset = load_nc_dataset(args.dataset, args.sub_dataset)
data = dataset_split(dataset_name = args.dataset)
# if len(dataset.label.shape) == 1:
#     dataset.label = dataset.label.unsqueeze(1)
# dataset.label = dataset.label.to(device)

# split_idx = dataset.get_idx_split(train_prop=0.1, valid_prop=0.1)

# n = dataset.graph['num_nodes']
# # infer the number of classes for non one-hot and one-hot labels
# c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
# d = dataset.graph['node_feat'].shape[1]
# one_side = (args.aug_side != "both")

# train_loader, subgraph_loader = None, None
# print(f"num nodes {n} | num classes {c} | num node feats {d}")
# eval_func = eval_acc
# train_idx = split_idx['train']
# train_idx = train_idx.to(device)
# print('making train loader')
# dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
# train_loader = make_loader(args, dataset, train_idx, device=device)
# test_loader = make_loader(args, dataset, split_idx['test'], True, device=device, test=True)
for run in range(args.runs):
    cluster_data = ClusterData(data, num_parts=args.num_parts)
    train_loader = ClusterLoader(cluster_data, batch_size=args.cluster_batch_size, shuffle=True, num_workers=8)
    aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.2)])
    gconv = GConv(data.num_features, args.hidden, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=args.hidden).to(device)
    contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)
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

with open('./new_result/{}_BGRL.csv'.format(args.dataset), 'a+') as file:
    file.write('\n')
    file.write('Time: {}\n'.format(datetime.datetime.now()))
    file.write('(E): BGRL Mean Accuracy: {}, with Std: {}'.format(best_acc, best_std))
    file.write('\n')
# def main():
#     args = get_arguments()
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     device = torch.device("cuda")
#     dataset = args.dataset
#     hidden_dim = args.hidden
#     pre_learning_rate = args.pre_learning_rate
#     total_result = []

#     for i in range(10):
#         data = build_graph(dataset).to(device)
#         aug1 = A.Compose([A.EdgeRemoving(pe=args.aug), A.FeatureMasking(pf=args.aug2)])
#         aug2 = A.Compose([A.EdgeRemoving(pe=args.aug), A.FeatureMasking(pf=args.aug2)])
#         gconv = GConv(input_dim=data.num_features, hidden_dim=hidden_dim, num_layers=2).to(device)
#         encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim).to(device)
        
#         contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)

#         optimizer = Adam(encoder_model.parameters(), lr=pre_learning_rate)

#         with tqdm(total=args.preepochs, desc='(T)') as pbar:
#             for epoch in range(args.preepochs):
#                 loss = train(encoder_model, contrast_model, data, optimizer)
#                 pbar.set_postfix({'loss': loss})
#                 pbar.update()
#         test_result = test(encoder_model, data)
#         total_result.append(test_result["accuracy"])
    
    



#     with open('./results/nc_BRGL_{}_{}.csv'.format(args.dataset, args.gnn), 'a+') as file:
#             file.write('\n')
#             file.write('pretrain epochs = {}\n'.format(args.preepochs))
#             file.write('pre_learning_rate = {}\n'.format(args.pre_learning_rate))
#             file.write('hidden_dim = {}\n'.format(args.hidden))
#             file.write('aug = {}\n'.format(args.aug))
#             file.write('aug2 = {}\n'.format(args.aug2))
#             file.write('(E): BGRL Mean Accuracy: {}, with Std: {}'.format(np.mean(total_result), np.std(total_result)))


# if __name__ == '__main__':
#     main()