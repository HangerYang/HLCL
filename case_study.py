import numpy as np
import scipy.sparse as sp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import networkx as nx
import pickle as pkl

import torch_geometric.datasets as datasets
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, to_networkx


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora")
parser.add_argument('--root', type=str, default="SpCo/dataset/")
parser.add_argument('--interval', type=str, default="low", help="Augmentation interval of V")
parser.add_argument('--ratio', type=float, default="0.2", help="Different addtion ratio")

args, _ = parser.parse_known_args()

print(args.dataset, args.root, args.interval, args.ratio)

def normalize(mx):
    """Row-normalize sparse matrix.
    """
    r_sum = np.array(mx.sum(1))
    r_inv = np.power(r_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
        
def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret
    
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Contrast(nn.Module):
    def __init__(self, hidden_dim):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = 0.5
        self.lam = 0.5
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        
        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.diag()).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.diag()).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc

class Model(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(Model, self).__init__()
        self.gcn = GCN(in_ft, out_ft)
        self.con = Contrast(out_ft)
    
    def forward(self, feat, adj1, adj2):
        emb1 = self.gcn(feat, adj1)
        emb2 = self.gcn(feat, adj2)
        loss = self.con(emb1, emb2)
        return loss
    
    def get_emb(self, feat, adj):
        return self.gcn(feat, adj).detach()
    



out_ft = 8 # This parameter is randomly set. And we think if out_ft is larger, the performance will be better.
device = "cuda:5"
adj_ =sp.load_npz( args.root + args.dataset + "/adj.npz")
adj = sparse_mx_to_torch_sparse_tensor(adj_).to(device)
features = sp.load_npz( args.root + args.dataset + "/feat.npz")
features = torch.FloatTensor(features.A).to(device)
 
label = np.load( args.root + args.dataset + "/label.npy")
label = torch.LongTensor(label).to(device)
idx_train = np.load( args.root + args.dataset + "/train20.npy")
idx_val = np.load( args.root + args.dataset + "/val.npy")
idx_test = np.load( args.root + args.dataset + "/test.npy")
nb_classes = len(set(label))
num_node = len(label)


degrees = np.array(adj_.sum(axis=1)).flatten() + 1e-10
D_inv_sqrt = sp.diags(1. / np.sqrt(degrees))

# Compute the symmetric normalized Laplacian
I = sp.eye(adj_.shape[0])  # Identity matrix
L = I - D_inv_sqrt @ adj_ @ D_inv_sqrt
# Perform eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
def augment_adj_matrix(eigenvalues, eigenvectors, n_eigenvectors, frequency='low'):
    n_total = eigenvalues.shape[0]
    V_aug = np.zeros_like(eigenvectors[:, :n_total])

    # Indices for high-frequency and low-frequency eigenvectors
    high_freq_indices = np.arange(n_total // 2, n_total)
    low_freq_indices = np.arange(n_total // 2)

    if frequency == 'low':
        # Keep high-frequency eigenvectors, add back lowest from low-frequency
        selected_indices = np.concatenate([low_freq_indices[:n_eigenvectors], high_freq_indices])
    elif frequency == 'high':
        # Keep low-frequency eigenvectors, add back lowest from high-frequency
        selected_indices = np.concatenate([low_freq_indices, high_freq_indices[:n_eigenvectors]])
    else:
        raise ValueError("Frequency must be 'low' or 'high'")

    # Reconstruct the partial matrix using selected eigenvectors
    for i in selected_indices:
        V_aug += np.outer(eigenvectors[:, i], eigenvectors[:, i])

    # Apply the threshold to ignore very small edge values
    # V_aug_thresholded = np.where(np.abs(V_aug) > threshold, V_aug, 0)

    return V_aug
# Define the percentage of eigenvectors to use
percentage = args.ratio  # 20%
n_total = eigenvalues.shape[0]
n_eigenvectors = int(n_total * percentage)

# Generate the augmented matrix
V_ = augment_adj_matrix(eigenvalues, eigenvectors, n_eigenvectors, frequency=args.interval)

V = sp.csr_matrix(V_)
sele = sparse_mx_to_torch_sparse_tensor(V).float().to(device)
adj1 = sele
adj2 = adj.float()
model = Model(features.shape[1], out_ft)
model.to(device)
optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0)
cnt = 0
best = 1e9
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    loss = model(features, adj1, adj2)
    loss.backward()
    optimizer.step()
model.eval()
emb = model.get_emb(features, adj.float())

train_embs = emb[idx_train]
val_embs = emb[idx_val]
test_embs = emb[idx_test]

train_lbls = label[idx_train]
val_lbls = label[idx_val]
test_lbls = label[idx_test]

tot = torch.zeros(1)
tot = tot.to(device)
xent = nn.CrossEntropyLoss()
accs = []

for _ in range(50):
    log = LogReg(out_ft, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.0)
    log.to(device)

    pat_steps = 0
    best_acc = torch.zeros(1)
    best_acc = best_acc.cuda()
    for _ in range(150):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        
        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    tot += acc

print('Average accuracy:', tot / 50)
accs = torch.stack(accs)
print(accs.mean())
print(accs.std())
f=open(args.dataset+args.interval+str(args.ratio)+".txt","a")
f.write(str(accs.mean().cpu().data.numpy())+"\n")