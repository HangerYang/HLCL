import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.utils import degree,get_laplacian
from torch_geometric import transforms as T
from utility.data import dataset_split
import torch.nn as nn
import argparse
import random
from HLCL.models import HLCLConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora")
parser.add_argument('--low_first', action='store_true')
parser.add_argument('--model', type=str, default="GCN")
parser.add_argument('--high_pass', action='store_true')
parser.add_argument('--sep', action='store_true')
args, _ = parser.parse_known_args()
torch.manual_seed(42)
np.random.seed(42)

def separate_edges_by_label(edge_index, labels):
    same_label_edges = set()
    diff_label_edges = set()

    # Process existing edges
    for i in range(edge_index.size(1)):
        source, target = edge_index[:, i].tolist()
        if labels[source] == labels[target]:
            same_label_edges.add((source, target))
            same_label_edges.add((target, source))  # Add symmetric edge
        else:
            diff_label_edges.add((source, target))
            diff_label_edges.add((target, source))  # Add symmetric edge

    # Ensure all nodes are included in diff_label_edges
    all_nodes = set(range(len(labels)))
    for node in all_nodes:
        # If node has no edge in diff_label_edges, add one
        if not any(node in edge for edge in diff_label_edges):
            # Find nodes with a different label
            diff_label_nodes = [n for n in all_nodes if labels[n] != labels[node]]

            # Select a random node and add edges in both directions
            if diff_label_nodes:
                random_node = random.choice(diff_label_nodes)
                diff_label_edges.add((node, random_node))
                diff_label_edges.add((random_node, node))

    return torch.tensor(list(same_label_edges)).t(), torch.tensor(list(diff_label_edges)).t()
# Function to create train, validation, and test masks
def create_masks(node_labels, train_size=0.6, val_size=0.2):
    idx = np.arange(node_labels.shape[0])
    idx_train, idx_test = train_test_split(idx, test_size=1 - train_size, random_state=42, stratify=node_labels)
    idx_train, idx_val = train_test_split(idx_train, test_size=val_size / (train_size + val_size), random_state=42, stratify=node_labels[idx_train])

    train_mask = torch.zeros(node_labels.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(node_labels.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(node_labels.shape[0], dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return train_mask, val_mask, test_mask

def train():
    model.train()
    optimizer.zero_grad()
    if args.model == "GCN":
        out = model(X_reconstructed_k, data.edge_index, high_pass = args.high_pass)
    elif args.model == "MLP":
        out = model(X_reconstructed_k)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    loss.backward()
    optimizer.step()
    return loss.item(), val_loss.item()

# Evaluation function
def evaluate(mask):
    model.eval()
    if args.model == "GCN":
        out = model(X_reconstructed_k, data.edge_index, high_pass = args.high_pass)
    elif args.model == "MLP":
        out = model(X_reconstructed_k)
    pred = out[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc
# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# Define the GCN model
# class GCN(torch.nn.Module):
#     def __init__(self, num_features, hidden_dim, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, num_classes)

#     def forward(self, x, edge_index):
#         # x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)
    
def get_lap_degree(edge_index, size, device):
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(edge_index, normalization='sym', num_nodes=size)
    L_sym = torch.sparse_coo_tensor(edge_index_laplacian, edge_weight_laplacian, size=(size, size))
    node_degrees = degree(edge_index[0], num_nodes=size)
    D = torch.diag(node_degrees)
    D_tilde_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
    D_tilde_sqrt_inv = torch.diag(torch.sqrt(torch.diag(D)))
    L_sym = L_sym.to(device)
    D_tilde_sqrt_inv = D_tilde_sqrt_inv.to(device)
    D_tilde_sqrt = D_tilde_sqrt.to(device)
    return L_sym, D_tilde_sqrt, D_tilde_sqrt_inv

device = "cuda:5"
low_first = args.low_first
# Load the Cora dataset
# dataset = Planetoid(root='/data/', name='cora', transform=T.NormalizeFeatures())
dataset_name = args.dataset
data = dataset_split(dataset_name = dataset_name)
# Create masks
data.train_mask, data.val_mask, data.test_mask = create_masks(data.y, train_size=0.6, val_size=0.2)
if args.sep:
    same_label_edge_index, diff_label_edge_index = separate_edges_by_label(data.edge_index, data.y)
    L_sym_same, D_tilde_sqrt_same, D_tilde_sqrt_inv_same = get_lap_degree(same_label_edge_index, data.x.size(0), device)
    L_sym_diff, D_tilde_sqrt_diff, D_tilde_sqrt_inv_diff = get_lap_degree(diff_label_edge_index, data.x.size(0), device)
else:
    L_sym, D_tilde_sqrt, D_tilde_sqrt_inv = get_lap_degree(data.edge_index, data.x.size(0), device)

data = data.to(device)
sep_num = 2 if args.sep else 1
num_classes = data.y.max().item() + 1

for edx in range(sep_num):
    if sep_num > 1 and edx == 0:
        text = "same"
        L_sym = L_sym_same
        D_tilde_sqrt = D_tilde_sqrt_same
        D_tilde_sqrt_inv = D_tilde_sqrt_inv_same
        f=open("feature_study_{}_lf:{}_{}_sep:{}hp:_{}.txt".format(args.dataset, args.low_first, args.model, args.sep, args.high_pass),"a")
        f.write(f'SAME \n')
    elif sep_num > 1 and edx == 1:
        text = "diff"
        L_sym = L_sym_diff
        D_tilde_sqrt = D_tilde_sqrt_diff
        D_tilde_sqrt_inv = D_tilde_sqrt_inv_diff
        f=open("feature_study_{}_lf:{}_{}_sep:{}hp:_{}.txt".format(args.dataset, args.low_first, args.model, args.sep, args.high_pass),"a")
        f.write(f'DIFF \n')
    
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ratio = r # Select the fraction of low-frequency components you wish to keep
        eigenvalues, eigenvectors = torch.linalg.eigh(L_sym.to_dense(), UPLO='L')
        k = int(ratio * eigenvalues.shape[0]) 
        if low_first:
            U_k = eigenvectors[:, :k]
        else:
            U_k = eigenvectors[:, -k:]
        X_hat_k = U_k.T @ D_tilde_sqrt @ data.x
        X_reconstructed_k = D_tilde_sqrt_inv @ U_k @ X_hat_k        
        # Instantiate the model
        if args.model == "GCN":
            model = HLCLConv(input_dim=data.num_features, hidden_dim=16, output_dim=num_classes).to(device)
        elif args.model == "MLP":
            model = MLP(input_dim=data.num_features, hidden_dim=64, output_dim=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Early stopping parameters
        patience = 10  # Number of epochs to wait for improvement before stopping
        min_delta = 0.001  # Minimum change in the monitored quantity to qualify as an improvement
        best_val_loss = float('inf')
        counter = 0  # Counts the number of epochs without improvement
        # Train and evaluate with early stopping
        for epoch in range(200):
            loss, val_loss = train()
            train_acc = evaluate(data.train_mask)
            val_acc = evaluate(data.val_mask)
            test_acc = evaluate(data.test_mask)

            print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'Ratio: {ratio}, Epoch: {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                    print("Early stopping triggered")
                    f=open("feature_study_{}_lf:{}_{}_sep:{}hp:_{}.txt".format(args.dataset, args.low_first, args.model, args.sep, args.high_pass),"a")
                    f.write(f'Ratio: {ratio}, Test: {test_acc:.4f} \n')
                    break
        if counter < patience:
            f=open("feature_study_{}_lf:{}_{}_sep:{}hp:_{}.txt".format(args.dataset, args.low_first, args.model, args.sep, args.high_pass),"a")
            f.write(f'Ratio: {ratio}, Test: {test_acc:.4f} \n')
        

