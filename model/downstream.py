import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.models.attentive_fp import GATEConv

class FBGCN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.high = nn.Linear(in_dim, out_dim, False) 
        self.low = nn.Linear(in_dim, out_dim, False) 
        self.aL = nn.Parameter(torch.tensor(1.))
        self.aH = nn.Parameter(torch.tensor(1.))
               
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.high.weight, gain)
        nn.init.xavier_normal_(self.low.weight, gain)
        self.aL = nn.Parameter(torch.tensor(1.))
        self.aH = nn.Parameter(torch.tensor(1.))

    def forward(self, x,Lsym,Anorm):
        Lhp = Lsym
        Hh = F.relu(torch.mm(Lhp, self.high(x)))  
        Llp = Anorm
        Hl = F.relu(torch.mm(Llp, self.low(x)))   
        # return (self.aL * Hl + self.aH * Hh)
        return (Hl + Hh)

class FBGCN(nn.Module):
    def __init__(self, n_layer, in_dim, hi_dim, out_dim, dropout):
        """
        :param n_layer: number of layers
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super().__init__()
        assert(n_layer > 0)
        self.num_layers = n_layer
        self.stacks = nn.ModuleList()
        # first layer
        self.stacks.append(FBGCN_Layer(in_dim, hi_dim))
        # inner layers
        # for _ in range(n_layer - 2):
        #     self.stacks.append(FBGCN_Layer(hi_dim, hi_dim)
        # last layer
        self.stacks.append(FBGCN_Layer(hi_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for fbgcn in self.stacks:
            fbgcn.reset_parameters()
    def forward(self, x, lsym, anorm,a=0.5,b=0.5):
        # first layer
        x = F.relu(self.stacks[0](x, lsym, anorm))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # inner layers
        # if self.num_layers > 2:
        #     for layer in range(self.num_layers - 1):
        #          x = F.relu(self.stacks[layer](x, edge_index,lsym, anorm))
        #          x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer
        return F.log_softmax(self.stacks[-1](x, lsym, anorm), dim=1)

class FBGCN_Head(nn.Module):
    def __init__(self, n_layer, in_dim, hi_dim, sec_hid_dim, out_dim, dropout):
        """
        :param n_layer: number of layers
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super().__init__()
        assert(n_layer > 0)
        self.num_layers = n_layer
        self.stacks = nn.ModuleList()
        # first layer
        self.stacks.append(FBGCN_Layer(in_dim, hi_dim))
        # inner layers
        # for _ in range(n_layer - 2):
        #     self.stacks.append(FBGCN_Layer(hi_dim, hi_dim)
        # last layer
        self.stacks.append(FBGCN_Layer(hi_dim, sec_hid_dim))
        self.stacks.append(nn.Linear(sec_hid_dim, out_dim))
        # self.stacks.append(nn.Linear(64, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for fbgcn in self.stacks:
            fbgcn.reset_parameters()
    def forward(self, x, lsym, anorm):
        # first layer
        x = F.relu(self.stacks[0](x,lsym, anorm))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.stacks[1](x,lsym, anorm)
        # x = self.stacks[2](x)
        # inner layers
        # if self.num_layers > 2:
        #     for layer in range(self.num_layers - 1):
        #          x = F.relu(self.stacks[layer](x, edge_index,replace, anorm))
        #          x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer
        return F.log_softmax(self.stacks[-1](x), dim=1)


class GCN(nn.Module):
    def __init__(self, n_layer, in_dim, hi_dim, out_dim, dropout):
        """
        :param n_layer: number of layers
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super(GCN, self).__init__()
        assert (n_layer > 0)

        self.num_layers = n_layer
        self.gcns = nn.ModuleList()
        # first layer
        self.gcns.append(GCNConv(in_dim, hi_dim))
        # inner layers
        # for _ in range(n_layer - 2):
        #     self.gcns.append(GCNConv(hi_dim, hi_dim))
        # last layer
        self.gcns.append(GCNConv(hi_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()
        self.activation = torch.nn.PReLU()
    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    def forward(self, x, edge_index):
        # first layer
        x = self.activation(self.gcns[0](x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # inner layers
        # if self.num_layers > 2:
        #     for layer in range(1, self.num_layers - 1):
        #         x = F.relu(self.gcns[layer](x, edge_index))
        #         x = F.dropout(x, p=self.dropout, training=self.training)

        # last layer
        return F.log_softmax(self.activation(self.gcns[self.num_layers - 1](x, edge_index)), dim = 1)
class GAT(nn.Module):
    def __init__(self, n_layer, in_dim, hi_dim, out_dim, dropout):
        """
        :param n_layer: number of layers
        :param in_dim: input dimension
        :param hi_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout ratio
        """
        super(GAT, self).__init__()
        assert (n_layer > 0)

        self.num_layers = n_layer
        self.gats = nn.ModuleList()
        # first layer
        self.gats.append(GATConv(in_dim, hi_dim, 8))
        # inner layers
        # for _ in range(n_layer - 2):
        #     self.gcns.append(GCNConv(hi_dim, hi_dim))
        # last layer
        self.gats.append(GATConv(hi_dim * 8, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for gat in self.gats:
            gat.reset_parameters()

    def forward(self, x, edge_index):
        # first layer
        x = F.relu(self.gats[0](x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # inner layers
        # if self.num_layers > 2:
        #     for layer in range(1, self.num_layers - 1):
        #         x = F.relu(self.gats[layer](x, edge_index))
        #         x = F.dropout(x, p=self.dropout, training=self.training)

        # last layer
        return F.log_softmax(self.gats[self.num_layers - 1](x, edge_index), dim = 1)
    