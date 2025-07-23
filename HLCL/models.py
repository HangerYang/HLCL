import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter
from torch_geometric.utils import get_laplacian,add_self_loops, degree, homophily
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from HLCL.utils import res_combine, representation_combine, union, intersect, edge_create_,representation_combine_supervised, res_combine_supervised
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from GCL.losses import Loss
from abc import ABC, abstractmethod
from torch import is_tensor

class Mix_Pass(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
    def forward(self, x, edge_index, edge_weight = None, high_pass = False):
        if high_pass:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))
            edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization="sym")
            
            x = self.lin(x)
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            out += self.bias
        else:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))
            # edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization="sym")
            edge_index, edge_weight = gcn_norm(edge_index, edge_weight,x.size(0), False, False)
            # Step 2: Linearly transform node feature matrix.
            x = self.lin(x)

            # Step 3: Compute normalization.
            # Step 4-5: Start propagating messages.
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

            # Step 6: Apply a final bias vector.
            out += self.bias
        return out
    def message(self, x_j, edge_weight=None):
        return edge_weight.view(-1, 1) * x_j
    
class HLCLConv_supervised(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, proj_dim, output_dim, activation=torch.nn.ReLU, num_layers=2, dropout=0.2):
        super(HLCLConv_supervised, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, output_dim)
        self.dropout = dropout
        self.activation = activation()
        self.num_layers = num_layers
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.layers = torch.nn.ModuleList()
        self.layers.append(Mix_Pass(input_dim, hidden_dim))
        for _ in range(self.num_layers - 2):
            self.layers.append(Mix_Pass(hidden_dim, hidden_dim))
        self.layers.append(Mix_Pass(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None, high_pass=False):
        z = x
        z = F.dropout(z, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.layers):
            # print(i)
            # print(conv.lin.shape)
            # print(conv.out_channels)
            if i != self.num_layers - 1:
                z = conv(z, edge_index, edge_weight, high_pass)
                z = self.activation(z)
                z = F.dropout(z, p=self.dropout, training=self.training)
        zs = conv(z, edge_index, edge_weight, high_pass)
        res = self.project(zs)
        # zs = self.log_softmax(zs)
        return zs, res
    def reset_parameters(self):
        for hlcl in self.layers:
            hlcl.reset_parameters()
    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        z = F.dropout(z, p=0.5, training=self.training)
        return F.log_softmax(self.fc2(z), dim=1)
    
    
class HLCLConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers, dropout=0.2):
        super(HLCLConv, self).__init__()
        self.dropout = dropout
        self.activation = activation()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(Mix_Pass(input_dim, hidden_dim))
        for _ in range(self.num_layers - 1):
            self.layers.append(Mix_Pass(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None, high_pass=False):
        z = x
        z = F.dropout(z, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight,high_pass)
            z = self.activation(z)
            if i != self.num_layers - 1:
                z = F.dropout(z, p=self.dropout, training=self.training)
        return z
    def reset_parameters(self):
        for hlcl in self.layers:
            hlcl.reset_parameters()
    
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

    def sample(self, anchor, sample, pos_mask = None, neg_mask = None, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        if pos_mask is None:
            pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        if neg_mask is None:
            neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask
    
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
                extra_pos_mask=None, extra_neg_mask=None, pos_mask=None, neg_mask=None):
        if not is_tensor(neg_mask):
            neg_mask = None
        if self.mode == 'L2L':
            anchor1, sample1, pos_mask1, neg_mask1= self.sampler(anchor=h1, sample=h2, pos_mask = pos_mask, neg_mask = neg_mask)
            anchor2, sample2, pos_mask2, neg_mask2= self.sampler(anchor=h2, sample=h1, pos_mask = pos_mask, neg_mask = neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5
    
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder 
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, args, data, edges = False, device=None):
        aug1, aug2, aug3, aug4 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(data.x, data.low_edge_index, data.edge_weight)
        x2, edge_index2, edge_weight2 = aug2(data.x, data.high_edge_index, data.edge_weight)

        x3, edge_index3, edge_weight3 = aug3(data.x, data.low_edge_index, data.edge_weight)
        x4, edge_index4, edge_weight4 = aug4(data.x, data.high_edge_index, data.edge_weight)
        # edge_index0 = add_edge(edge_index, 0.5)
        # z = self.encoder(x, edge_index0, edge_weight0)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        z3 = self.encoder(x3, edge_index3, edge_weight3, high_pass=True)
        z4 = self.encoder(x4, edge_index4, edge_weight4, high_pass=True)
        z = torch.cat((z2,z3),dim=1)
        if edges:
            if args.infer_combine_x:
                low_edges, high_edges, _, _ = edge_create_(args, z, data.edge_index, device)
            else:
                low_edges, high_edges, _, _ = edge_create_(args, z1, data.edge_index, device)
            data.low_edge_index = low_edges
            data.high_edge_index = high_edges
            data.low_edge_weight = torch.ones(data.low_edge_index.shape[1]).to(device)
            data.high_edge_weight = torch.ones(data.high_edge_index.shape[1]).to(device)
            return representation_combine(args, z1, z2, z3, z4), data
        else:
            return representation_combine(args, z1, z2, z3, z4)
    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)



class SepEncoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim, high_pass=False):
        super(SepEncoder, self).__init__()
        self.encoder = encoder 
        self.augmentor = augmentor
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)
        self.high_pass = high_pass

    def forward(self, data):
        aug1, aug2 = self.augmentor
        if self.high_pass:
            x, edge_index, edge_weight = aug1(data.x, data.high_edge_index, data.edge_weight)
        else:
            x, edge_index, edge_weight = aug2(data.x, data.low_edge_index, data.edge_weight)
        z = self.encoder(x, edge_index, edge_weight,high_pass=self.high_pass)
        return z
    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)