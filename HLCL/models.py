import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter
from torch_geometric.utils import get_laplacian,add_self_loops, degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from HLCL.utils import res_combine, representation_combine
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
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
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))
            edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization="sym")
            
            x = self.lin(x)
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            out += self.bias
        else:
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))
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
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder 
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, args, x, lp_edge_index, hp_edge_index, lp_edge_weight, hp_edge_weight, origin_edge_index = None, edges = False, device=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, lp_edge_index, lp_edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, hp_edge_index, hp_edge_weight)
        # edge_index0 = add_edge(edge_index, 0.5)
        # z = self.encoder(x, edge_index0, edge_weight0)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2, high_pass = True)

        if edges:
            return res_combine(args, device, origin_edge_index, args.low_k, args.high_k, z1, z2)
        else:
            return representation_combine(args, z1, z2)
    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

class GCN(torch.nn.Module):
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
        self.gcns = torch.nn.ModuleList()
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