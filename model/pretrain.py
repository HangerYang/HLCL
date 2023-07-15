import torch
import torch.nn as nn
import torch.nn.functional as F
import GCL.augmentors as A

device = torch.device("cpu")

class Pre_Mix_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.encoder = nn.Linear(in_dim, out_dim, bias = False)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.encoder.weight, gain)

    def forward(self, x, Lsym, Anorm, encode="low"):
        if encode == "high":
            Lhp = Lsym
            Hh = F.relu(torch.mm(Lhp, self.encoder(x)))      
            return Hh
        else:
            Llp = Anorm
            Hl = F.relu(torch.mm(Llp, self.encoder(x)))      
            return Hl

class Pre_Train(torch.nn.Module):
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
        self.stacks = torch.nn.ModuleList()
        # first layer
        self.stacks.append(Pre_Mix_Layer(in_dim, hi_dim))
        # inner layers
        # for _ in range(n_layer - 2):
        #     self.stacks.append(Pre_Mix_Layer(hi_dim, hi_dim))
        # last layer
        self.stacks.append(Pre_Mix_Layer(hi_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for hplayer in self.stacks:
            hplayer.reset_parameters()

    def forward(self, x, lsym, anorm, encode="low"):
        if encode == "high":
            x = F.relu(self.stacks[0](x, lsym, anorm, encode))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.stacks[-1](x, lsym, anorm, encode)
        else:
            x = F.relu(self.stacks[0](x, lsym, anorm, encode))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.stacks[-1](x, lsym, anorm, encode)


class Pre_HighPass_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.high = nn.Linear(in_dim, out_dim, bias = False)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.high.weight, gain)

    def forward(self, x,Lsym):
        Lhp = Lsym
        Hh = F.relu(torch.mm(Lhp, self.high(x)))      
        return Hh

class Pre_LowPass_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.low = nn.Linear(in_dim, out_dim, bias = False)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.low.weight, gain)

    def forward(self, x, Anorm):
        Llp = Anorm
        Hl = F.relu(torch.mm(Llp, self.low(x)))      
        return Hl

class Pre_HighPass(nn.Module):
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
        self.stacks.append(Pre_HighPass_Layer(in_dim, hi_dim))
        # inner layers
        # for _ in range(n_layer - 2):
        #     self.stacks.append(Pre_HighPass_Layer(hi_dim, hi_dim))
        # last layer
        self.stacks.append(Pre_HighPass_Layer(hi_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for hplayer in self.stacks:
            hplayer.reset_parameters()
    def forward(self, x, lsym):
        # first layer
        x = F.relu(self.stacks[0](x, lsym))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # inner layers
        # if self.num_layers > 2:
        #     for layer in range(self.num_layers - 1):
        #          x = F.relu(self.stacks[layer](x, lsym))
        #          x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer
        return self.stacks[-1](x, lsym)
class Pre_LowPass(nn.Module):
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
        self.stacks.append(Pre_LowPass_Layer(in_dim, hi_dim))
        # inner layers
        # for _ in range(n_layer - 2):
        #     self.stacks.append(Pre_LowPass_Layer(hi_dim, hi_dim))
        # last layer
        self.stacks.append(Pre_LowPass_Layer(hi_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lplayer in self.stacks:
            lplayer.reset_parameters()
    def forward(self, x, anorm):
        # first layer
        x = F.relu(self.stacks[0](x, anorm))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # inner layers
        # if self.num_layers > 2:
        #     for layer in range(self.num_layers - 1):
        #          x = F.relu(self.stacks[layer](x, anorm))
        #          x = F.dropout(x, p=self.dropout, training=self.training)
        # last layer
        return self.stacks[-1](x, anorm)

def pt_model(high_model, low_model, contrast_model, optimizer, data):
    high_model.train()
    low_model.train()
    optimizer.zero_grad()
    # h1 = high_model(drop_feature(data.x, 0.2), data.lsym)
    # h2 = low_model(drop_feature(data.x, 0.2), data.anorm)

    h1 = high_model(data.x, data.lsym)
    h2 = low_model(data.x, data.anorm)
    loss = contrast_model(h1=h1, h2=h2)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def get_augmentor(augmentor, one_side=False, side=None, aug_ratio=0.5, aug2_ratio = 0.5):
    high_pass_ratio = aug_ratio
    high_pass_ratio2 = aug2_ratio
    low_pass_ratio = aug_ratio
    low_pass_ratio2 = aug2_ratio
    if (one_side):
        if side == "high":
            low_pass_ratio = 0.
            low_pass_ratio2 = aug2_ratio
        else:
            high_pass_ratio = 0.
            high_pass_ratio2
    if augmentor == "FM":
        return (A.FeatureMasking(high_pass_ratio), A.FeatureMasking(low_pass_ratio))
    if augmentor == "ER":
        return (A.EdgeRemoving(high_pass_ratio), A.EdgeRemoving(low_pass_ratio))
    if augmentor == "ND":
        return (A.NodeDropping(high_pass_ratio), A.NodeDropping(low_pass_ratio))
    if augmentor == "FD":
        if not one_side:
            return (A.FeatureDropout(high_pass_ratio), A.FeatureDropout(low_pass_ratio))
        elif side == "high":
            return (A.FeatureDropout(high_pass_ratio), A.FeatureDropout(1.))
        else:
            return (A.FeatureDropout(1.), A.FeatureDropout(low_pass_ratio))
    if augmentor == "COMP":
        return (A.Compose([A.EdgeRemoving(pe=high_pass_ratio), A.FeatureMasking(pf=high_pass_ratio2)]), A.Compose([A.EdgeRemoving(pe=low_pass_ratio), A.FeatureMasking(pf=low_pass_ratio2)]))
        


