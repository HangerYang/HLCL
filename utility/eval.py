from re import split
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torch
from typing import Union

import torch
from torch import Tensor
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree

def evaluate_metrics(data, out, r):
    outputs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.cross_entropy(out[mask[r] == 1], data.y[mask[r] == 1]).item()
        pred = out[mask[r] == 1].max(dim=1)[1]

        outputs['{}_loss'.format(key)] = loss
        outputs['{}_f1'.format(key)] = f1_score(data.y[mask[r] == 1].cpu(), pred.data.cpu().numpy(), average='micro')
        outputs['{}_acc'.format(key)] = accuracy_score(data.y[mask[r] == 1].cpu(), pred.data.cpu().numpy())
    return outputs

#
# Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
#

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_cp=False, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_cp = save_cp
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.save_cp:
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss


def homophily(edge_index: Adj, y: Tensor, batch: OptTensor = None,
              method: str = 'edge') -> Union[float, Tensor]:
    r"""The homophily of a graph characterizes how likely nodes with the same
    label are near each other in a graph.
    There are many measures of homophily that fits this definition.
    In particular:

    - In the `"Beyond Homophily in Graph Neural Networks: Current Limitations
      and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper, the
      homophily is the fraction of edges in a graph which connects nodes
      that have the same class label:

      .. math::
        \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge y_v = y_w \} | }
        {|\mathcal{E}|}

      That measure is called the *edge homophily ratio*.

    - In the `"Geom-GCN: Geometric Graph Convolutional Networks"
      <https://arxiv.org/abs/2002.05287>`_ paper, edge homophily is normalized
      across neighborhoods:

      .. math::
        \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (w,v) : w
        \in \mathcal{N}(v) \wedge y_v = y_w \} |  } { |\mathcal{N}(v)| }

      That measure is called the *node homophily ratio*.

    - In the `"Large-Scale Learning on Non-Homophilous Graphs: New Benchmarks
      and Strong Simple Methods" <https://arxiv.org/abs/2110.14446>`_ paper,
      edge homophily is modified to be insensitive to the number of classes
      and size of each class:

      .. math::
        \frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, h_k - \frac{|\mathcal{C}_k|}
        {|\mathcal{V}|} \right),

      where :math:`C` denotes the number of classes, :math:`|\mathcal{C}_k|`
      denotes the number of nodes of class :math:`k`, and :math:`h_k` denotes
      the edge homophily ratio of nodes of class :math:`k`.

      Thus, that measure is called the *class insensitive edge homophily
      ratio*.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
        batch (LongTensor, optional): Batch vector\
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns
            each node to a specific example. (default: :obj:`None`)
        method (str, optional): The method used to calculate the homophily,
            either :obj:`"edge"` (first formula), :obj:`"node"` (second
            formula) or :obj:`"edge_insensitive"` (third formula).
            (default: :obj:`"edge"`)
    """
    assert method in {'edge', 'node', 'edge_insensitive'}
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index

    if method == 'edge':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        if batch is None:
            return float(out.mean())
        else:
            dim_size = int(batch.max()) + 1
            return scatter_mean(out, batch[col], dim=0, dim_size=dim_size)

    elif method == 'node':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        out = scatter_mean(out, col, 0, dim_size=y.size(0))
        if batch is None:
            return float(out.mean())
        else:
            return scatter_mean(out, batch, dim=0)

    elif method == 'edge_insensitive':
        assert y.dim() == 1
        num_classes = int(y.max()) + 1
        assert num_classes >= 2
        batch = torch.zeros_like(y) if batch is None else batch
        num_nodes = degree(batch, dtype=torch.int64)
        num_graphs = num_nodes.numel()
        batch = num_classes * batch + y

        h = homophily(edge_index, y, batch, method='edge')
        h = h.view(num_graphs, num_classes)

        counts = batch.bincount(minlength=num_classes * num_graphs)
        counts = counts.view(num_graphs, num_classes)
        proportions = counts / num_nodes.view(-1, 1)

        out = (h - proportions).clamp_(min=0).sum(dim=-1)
        out /= num_classes - 1
        return out if out.numel() > 1 else float(out)

    else:
        raise NotImplementedError