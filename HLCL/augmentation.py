import torch
import networkx as nx
import torch.nn.functional as F

from typing import Optional
from GCL.utils import normalize
from torch_sparse import SparseTensor, coalesce
from torch_scatter import scatter
from torch_geometric.transforms import GDC
from torch.distributions import Uniform, Beta
from torch_geometric.utils import dropout_adj, to_networkx, to_undirected, degree, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, sort_edge_index, add_self_loops, subgraph
from torch.distributions.bernoulli import Bernoulli
from GCL.augmentors.augmentor import Graph, Augmentor

class EdgeAdding(Augmentor):
    def __init__(self, pe=0.2):
        super(EdgeAdding, self).__init__()
        self.pe=pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        edge_index = add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)


class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True):
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_ppr(
            edge_index, edge_weights,
            alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res
    
def add_edge(edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
    
    num_edges = edge_index.size()[1]
    num_nodes = edge_index.max().item() + 1
    num_add = int(num_edges * ratio)

    new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    N = edge_index.max().item() + 1
    edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device)

    return coalesce(edge_index, edge_weight, N, N)[0]
def compute_ppr(edge_index, edge_weight=None, alpha=0.2, eps=0.1, ignore_edge_attr=True, add_self_loop=True):
    N = edge_index.max().item() + 1
    if ignore_edge_attr or edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device)
    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')
    diff_mat = GDC().diffusion_matrix_exact(
        edge_index, edge_weight, N, method='ppr', alpha=alpha)
    edge_index, edge_weight = GDC().sparsify_dense(diff_mat, method='threshold', eps=eps)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')

    return edge_index, edge_weight