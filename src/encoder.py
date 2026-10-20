# Graph encoder: GCNConv optionally with edge-feature residual injection
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv


class TwinGraphEncoder(nn.Module):
    """Twin encoder — solute and solvent streams share weights."""

    def __init__(
        self,
        *,
        node_dim: int = 13,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_edge_injection: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.layers = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            self.layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.edge_proj = nn.Linear(edge_dim, node_dim) if use_edge_injection else None

    def _inject_edge_features(self, batch: Batch) -> torch.Tensor:
        """Project / aggregate edge features to node-dim and inject as residual."""
        if batch.edge_attr is None or batch.edge_attr.numel() == 0:
            return batch.x
        if self.edge_proj is None:
            return batch.x

        x = batch.x
        edge_feat = self.edge_proj(batch.edge_attr)
        src = batch.edge_index[0].to(x.device)
        edge_feat = edge_feat.to(x.device)
        N = int(x.size(0))
        node_dim = edge_feat.size(1)
        agg = x.new_zeros(N, node_dim)
        agg = agg.index_add(0, src, edge_feat)
        ones = x.new_ones(src.size(0), 1)
        count = x.new_zeros(N, 1)
        count = count.index_add(0, src, ones)
        count = torch.clamp(count, min=1.0)
        agg = agg / count
        return x + agg

    def encode_atoms(self, batch: Batch) -> torch.Tensor:
        x = self._inject_edge_features(batch)
        edge_index = batch.edge_index
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.encode_atoms(batch)
