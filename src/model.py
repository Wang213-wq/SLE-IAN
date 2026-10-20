"""SLE_IAN model: dual-stream GNN + External Attention + physics-grounded readout."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from config.project_config import DEFAULT
from src.encoder import TwinGraphEncoder
from src.external_attention import BidirectionalExternalAttention, masked_mean
from src.sle_loss import ln_x_to_logS


class SLE_IANModel(nn.Module):
    """
    Architecture: solute/solvent TwinGNN encoders → External Attention interaction → pool → readout.
    Two readout modes:
      - from_gamma: predict ln γ, then enforce hard SLE: ln x = ideal − ln γ → LogS
      - direct: predict targets directly, no physical constraint
    """

    def __init__(
        self,
        *,
        node_dim: int = DEFAULT["node_dim"],
        edge_dim: int = DEFAULT["edge_dim"],
        cond_dim: int = DEFAULT["cond_dim"],
        hidden_dim: int = DEFAULT["hidden_dim"],
        gnn_layers: int = DEFAULT["gnn_layers"],
        ea_slots: int = DEFAULT["ea_slots"],
        mlp_hidden: int = DEFAULT["mlp_hidden"],
        dropout: float = DEFAULT["dropout"],
        solvent_mode: str = "concat",  
        readout_mode: str = "direct", 
        ea_gated: bool = False,
        ea_gate_init: float = -2.0,
        use_ea: bool = True,
        use_edge_injection: bool = False,
    ) -> None:
        super().__init__()
        self.solvent_mode = solvent_mode
        self.hidden_dim = hidden_dim
        self.readout_mode = readout_mode
        self.use_ea = bool(use_ea)

        self.encoder = TwinGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            dropout=dropout,
            use_edge_injection=use_edge_injection,
        )
        self.ea = (
            BidirectionalExternalAttention(
                hidden_dim,
                num_slots=ea_slots,
                dropout=dropout,
                gated=ea_gated,
                gate_init=ea_gate_init,
            )
            if self.use_ea
            else None
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.ReLU(),
        )
        fuse_in = hidden_dim * 3

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        self.head_logS = nn.Linear(mlp_hidden, 1)
        self.head_ln_x = nn.Linear(mlp_hidden, 1)
        self.head_ln_gamma = nn.Linear(mlp_hidden, 1)

    def _encode_dense(self, graph_batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.encoder(graph_batch)
        return to_dense_batch(atoms, graph_batch.batch)

    def _pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(h, mask)

    def _merge_solvent(
        self,
        h1: torch.Tensor,
        m1: torch.Tensor,
        h2: torch.Tensor,
        m2: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.solvent_mode != "x1_weighted":
            return torch.cat([h1, h2], dim=1), torch.cat([m1, m2], dim=1)
        x1v = x1.view(-1, 1, 1).clamp(0.0, 1.0)
        x2v = 1.0 - x1v
        return torch.cat([h1 * x1v, h2 * x2v], dim=1), torch.cat([m1, m2], dim=1)

    def forward(
        self,
        solute_batch: Batch,
        solvent1_batch: Batch,
        solvent2_batch: Batch,
        cond: torch.Tensor,
        *,
        mw_solute: torch.Tensor | None = None,
        mw_solvent: torch.Tensor | None = None,
        ideal_ln_x: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Shared encoder applied to all three molecular graphs
        h_u, m_u = self._encode_dense(solute_batch)
        h_v1, m_v1 = self._encode_dense(solvent1_batch)
        h_v2, m_v2 = self._encode_dense(solvent2_batch)
        x1 = cond[:, 1]
        # Merge two solvents (x1-weighted if solvent_mode == "x1_weighted", else simple concat)
        h_v, m_v = self._merge_solvent(h_v1, m_v1, h_v2, m_v2, x1)

        # Bidirectional EA between solute and solvent representations
        if self.ea is not None:
            h_u, h_v = self.ea(h_u, m_u, h_v, m_v)
        # Graph-level pool + condition vector encoding
        z_u = self._pool(h_u, m_u)
        z_v = self._pool(h_v, m_v)
        z_c = self.cond_mlp(cond)

        # Fuse molecular reps with condition vector
        fused = self.fuse(torch.cat([z_u, z_v, z_c], dim=-1))

        # Physics-grounded readout: ln_gamma → ln_x = ideal_ln_x - ln_gamma → LogS
        if self.readout_mode == "from_gamma":
            # Predict ln_gamma from fused features
            if ideal_ln_x is None:
                raise ValueError("readout_mode=from_gamma requires ideal_ln_x")
            if mw_solute is None or mw_solvent is None:
                raise ValueError("readout_mode=from_gamma requires mw_solute / mw_solvent")
            ln_gamma = self.head_ln_gamma(fused).squeeze(-1)
            ln_x = ideal_ln_x - ln_gamma
            logS = ln_x_to_logS(ln_x, mw_solute, mw_solvent)
        else:
            # Direct head ablation: no physical constraint — predicts ln_x / logS / ln_gamma independently
            ln_x = self.head_ln_x(fused).squeeze(-1)
            logS = self.head_logS(fused).squeeze(-1)
            ln_gamma = self.head_ln_gamma(fused).squeeze(-1)
        return {
            "logS": logS,
            "ln_x": ln_x,
            "ln_gamma": ln_gamma,
            "fused": fused,
        }
