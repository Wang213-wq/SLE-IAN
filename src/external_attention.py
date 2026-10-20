# External Attention: learnable global memory keys Mk / values Mv (Guo et al.)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_slots: int = 64,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.residual = residual
        self.Mk = nn.Linear(dim, num_slots, bias=False)
        self.Mv = nn.Linear(num_slots, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.Mk(x)
        attn = F.softmax(logits, dim=-1)
        if mask is not None:
            attn = attn * mask.unsqueeze(-1).to(attn.dtype)
        out = self.Mv(attn)
        out = self.dropout(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        if self.residual:
            out = self.norm(x + out)
        else:
            out = self.norm(out)
        return out


class BidirectionalExternalAttention(nn.Module):
    """
    Bidirectional External Attention between solute and solvent streams.
    When gated=True: output = (1−α)·input + α·EA-interaction, with α learnable and initialised small
    (default gate_init≈−2 → sigmoid ≈ 0.12) to prevent the interaction branch from dominating early.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_slots: int = 64,
        dropout: float = 0.0,
        gated: bool = False,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.gated = gated
        self.shared = ExternalAttention(dim, num_slots=num_slots, dropout=dropout)
        self.cross_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        # sigmoid(gate_init ≈ -2) ≈ 0.12, conservative start — EA branch off by default
        self.ea_logit = nn.Parameter(torch.tensor(float(gate_init))) if gated else None

    def forward(
        self,
        h_solute: torch.Tensor,
        mask_solute: torch.Tensor,
        h_solvent: torch.Tensor,
        mask_solvent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_s0, h_v0 = h_solute, h_solvent
        ns = h_solute.size(1)
        nv = h_solvent.size(1)
        # Concatenate solute + solvent, run through shared EA, then split back by length
        h_cat = torch.cat([h_solute, h_solvent], dim=1)
        m_cat = torch.cat([mask_solute, mask_solvent], dim=1)
        h_ref = self.shared(h_cat, m_cat)
        h_s = h_ref[:, :ns]
        h_v = h_ref[:, ns : ns + nv]

        # Gated cross-pool: solute attends to pooled solvent, solvent to pooled solute
        s_pool = masked_mean(h_s, mask_solute)
        v_pool = masked_mean(h_v, mask_solvent)
        s_gate = torch.sigmoid(
            self.cross_proj(torch.cat([h_s, v_pool.unsqueeze(1).expand_as(h_s)], dim=-1))
        )
        v_gate = torch.sigmoid(
            self.cross_proj(torch.cat([h_v, s_pool.unsqueeze(1).expand_as(h_v)], dim=-1))
        )
        h_s = self.norm(h_s + s_gate * v_pool.unsqueeze(1))
        h_v = self.norm(h_v + v_gate * s_pool.unsqueeze(1))

        # Learnable gated residual mixing
        if self.ea_logit is not None:
            alpha = torch.sigmoid(self.ea_logit)
            h_s = (1.0 - alpha) * h_s0 + alpha * h_s
            h_v = (1.0 - alpha) * h_v0 + alpha * h_v
        return h_s, h_v


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-8)
