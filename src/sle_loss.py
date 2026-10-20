# SLE ideal solubility + LogS physics-grounded readout + training loss (data term + optional lnγ shrinkage)
from __future__ import annotations

import numpy as np
import torch

R_J_PER_MOL_K = 8.314462618


def ideal_ln_x_np(T_K: float, Tm_K: float, Hfus_J_per_mol: float) -> float:
    if not np.isfinite(T_K) or not np.isfinite(Tm_K) or not np.isfinite(Hfus_J_per_mol):
        return np.nan
    if T_K <= 0 or Tm_K <= 0 or T_K >= Tm_K:
        return np.nan
    return (Hfus_J_per_mol / R_J_PER_MOL_K) * (1.0 / Tm_K - 1.0 / T_K)


def data_mae_loss(
    pred_logS: torch.Tensor,
    target_logS: torch.Tensor,
) -> torch.Tensor:
    return (pred_logS - target_logS).abs().mean()


def ln_x_to_logS(
    ln_x: torch.Tensor,
    mw_solute: torch.Tensor,
    mw_solvent: torch.Tensor,
) -> torch.Tensor:
    # Ideal solubility: ln(x_ideal) = ΔHfus/R · (1/Tm - 1/T)
    x = torch.exp(ln_x).clamp(1e-12, 1.0 - 1e-12)
    g = 100.0 * mw_solute * x / (mw_solvent * (1.0 - x)).clamp_min(1e-12)
    return torch.log10(g.clamp_min(1e-30))


def total_loss(
    pred_logS: torch.Tensor,
    target_logS: torch.Tensor,
    ln_gamma: torch.Tensor,
    *,
    lambda_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss = LogS MAE data term."""
    l_data = data_mae_loss(pred_logS, target_logS)
    l_gamma = pred_logS.new_zeros(())
    if lambda_gamma > 0:
        l_gamma = ln_gamma.pow(2).mean()
    loss = l_data + float(lambda_gamma) * l_gamma
    stats = {
        "loss": float(loss.detach().cpu()),
        "L_data": float(l_data.detach().cpu()),
        "L_gamma": float(l_gamma.detach().cpu()),
        "lambda_gamma": float(lambda_gamma),
    }
    return loss, stats
