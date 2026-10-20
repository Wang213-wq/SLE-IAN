# Evaluation metrics: MAE / RMSE / R² / Pearson / hard-MAE
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.model import SLE_IANModel


@dataclass
class MetricBundle:
    n: int
    mae: float
    rmse: float
    r2: float
    pearson: float
    hard_mae: float = float("nan")
    hard_n: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def r2_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    # Coefficient of determination (R²) and Pearson correlation
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan"), float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    if y_true.size < 2 or float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return r2, pearson


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    # Compute regression metrics purely from ground-truth / prediction arrays (used to backfill history)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae, rmse = mae_rmse(y_true, y_pred)
    r2, pearson = r2_pearson(y_true, y_pred)
    hard_mask = y_true < 0.0
    hard_n = int(hard_mask.sum())
    if hard_n > 0:
        hard_mae, _ = mae_rmse(y_true[hard_mask], y_pred[hard_mask])
    else:
        hard_mae = float("nan")
    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "hard_mae": hard_mae,
        "hard_n": hard_n,
    }


@torch.no_grad()
def evaluate_loader(
    model: SLE_IANModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[MetricBundle, pd.DataFrame]:
    model.eval()
    ys, ps = [], []
    rows = []
    for batch in loader:
        solute = batch["solute"].to(device)
        s1 = batch["solvent1"].to(device)
        s2 = batch["solvent2"].to(device)
        cond = batch["cond"].to(device)
        out = model(
            solute,
            s1,
            s2,
            cond,
            mw_solute=batch["mw_solute"].to(device),
            mw_solvent=batch["mw_solvent"].to(device),
            ideal_ln_x=batch["ideal_ln_x"].to(device),
        )
        y = batch["logS"].numpy()
        p = out["logS"].detach().cpu().numpy()
        ys.append(y)
        ps.append(p)
        for i, rec in enumerate(batch["recs"]):
            rows.append(
                {
                    "row_idx": rec.idx,
                    "drug": rec.drug,
                    "system": rec.system,
                    "split": rec.split,
                    "logS_true": float(y[i]),
                    "logS_pred": float(p[i]),
                    "abs_err": float(abs(p[i] - y[i])),
                    "sle_valid": bool(rec.sle_valid),
                }
            )

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    mae, rmse = mae_rmse(y_all, p_all)
    r2, pearson = r2_pearson(y_all, p_all)
    hard_mask = y_all < 0.0
    hard_n = int(hard_mask.sum())
    if hard_n > 0:
        hard_mae, _ = mae_rmse(y_all[hard_mask], p_all[hard_mask])
    else:
        hard_mae = float("nan")
    bundle = MetricBundle(
        n=int(len(y_all)),
        mae=mae,
        rmse=rmse,
        r2=r2,
        pearson=pearson,
        hard_mae=hard_mae,
        hard_n=hard_n,
    )
    return bundle, pd.DataFrame(rows)


def save_metrics(bundle: MetricBundle, path: Path, *, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bundle.to_dict()
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
