# SLE_IAN project-level defaults
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT

DEFAULT = {
    "core_csv": WORKSPACE_ROOT / "dataset" / "public dataset and split" / "Bao2024.csv",
    "group_split_csv": WORKSPACE_ROOT
    / "dataset"
    / "public dataset and split"
    / "binary_group_split.csv",
    "exclude_types": ("Lab",),
    "split_group_col": "Drug-solvent system",
    "split_seed": 42,
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
    # Zero-overlap Drug-solvent system grouping
    "use_group_split": True,
    # Graph features
    "node_dim": 13,
    "edge_dim": 6,
    "cond_dim": 13,
    # Encoder
    "hidden_dim": 128,
    "gnn_layers": 3,
    "dropout": 0.1,
    # External Attention
    "ea_slots": 64,
    # Prediction head
    "mlp_hidden": 128,
    # Training
    "batch_size": 64,
    "eval_batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 200,
    "patience": 30,
    "grad_clip": 5.0,
    "num_workers": 0,
}

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_VERSION = "sle_ian_v2"


def run_dirs(experiment: str) -> dict[str, Path]:
    """Return checkpoint / results directories for an experiment name, creating them if needed."""
    ckpt = CHECKPOINT_DIR / experiment
    res = RESULTS_DIR / experiment
    ckpt.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    return {"checkpoint": ckpt, "results": res}
