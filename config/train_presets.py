# Model presets: main model, architecture ablations, training-strategy ablations, Sol2026 fine-tuning

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from config.project_config import DEFAULT

PRESETS: dict[str, dict[str, Any]] = {
    # Main model
    "exp_geat_phys_group_edge_ea64": {
        "description": "Main model (SLE-IAN, group_split, ea64): edge-feature injection + gated EA + from_gamma hard SLE",
        "use_group_split": True,
        "val_hard_beta": 0.4,
        "hidden_dim": 128,
        "gnn_layers": 3,
        "ea_slots": 64,
        "ea_gated": True,
        "ea_gate_init": -2.0,
        "use_ea": True,
        "mlp_hidden": 128,
        "solvent_mode": "x1_weighted",
        "readout_mode": "from_gamma",
        "lambda_gamma": 1e-3,
        "use_edge_injection": True,
        "lr": 2e-4,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "eval_batch_size": 256,
        "max_epochs": 100,
        "patience": 25,
        "grad_clip": 5.0,
        "init_checkpoint": "checkpoints/external_sensitivity_group/32/best.pt",
    },

    # Architecture ablations
    # GNN_only (edge injection, ea64)
    "exp_ablate_gnn_only_group_edge_ea64": {
        "description": "Arch ablation · GNN_only (edge, ea64): GNN only + direct head",
        "use_group_split": True,
        "val_hard_beta": 0.4,
        "hidden_dim": 128,
        "gnn_layers": 3,
        "ea_slots": 64,
        "ea_gated": False,
        "ea_gate_init": -2.0,
        "use_ea": False,
        "mlp_hidden": 128,
        "solvent_mode": "x1_weighted",
        "readout_mode": "direct",
        "lambda_gamma": 0.0,
        "use_edge_injection": True,
        "lr": 5e-4,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "eval_batch_size": 256,
        "max_epochs": 300,
        "patience": 50,
        "grad_clip": 5.0,
        "init_checkpoint": None,
    },
    # GNN_SLE (edge injection, ea64)
    "exp_ablate_gnn_sle_group_edge_ea64": {
        "description": "Arch ablation · GNN_SLE (edge, ea64): GNN + hard SLE, no EA",
        "use_group_split": True,
        "val_hard_beta": 0.4,
        "hidden_dim": 128,
        "gnn_layers": 3,
        "ea_slots": 64,
        "ea_gated": False,
        "ea_gate_init": -2.0,
        "use_ea": False,
        "mlp_hidden": 128,
        "solvent_mode": "x1_weighted",
        "readout_mode": "from_gamma",
        "lambda_gamma": 1e-3,
        "use_edge_injection": True,
        "lr": 5e-4,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "eval_batch_size": 256,
        "max_epochs": 300,
        "patience": 50,
        "grad_clip": 5.0,
        "init_checkpoint": None,
    },
    # GEAT (edge injection, ea64)
    "exp_ablate_geat_group_edge_ea64": {
        "description": "Arch ablation · GEAT (edge, ea64): gated EA + direct head (no hard SLE)",
        "use_group_split": True,
        "val_hard_beta": 0.4,
        "hidden_dim": 128,
        "gnn_layers": 3,
        "ea_slots": 64,
        "ea_gated": True,
        "ea_gate_init": -2.0,
        "use_ea": True,
        "mlp_hidden": 128,
        "solvent_mode": "x1_weighted",
        "readout_mode": "direct",
        "lambda_gamma": 0.0,
        "use_edge_injection": True,
        "lr": 5e-4,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "eval_batch_size": 256,
        "max_epochs": 300,
        "patience": 50,
        "grad_clip": 5.0,
        "init_checkpoint": None,
    },

    # Training-strategy ablations (ea64)
    # E1: baseline — no edge injection, cold start
    "exp_geat_phys_group": {
        "description": "Train-strategy ablation · E1 (ea64): no edges · cold start",
        "use_group_split": True,
        "val_hard_beta": 0.4,
        "hidden_dim": 128,
        "gnn_layers": 3,
        "ea_slots": 64,
        "ea_gated": True,
        "ea_gate_init": -2.0,
        "use_ea": True,
        "mlp_hidden": 128,
        "solvent_mode": "x1_weighted",
        "readout_mode": "from_gamma",
        "lambda_gamma": 1e-3,
        "use_edge_injection": False,
        "lr": 5e-4,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "eval_batch_size": 256,
        "max_epochs": 300,
        "patience": 50,
        "grad_clip": 5.0,
        "init_checkpoint": None,
    },
    # E2 = +Edge injection · cold start
    "exp_ea_protocol_edge_ea64": {
        "description": "Train-strategy ablation · E2 (ea64): +Edge injection · cold start",
        "use_group_split": True,
        "val_hard_beta": 0.4,
        "hidden_dim": 128,
        "gnn_layers": 3,
        "ea_slots": 64,
        "ea_gated": True,
        "ea_gate_init": -2.0,
        "use_ea": True,
        "mlp_hidden": 128,
        "solvent_mode": "x1_weighted",
        "readout_mode": "from_gamma",
        "lambda_gamma": 1e-3,
        "use_edge_injection": True,
        "lr": 5e-4,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "eval_batch_size": 256,
        "max_epochs": 300,
        "patience": 50,
        "grad_clip": 5.0,
        "init_checkpoint": None,
    },
    # E3 = +Warm-start (Phase-I checkpoint) · no edges
    "exp_ea_protocol_warm_ea64": {
        "description": "Train-strategy ablation · E3 (ea64): +Warm start · no edges",
        "use_group_split": True,
        "val_hard_beta": 0.4,
        "hidden_dim": 128,
        "gnn_layers": 3,
        "ea_slots": 64,
        "ea_gated": True,
        "ea_gate_init": -2.0,
        "use_ea": True,
        "mlp_hidden": 128,
        "solvent_mode": "x1_weighted",
        "readout_mode": "from_gamma",
        "lambda_gamma": 1e-3,
        "use_edge_injection": False,
        "lr": 5e-4,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "eval_batch_size": 256,
        "max_epochs": 100,
        "patience": 25,
        "grad_clip": 5.0,
        "init_checkpoint": "checkpoints/external_sensitivity_group/32/best.pt",
    }
}


def resolve_preset(name: str | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT)
    cfg.update(
        {
            "solvent_mode": "concat",
            "readout_mode": "direct",
            "init_checkpoint": None,
            "ea_gated": False,
            "lambda_gamma": 0.0,
            "use_ea": True,
            "use_edge_injection": False,
        }
    )
    if not name:
        return cfg
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset {name!r}; available: {known}")
    cfg.update(PRESETS[name])
    cfg["experiment"] = name
    return cfg
