# Sol2026 external-dataset zero-shot validation (loads Bao2024-trained weights)
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from config.console import configure_console_utf8
from config.project_config import DEFAULT

configure_console_utf8()

from src.data import (
    build_datasets,
    collate_batch,
    filter_core,
    load_core_table,
    parse_records,
)
from src.evaluate import evaluate_loader, save_metrics
from src.model import SLE_IANModel


def build_model_from_ckpt(ckpt_path: Path, device: torch.device) -> SLE_IANModel:
    # Rebuild model from checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck = state.get("config", {})

    model = SLE_IANModel(
        hidden_dim=ck.get("hidden_dim", 128),
        gnn_layers=ck.get("gnn_layers", 3),
        ea_slots=ck.get("ea_slots", 64),
        mlp_hidden=ck.get("mlp_hidden", 128),
        dropout=ck.get("dropout", 0.15),
        solvent_mode=ck.get("solvent_mode", "x1_weighted"),
        readout_mode=ck.get("readout_mode", "from_gamma"),
        ea_gated=ck.get("ea_gated", True),
        ea_gate_init=ck.get("ea_gate_init", -2.0),
        use_ea=ck.get("use_ea", True),
        use_edge_injection=ck.get("use_edge_injection", True),
    ).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    print(f"[load] {ckpt_path} | ep={state.get('epoch', '?')} best_val_MAE={state.get('best_val_mae', '?')}", flush=True)
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="External-dataset validation (Bao2024 weights → Sol2026 full-set prediction)")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "external_sensitivity_group_edge" / "64" / "best.pt",
        help="Bao2024-trained best.pt",
    )
    p.add_argument(
        "--core-csv",
        type=Path,
        default=DEFAULT["core_csv"].parent.parent / "private dataset and split" / "Sol2026.csv",
        help="External-dataset CSV (columns already aligned)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Load Bao2024-trained weights
    model = build_model_from_ckpt(args.checkpoint, device)

    # 2. Load external dataset (entire Sol2026 as test)
    df = load_core_table(args.core_csv)
    df = filter_core(df.copy())
    records = parse_records(df)
    print(f"[data] external {args.core_csv.name}: {len(records)} records", flush=True)
    if not records:
        raise RuntimeError("parse_records returned empty — check CSV columns")

    # Treat everything as test (no train/val/test split)
    from src.data import GEATDataset
    test_ds = GEATDataset(records)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # 3. Predict + metrics
    bundle, pred_df = evaluate_loader(model, loader, device)
    print(
        f"[external] MAE={bundle.mae:.4f} RMSE={bundle.rmse:.4f} "
        f"R2={bundle.r2:.4f} Pearson={bundle.pearson:.4f} "
        f"hard_MAE={bundle.hard_mae:.4f} n={bundle.n}",
        flush=True,
    )

    # 4. Save
    out_dir = ROOT / "results" / "external_validation_sol2026"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions_external.csv", index=False, encoding="utf-8-sig")
    save_metrics(
        bundle,
        out_dir / "metrics_external.json",
        extra={
            "checkpoint": str(args.checkpoint),
            "external_csv": str(args.core_csv),
            "n_samples": len(records),
        },
    )
    print(f"[done] → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
