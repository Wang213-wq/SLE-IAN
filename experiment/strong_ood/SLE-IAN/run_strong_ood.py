# SLE_IAN strong-OOD four-mode suite (solvent / solute / scaffold / temperature) — trained on frozen split_indices tables
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # SLE_IAN/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.console import configure_console_utf8

configure_console_utf8()

import torch
import torch_geometric

torch_geometric.typing.WITH_TORCH_SCATTER = False

from config.project_config import CHECKPOINT_DIR
from config.train_presets import resolve_preset
from src.train import cfg_from_preset, train_model

OOD_MODES = ("solvent", "solute", "scaffold", "temperature")
# Neutral frozen splits — built & verified by data_split/build_strong_ood_splits.py
STRONG_OOD_SPLITS = ROOT / "dataset" / "strong_ood_splits"
OUT_JSON = ROOT / "results" / "strong_ood" / "ood_phys.json"

# Default recipe: main model preset (edge injection + ea_slots=64), fine-tuned from the main model checkpoint
DEFAULT_PRESET = "exp_geat_phys_group_edge_ea64"
DEFAULT_EA_SLOTS = 64
DEFAULT_INIT_CKPT = ROOT / "checkpoints" / "external_sensitivity_group_edge" / "64" / "best.pt"


def _pick_resume_ckpt(mode: str, *, min_epoch: int, suffix: str = "") -> Path | None:
    """Prefer last.pt; fall back to *_backup.pt if the existing ckpt is too short (cold-start overwrite)."""
    d = CHECKPOINT_DIR / f"exp_geat_phys_ood_{mode}{suffix}"
    candidates = [
        d / "last.pt",
        d / "best.pt",
    ]
    best: tuple[int, Path] | None = None
    for p in candidates:
        if not p.is_file():
            continue
        try:
            s = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        ep = int(s.get("epoch", 0))
        if ep < min_epoch:
            continue
        if best is None or ep > best[0]:
            best = (ep, p)
    return None if best is None else best[1]


def train_one(
    mode: str,
    *,
    device: str,
    max_epochs: int,
    patience: int,
    resume: bool,
    resume_min_epoch: int,
    preset_name: str = DEFAULT_PRESET,
    ea_slots: int = DEFAULT_EA_SLOTS,
    init_checkpoint: str | None = None,
) -> dict:
    split_csv = STRONG_OOD_SPLITS / mode / "split_indices.csv"
    if not split_csv.is_file():
        raise FileNotFoundError(split_csv)

    preset = resolve_preset(preset_name)
    cfg = cfg_from_preset(preset, device=device)
    # Strong-OOD uses frozen split_indices, not group_split tables
    cfg.use_group_split = False
    cfg.split_indices_csv = str(split_csv.resolve())
    cfg.split_mode_name = mode
    # ea_slots + optional _edge suffix for output directory
    cfg.ea_slots = int(ea_slots)
    is_edge = "edge" in preset_name.lower()
    exp_suffix = f"_group_ea{ea_slots}" + ("_edge" if is_edge else "")
    cfg.experiment = f"exp_geat_phys_ood_{mode}{exp_suffix}"
    # Fine-tune from sensitivity best checkpoint (optional)
    if init_checkpoint:
        ic = Path(init_checkpoint)
        if not ic.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {ic}")
        cfg.init_checkpoint = str(ic.resolve())
        print(f"  [init] {mode}: fine-tune from {ic.name}", flush=True)
    else:
        cfg.init_checkpoint = None
    cfg.max_epochs = int(max_epochs)
    cfg.patience = int(patience)
    cfg.resume_checkpoint = None

    if resume:
        ck = _pick_resume_ckpt(mode, min_epoch=resume_min_epoch, suffix=exp_suffix)
        if ck is not None:
            cfg.resume_checkpoint = str(ck.resolve())
        else:
            print(
                f"  [resume] {mode}: no usable checkpoint (epoch ≥ {resume_min_epoch}); cold start instead",
                flush=True,
            )

    print(f"\n===== SLE-IAN strong-OOD | {mode} (group_split, ea_slots={ea_slots}) =====", flush=True)
    print(
        f"  split_indices={cfg.split_indices_csv} | "
        f"max_epochs={cfg.max_epochs} patience={cfg.patience} | "
        f"init_ckpt={cfg.init_checkpoint or 'None'} | "
        f"resume={cfg.resume_checkpoint or 'None'}",
        flush=True,
    )
    summary = train_model(cfg)
    lb = summary.get("leaderboard") or {}
    metrics = (summary.get("metrics") or {}).get("test") or {}
    return {
        "experiment": cfg.experiment,
        "mode": mode,
        "preset": preset_name,
        "ea_slots": cfg.ea_slots,
        "init_checkpoint": cfg.init_checkpoint,
        "test_MAE_LogS": lb.get("test_MAE_LogS", metrics.get("mae")),
        "test_hard_MAE_LogS": lb.get("test_hard_MAE_LogS", metrics.get("hard_mae")),
        "n_test": lb.get("n_test", metrics.get("n")),
        "best_epoch": summary.get("best_epoch"),
        "best_val_mae": summary.get("best_val_mae"),
        "stopped_early": summary.get("stopped_early"),
        "resume_from": cfg.resume_checkpoint,
        "results_dir": str((ROOT / "results" / cfg.experiment).resolve()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SLE-IAN strong OOD suite (main model: edge injection + ea_slots=64)")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(OOD_MODES),
        choices=list(OOD_MODES),
    )
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last.pt (or *_backup.pt) instead of cold start",
    )
    p.add_argument(
        "--resume-min-epoch",
        type=int,
        default=50,
        help="Treat a checkpoint older than this epoch as unusable (avoids cold-start overwrites)",
    )
    p.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"Base preset (default: {DEFAULT_PRESET})",
    )
    p.add_argument(
        "--ea-slots",
        type=int,
        default=DEFAULT_EA_SLOTS,
        help=f"EA memory size (default: {DEFAULT_EA_SLOTS}, chosen by sensitivity analysis)",
    )
    p.add_argument(
        "--init-checkpoint",
        default=str(DEFAULT_INIT_CKPT),
        help=f"Fine-tune starting weights (default: {DEFAULT_INIT_CKPT}); empty string to disable",
    )
    p.add_argument("--yes", action="store_true", help="Confirm training start")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve init_checkpoint — empty string means cold start
    init_ckpt = args.init_checkpoint if args.init_checkpoint else None
    if init_ckpt and not Path(init_ckpt).is_file():
        print(f"[warn] init_checkpoint not found, switching to cold start: {init_ckpt}", flush=True)
        init_ckpt = None

    is_edge_preset = "edge" in args.preset.lower()
    exp_suffix = f"_group_ea{args.ea_slots}" + ("_edge" if is_edge_preset else "")
    plan = []
    for mode in args.modes:
        ck = _pick_resume_ckpt(mode, min_epoch=args.resume_min_epoch, suffix=exp_suffix) if args.resume else None
        plan.append(
            {
                "mode": mode,
                "preset": args.preset,
                "ea_slots": args.ea_slots,
                "init_checkpoint": init_ckpt,
                "resume_ckpt": str(ck) if ck else None,
            }
        )
    print(
        json.dumps(
            {
                "protocol": "strong OOD suite (main model preset + main model ckpt fine-tune)",
                "schedule": {
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                    "resume": args.resume,
                    "resume_min_epoch": args.resume_min_epoch,
                },
                "plan": plan,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not args.yes:
        print(
            "[dry-run] Examples:\n"
            "  python experiment/strong_ood/SLE-IAN/run_strong_ood.py --device cuda --max-epochs 1000 "
            "--patience 100 --yes\n"
            "  Resume: python experiment/strong_ood/SLE-IAN/run_strong_ood.py --device cuda "
            "--max-epochs 1000 --patience 100 --resume --yes",
            flush=True,
        )
        return

    rows = []
    for mode in args.modes:
        phys = train_one(
            mode,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            resume=args.resume,
            resume_min_epoch=args.resume_min_epoch,
            preset_name=args.preset,
            ea_slots=args.ea_slots,
            init_checkpoint=init_ckpt,
        )
        row = {
            "mode": mode,
            "n_test": phys.get("n_test"),
            "SLE_IAN_MAE": phys.get("test_MAE_LogS"),
            "SLE_IAN_hard_MAE": phys.get("test_hard_MAE_LogS"),
            "phys": phys,
        }
        rows.append(row)
        print(f"[result] {mode}: SLE_IAN={row['SLE_IAN_MAE']}", flush=True)

    payload = {
        "protocol": {
            "name": "strong_ood_paper_suite_group_ea64",
            "meaning": "solvent / solute / scaffold / temperature holdout",
            "phys_recipe": f"{args.preset} (ea_slots={args.ea_slots}), fine-tune from main model ckpt",
            "schedule": {
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "resume": args.resume,
            },
            "fairness": "same row indices as dataset/strong_ood_splits/<mode>/split_indices.csv",
        },
        "rows": rows,
    }
    out_json = OUT_JSON
    if is_edge_preset:
        # Distinguish by ea_slots × edge injection to avoid clobbering results
        out_json = ROOT / "results" / f"ood_strong_phys_group_ea{args.ea_slots}_edge.json"
    else:
        # Also distinguish non-edge runs by ea_slots for future extension
        if args.ea_slots != 64:
            out_json = ROOT / "results" / f"ood_strong_phys_group_ea{args.ea_slots}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] → {out_json}", flush=True)


if __name__ == "__main__":
    main()
