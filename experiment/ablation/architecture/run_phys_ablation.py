# Architecture ablation: GNN_SLE / GEAT / GNN_only — train & report
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.console import configure_console_utf8

configure_console_utf8()

import torch_geometric

torch_geometric.typing.WITH_TORCH_SCATTER = False

import config.project_config as pc
import src.train as train_mod
from config.train_presets import resolve_preset
from src.train import cfg_from_preset, train_model

# Default: group_split + edge injection + Memory_size=64
DEFAULT_EA_SLOTS = 64
DEFAULT_ROOT_NAME = "ablation_group_m64_edge"


ABLATION_ROOT = ROOT / DEFAULT_ROOT_NAME

# edge injection, ea64
VARIANT_PRESETS = {
    "GNN_SLE": "exp_ablate_gnn_sle_group_edge_ea64",
    "GEAT": "exp_ablate_geat_group_edge_ea64",
    "GNN_only": "exp_ablate_gnn_only_group_edge_ea64",
}

VARIANTS: list[dict] = [
    {
        "folder": "GNN_SLE",
        "preset": VARIANT_PRESETS["GNN_SLE"],
        "train": True,
        "role": "ablate_EAT",
        "eat": False,
        "hard_sle": True,
    },
    {
        "folder": "GEAT",
        "preset": VARIANT_PRESETS["GEAT"],
        "train": True,
        "role": "ablate_hard_SLE",
        "eat": True,
        "hard_sle": False,
    },
    {
        "folder": "GNN_only",
        "preset": VARIANT_PRESETS["GNN_only"],
        "train": True,
        "role": "double_ablate",
        "eat": False,
        "hard_sle": False,
    },
]


def variant_dir(folder: str) -> Path:
    return ABLATION_ROOT / folder


def last_pt_path(folder: str) -> Path:
    return variant_dir(folder) / "checkpoints" / "last.pt"


def best_pt_path(folder: str) -> Path:
    return variant_dir(folder) / "checkpoints" / "best.pt"


def _bind_run_dirs(folder: str) -> Path:
    home = variant_dir(folder)
    ckpt = home / "checkpoints"
    res = home / "results"
    ckpt.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    def _run_dirs(_experiment: str) -> dict[str, Path]:
        ckpt.mkdir(parents=True, exist_ok=True)
        res.mkdir(parents=True, exist_ok=True)
        return {"checkpoint": ckpt, "results": res}

    pc.run_dirs = _run_dirs  
    train_mod.run_dirs = _run_dirs  
    return home


def _restore_default_run_dirs() -> None:
    from config.project_config import run_dirs as default_run_dirs

    pc.run_dirs = default_run_dirs
    train_mod.run_dirs = default_run_dirs


def _ensure_tree(*, ea_slots: int) -> None:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        home = variant_dir(v["folder"])
        (home / "checkpoints").mkdir(parents=True, exist_ok=True)
        (home / "results").mkdir(parents=True, exist_ok=True)
        eat = "Y" if v["eat"] else "N"
        sle = "hard" if v["hard_sle"] else "direct"
        (home / "README.md").write_text(
            f"# {v['folder']}\n\n"
            f"- protocol: group_split\n"
            f"- Memory_size / ea_slots: {ea_slots}\n"
            f"- role: {v['role']}\n"
            f"- EAT: {eat}\n"
            f"- readout: {sle}\n"
            f"- preset: {v['preset']}\n"
            f"- resume: `checkpoints/last.pt`\n",
            encoding="utf-8",
        )


def _is_complete(folder: str) -> bool:
    lb_path = variant_dir(folder) / "results" / "leaderboard.json"
    if not lb_path.exists():
        return False
    try:
        lb = json.loads(lb_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return lb.get("test_MAE_LogS") is not None


def _peek_last_pt(folder: str) -> dict | None:
    path = last_pt_path(folder)
    if not path.exists():
        return None
    try:
        import torch

        rs = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "path": str(path),
            "epoch": int(rs.get("epoch", 0)),
            "best_epoch": int(rs.get("best_epoch", 0)),
            "best_val_mae": float(rs.get("best_val_mae", float("nan"))),
            "stale": int(rs.get("stale", 0)),
            "history_len": len(rs.get("history") or []),
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": str(exc)}


def _write_resume_meta(folder: str, extra: dict | None = None) -> None:
    home = variant_dir(folder)
    peek = _peek_last_pt(folder) or {}
    meta = {
        "folder": folder,
        "last_pt": str(last_pt_path(folder)),
        "best_pt": str(best_pt_path(folder)),
        "last_pt_exists": last_pt_path(folder).exists(),
        "best_pt_exists": best_pt_path(folder).exists(),
        "complete": _is_complete(folder),
        "peek": peek,
    }
    if extra:
        meta.update(extra)
    (home / "resume_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _leaderboard_row(folder: str, meta: dict) -> dict | None:
    if not _is_complete(folder):
        return None
    lb = json.loads(
        (variant_dir(folder) / "results" / "leaderboard.json").read_text(encoding="utf-8")
    )
    sm_path = variant_dir(folder) / "results" / "summary.json"
    best_epoch = None
    if sm_path.exists():
        best_epoch = json.loads(sm_path.read_text(encoding="utf-8")).get("best_epoch")
    return {
        "folder": folder,
        "preset": meta.get("preset"),
        "test_MAE_LogS": lb.get("test_MAE_LogS"),
        "test_RMSE_LogS": lb.get("test_RMSE_LogS"),
        "test_r2": lb.get("test_r2"),
        "test_hard_MAE_LogS": lb.get("test_hard_MAE_LogS"),
        "best_epoch": best_epoch,
        "source": "completed_leaderboard",
        "eat": meta["eat"],
        "hard_sle": meta["hard_sle"],
        "slot_dir": str(variant_dir(folder)),
        "resume": False,
    }


def _write_metrics(folder: str, row: dict) -> None:
    home = variant_dir(folder)
    home.mkdir(parents=True, exist_ok=True)
    (home / "metrics.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_summary(rows: list[dict], *, ea_slots: int) -> None:
    order = {v["folder"]: i for i, v in enumerate(VARIANTS)}
    rows_sorted = sorted(rows, key=lambda r: order.get(r["folder"], 99))
    payload = {
        "title": f"SLE_IAN ablation (group_split, Memory_size={ea_slots})",
        "protocol": "group_split_only",
        "ea_slots": ea_slots,
        "root": str(ABLATION_ROOT),
        "resume_per_variant": True,
        "rows": rows_sorted,
    }
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    (ABLATION_ROOT / "ablation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"# SLE_IAN architecture ablation (group_split, Memory_size={ea_slots})",
        "",
        f"Protocol: `group_split_only`; base preset aligned to main model (**ea_slots={ea_slots}**) — only EA / hard-SLE toggled.",
        "",
        "| Variant | EA | Hard-SLE | test MAE(LogS) | test R² | hard MAE | best_epoch | source |",
        "|------|-----|--------|----------------|---------|----------|------------|--------|",
    ]
    for r in rows_sorted:
        mae = r.get("test_MAE_LogS")
        r2 = r.get("test_r2")
        hard = r.get("test_hard_MAE_LogS")
        mae_s = f"{mae:.4f}" if mae is not None else "—"
        r2_s = f"{r2:.4f}" if r2 is not None else "—"
        hard_s = f"{hard:.4f}" if hard is not None else "—"
        be = r.get("best_epoch")
        be_s = str(be) if be is not None else "—"
        eat = "Y" if r.get("eat") else "N"
        sle = "Y" if r.get("hard_sle") else "N"
        lines.append(
            f"| {r['folder']} | {eat} | {sle} | {mae_s} | {r2_s} | {hard_s} | {be_s} | {r.get('source')} |"
        )
    lines += [
        "",
        "- **GNN_SLE**: EA disabled",
        "- **GEAT**: hard-SLE disabled (`direct`)",
        "- **GNN_only**: both ablated",
        f"- Artifacts: `{ABLATION_ROOT.as_posix()}`",
    ]
    (ABLATION_ROOT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _print_status(variants: list[dict]) -> None:
    print("ablation resume status", flush=True)
    for meta in variants:
        folder = meta["folder"]
        complete = _is_complete(folder)
        peek = _peek_last_pt(folder)
        if complete:
            lb = _leaderboard_row(folder, meta) or {}
            print(
                f"{folder}: COMPLETE MAE={lb.get('test_MAE_LogS')} "
                f"best_ep={lb.get('best_epoch')}",
                flush=True,
            )
        elif peek and "epoch" in peek:
            print(
                f"{folder}: RESUMABLE last_ep={peek['epoch']} "
                f"best@{peek.get('best_epoch')} stale={peek.get('stale')}",
                flush=True,
            )
        else:
            print(f"{folder}: FRESH (no last.pt)", flush=True)
        _write_resume_meta(folder)


def main() -> None:
    p = argparse.ArgumentParser(description="SLE_IAN ablation @ group_split")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ea-slots", type=int, default=DEFAULT_EA_SLOTS)
    p.add_argument(
        "--root",
        default=None,
        help="Ablation root name (relative to SLE_IAN/), default ablation_group_m64_edge",
    )
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--status", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    # Override global paths from CLI args
    global ABLATION_ROOT
    root_name = args.root or DEFAULT_ROOT_NAME
    ABLATION_ROOT = ROOT / root_name

    ea_slots = int(args.ea_slots)
    do_resume = bool(args.resume) and (not args.no_resume)
    variants = VARIANTS
    if args.only:
        wanted = set(args.only)
        variants = [v for v in VARIANTS if v["folder"] in wanted]
        if not variants:
            raise SystemExit(f"--only matched nothing: {args.only}")

    _ensure_tree(ea_slots=ea_slots)
    if args.status:
        _print_status(variants)
        return

    plan = {
        "protocol": "group_split_only",
        "ea_slots": ea_slots,
        "edge_injection": True,
        "root": str(ABLATION_ROOT),
        "variants": variants,
        "resume_per_variant": do_resume,
        "device": args.device,
    }
    (ABLATION_ROOT / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if not args.yes:
        print("[dry-run] pass --yes to start training", flush=True)
        _print_status(variants)
        return

    rows: list[dict] = []
    for meta in variants:
        folder = meta["folder"]
        if not args.force and _is_complete(folder):
            done = _leaderboard_row(folder, meta)
            assert done is not None
            rows.append(done)
            _write_metrics(folder, done)
            _write_resume_meta(folder, {"status": "skipped_complete"})
            print(
                f"[done-skip] {folder} MAE={done['test_MAE_LogS']:.4f}",
                flush=True,
            )
            _write_summary(rows, ea_slots=ea_slots)
            continue

        home = _bind_run_dirs(folder)
        last_pt = last_pt_path(folder)
        preset = resolve_preset(meta["preset"])
        cfg = cfg_from_preset(preset, device=args.device)
        cfg.experiment = f"ablation_group_m{ea_slots}/{folder}"
        cfg.ea_slots = ea_slots
        cfg.use_group_split = True
        if args.max_epochs is not None:
            cfg.max_epochs = int(args.max_epochs)
        if args.patience is not None:
            cfg.patience = int(args.patience)
        if cfg.init_checkpoint and not Path(cfg.init_checkpoint).is_absolute():
            cfg.init_checkpoint = str((ROOT / cfg.init_checkpoint).resolve())

        if do_resume and last_pt.exists():
            cfg.resume_checkpoint = str(last_pt)
            peek = _peek_last_pt(folder) or {}
            print(
                f"\n===== {folder} → {home} [RESUME ep{peek.get('epoch', '?')}] "
                f"ea_slots={ea_slots} group_split =====",
                flush=True,
            )
            _write_resume_meta(
                folder,
                {
                    "status": "resuming",
                    "resume_checkpoint": str(last_pt),
                    "ea_slots": ea_slots,
                },
            )
        else:
            cfg.resume_checkpoint = None
            mode = "fresh" if not last_pt.exists() else "no-resume"
            print(
                f"\n===== {folder} → {home} [{mode}] "
                f"ea_slots={ea_slots} group_split =====",
                flush=True,
            )
            _write_resume_meta(folder, {"status": mode, "ea_slots": ea_slots})

        summary = train_model(cfg)
        lb = summary.get("leaderboard") or {}
        row = {
            "folder": folder,
            "preset": meta["preset"],
            "ea_slots": ea_slots,
            "test_MAE_LogS": lb.get("test_MAE_LogS"),
            "test_RMSE_LogS": lb.get("test_RMSE_LogS"),
            "test_r2": lb.get("test_r2"),
            "test_hard_MAE_LogS": lb.get("test_hard_MAE_LogS"),
            "best_epoch": summary.get("best_epoch"),
            "source": "trained_resumed" if cfg.resume_checkpoint else "trained",
            "resume_from": cfg.resume_checkpoint,
            "eat": meta["eat"],
            "hard_sle": meta["hard_sle"],
            "slot_dir": str(home),
            "resume": bool(cfg.resume_checkpoint),
        }
        rows.append(row)
        _write_metrics(folder, row)
        _write_resume_meta(folder, {"status": "finished", "row": row})
        _write_summary(rows, ea_slots=ea_slots)

    _write_summary(rows, ea_slots=ea_slots)
    print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), flush=True)
    print(f"[done] → {ABLATION_ROOT}", flush=True)
    _restore_default_run_dirs()


if __name__ == "__main__":
    main()
