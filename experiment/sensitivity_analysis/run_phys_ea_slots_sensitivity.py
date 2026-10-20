# External Attention memory_size (ea_slots) — two-phase sensitivity study
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

# Default points at group_split product; main() rebinds per preset.
SENS_ROOT = ROOT / "checkpoints" / "external_sensitivity_group"
OFFICIAL_SLOTS = 64
DEFAULT_SLOTS = (16, 32, 64, 128, 256)


def slot_dir(slots: int) -> Path:
    return SENS_ROOT / str(int(slots))


def _bind_sens_root(*, edge_injection: bool = False) -> Path:
    # Phase I (no edge injection) / Phase II (edge injection) are kept apart so
    # a checkpoint never resumes into the wrong protocol.
    global SENS_ROOT
    SENS_ROOT = ROOT / "checkpoints" / (
        "external_sensitivity_group_edge" if edge_injection else "external_sensitivity_group"
    )
    return SENS_ROOT


def _bind_run_dirs(slots: int) -> Path:
    # Route train_model's checkpoint/results under checkpoints/external_sensitivity/{slots}/.
    home = slot_dir(slots)
    ckpt = home
    res = home / "results"
    ckpt.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    def _run_dirs(_experiment: str) -> dict[str, Path]:
        ckpt.mkdir(parents=True, exist_ok=True)
        res.mkdir(parents=True, exist_ok=True)
        return {"checkpoint": ckpt, "results": res}

    pc.run_dirs = _run_dirs  # type: ignore[assignment]
    train_mod.run_dirs = _run_dirs  # type: ignore[assignment]
    return home


def _restore_default_run_dirs() -> None:
    from config.project_config import run_dirs as default_run_dirs

    pc.run_dirs = default_run_dirs
    train_mod.run_dirs = default_run_dirs


def _ensure_tree(slots_list: list[int]) -> None:
    SENS_ROOT.mkdir(parents=True, exist_ok=True)
    for s in slots_list:
        home = slot_dir(s)
        home.mkdir(parents=True, exist_ok=True)
        (home / "results").mkdir(parents=True, exist_ok=True)
        readme = home / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# Memory_size = {s} (`ea_slots={s}`)\n\n"
                f"- `best.pt` (independent resume per slot)\n"
                f"- results/: leaderboard / summary / metrics\n"
                f"- metrics.json: summary row for this slot\n",
                encoding="utf-8",
            )


def _load_cached_row(slots: int) -> dict | None:
    path = slot_dir(slots) / "metrics.json"
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if row.get("test_MAE_LogS") is None:
        return None
    return row


def _leaderboard_row(slots: int) -> dict | None:
    lb_path = slot_dir(slots) / "results" / "leaderboard.json"
    sm_path = slot_dir(slots) / "results" / "summary.json"
    if not lb_path.exists():
        return None
    lb = json.loads(lb_path.read_text(encoding="utf-8"))
    if lb.get("test_MAE_LogS") is None:
        return None
    best_epoch = None
    if sm_path.exists():
        sm = json.loads(sm_path.read_text(encoding="utf-8"))
        best_epoch = sm.get("best_epoch")
    return {
        "ea_slots": slots,
        "Memory_size": slots,
        "experiment": f"{SENS_ROOT.name}/{slots}",
        "test_MAE_LogS": lb.get("test_MAE_LogS"),
        "test_RMSE_LogS": lb.get("test_RMSE_LogS"),
        "test_r2": lb.get("test_r2"),
        "test_hard_MAE_LogS": lb.get("test_hard_MAE_LogS"),
        "best_epoch": best_epoch,
        "source": "completed_leaderboard",
        "slot_dir": str(slot_dir(slots)),
    }


def _write_slot_metrics(slots: int, row: dict) -> None:
    home = slot_dir(slots)
    home.mkdir(parents=True, exist_ok=True)
    (home / "metrics.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_summary(rows: list[dict], *, protocol: str) -> None:
    rows_sorted = sorted(rows, key=lambda r: int(r["Memory_size"]))
    payload = {
        "title": "GEAT-Phys EA Memory_size (ea_slots) sensitivity",
        "protocol": protocol,
        "root": str(SENS_ROOT),
        "official_Memory_size": OFFICIAL_SLOTS,
        "resume_supported": True,
        "layout": f"{SENS_ROOT.name}/{{Memory_size}}/ (best.pt/last.pt flat) | results/",
        "rows": rows_sorted,
    }
    SENS_ROOT.mkdir(parents=True, exist_ok=True)
    (SENS_ROOT / "ea_slots_sensitivity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# External Attention Memory_size sensitivity",
        "",
        f"Protocol: `{protocol}` (zero-system-overlap group split)",
        f"Root: `{SENS_ROOT.as_posix()}` (each slot has its own subdir)",
        "",
        "| Memory_size | test MAE(LogS) | test R² | hard MAE | best_epoch | source | folder |",
        "|-------------|----------------|---------|----------|------------|--------|--------|",
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
        folder = f"{SENS_ROOT.name}/{r['Memory_size']}"
        lines.append(
            f"| {r['Memory_size']} | {mae_s} | {r2_s} | {hard_s} | {be_s} | {r.get('source')} | `{folder}` |"
        )
    lines += [
        "",
        "- Metrics: test MAE / **R² (determination coefficient)** / hard MAE — all written to `results/leaderboard.json`.",
        f"- Resume: `{SENS_ROOT.name}/{{N}}/last.pt`",
    ]
    lines.append(
        f"- Under group_split, Memory_size={OFFICIAL_SLOTS} is fully retrained on the same footing as every other slot."
    )
    (SENS_ROOT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="EA Memory_size sensitivity → external_sensitivity/")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--preset",
        default="exp_geat_phys_group",
        choices=["exp_geat_phys_group", "exp_geat_phys_group_edge_ea64"],
        help="Base preset name (Phase I: exp_geat_phys_group; Phase II: exp_geat_phys_group_edge_ea64)",
    )
    p.add_argument(
        "--slots",
        nargs="+",
        type=int,
        default=list(DEFAULT_SLOTS),
        help="Memory_size list (default 16 32 64 128 256)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    # Resolve protocol from preset
    is_edge_injection = args.preset == "exp_geat_phys_group_edge_ea64"
    protocol_name = "group_split_edge_injection" if is_edge_injection else "group_split_only"
    _bind_sens_root(edge_injection=is_edge_injection)

    # ea_slots is the only variable in the sensitivity sweep
    base_preset = resolve_preset(args.preset)

    do_resume = bool(args.resume) and (not args.no_resume)
    slots = [int(s) for s in args.slots]
    to_train = slots

    _ensure_tree(slots)
    plan = {
        "protocol": protocol_name,
        "base_preset": args.preset,
        "root": str(SENS_ROOT),
        "Memory_size_grid": slots,
        "to_train": to_train,
        "resume_per_slots": do_resume,
        "metrics": ["MAE_LogS", "RMSE_LogS", "R2", "hard_MAE_LogS"],
        "layout": {
            str(s): {
                "home": str(slot_dir(s)),
                "checkpoints": str(slot_dir(s)),
                "results": str(slot_dir(s) / "results"),
            }
            for s in slots
        },
        "seed": args.seed,
        "max_epochs": base_preset.get("max_epochs"),
        "patience": base_preset.get("patience"),
        "device": args.device,
    }
    (SENS_ROOT / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if not args.yes:
        print("[dry-run] dirs created; pass --yes to start training", flush=True)
        return

    rows: list[dict] = []
    for s in slots:
        if not args.force:
            done = _leaderboard_row(s) or _load_cached_row(s)
            if done is not None:
                rows.append(done)
                _write_slot_metrics(s, done)
                print(
                    f"[done-skip] Memory_size={s} MAE={done['test_MAE_LogS']:.4f} "
                    f"R2={done.get('test_r2')} → {slot_dir(s)}",
                    flush=True,
                )
                _write_summary(rows, protocol=protocol_name)
                continue

        home = _bind_run_dirs(s)
        last_pt = home / "last.pt"

        preset = resolve_preset(args.preset)
        cfg = cfg_from_preset(preset, device=args.device)
        cfg.seed = int(args.seed)
        cfg.ea_slots = int(s)
        cfg.experiment = f"{SENS_ROOT.name}/{s}"

        if do_resume and last_pt.exists():
            cfg.resume_checkpoint = str(last_pt)
            print(
                f"\n===== Memory_size={s} → {home} [RESUME] =====",
                flush=True,
            )
        else:
            cfg.resume_checkpoint = None
            mode = "fresh" if not last_pt.exists() else "no-resume"
            print(
                f"\n===== Memory_size={s} → {home} [{mode}] =====",
                flush=True,
            )

        summary = train_model(cfg)
        lb = summary.get("leaderboard") or {}
        row = {
            "ea_slots": s,
            "Memory_size": s,
            "experiment": cfg.experiment,
            "test_MAE_LogS": lb.get("test_MAE_LogS"),
            "test_RMSE_LogS": lb.get("test_RMSE_LogS"),
            "test_r2": lb.get("test_r2"),
            "test_hard_MAE_LogS": lb.get("test_hard_MAE_LogS"),
            "best_epoch": summary.get("best_epoch"),
            "source": "trained_resumed" if cfg.resume_checkpoint else "trained",
            "resume_from": cfg.resume_checkpoint,
            "slot_dir": str(home),
        }
        rows.append(row)
        _write_slot_metrics(s, row)
        _write_summary(rows, protocol=protocol_name)

    _write_summary(rows, protocol=protocol_name)
    print(json.dumps({"protocol": protocol_name, "rows": rows}, indent=2, ensure_ascii=False), flush=True)
    print(f"[done] → {SENS_ROOT}", flush=True)
    _restore_default_run_dirs()


if __name__ == "__main__":
    main()
