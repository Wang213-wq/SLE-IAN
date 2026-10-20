# EA protocol comparison: fixed memory_size=64, cross edge-injection × warm-start
from __future__ import annotations

import argparse
import json
import shutil
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
from src.train import train_model

SLOTS = (64,)  # fixed at 64
SEED = 42


FIXED_HPARAMS = dict(
    lr=5e-4,
    weight_decay=1e-4,
    dropout=0.15,
    batch_size=128,
    eval_batch_size=256,
    ea_gate_init=-2.0,
    lambda_gamma=1e-3,
)

# E2 and E3 configurations
STREAMS = {
    "edge": dict(
        label="E2:+Edge(edge injection)·from scratch",
        out_root=ROOT / "protocol_compare_edge_scratch",
        use_edge_injection=True,
        init_checkpoint=None,
        max_epochs=300,
        patience=50,
    ),
    "warm": dict(
        label="E3:+Warm(pretrained)·no edge",
        out_root=ROOT / "protocol_compare_warm_noedge",
        use_edge_injection=False,
        init_checkpoint="checkpoints/external_sensitivity_group/32/best.pt",
        max_epochs=100,
        patience=25,
    ),
}



def _bind_run_dirs(home: Path):
    ckpt = home / "checkpoints"
    res = home / "results"
    ckpt.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    def _run_dirs(_experiment: str):
        return {"checkpoint": ckpt, "results": res}

    pc.run_dirs = _run_dirs
    train_mod.run_dirs = _run_dirs
    return home


def _restore_default_run_dirs():
    from config.project_config import run_dirs as default_run_dirs

    pc.run_dirs = default_run_dirs
    train_mod.run_dirs = default_run_dirs


# Build cfg
def build_cfg(stream: dict, slots: int, device: str):
    from src.train import TrainConfig

    cfg = TrainConfig()
    cfg.device = device
    cfg.seed = SEED
    cfg.use_group_split = True
    cfg.readout_mode = "from_gamma"
    cfg.solvent_mode = "x1_weighted" 
    cfg.use_ea = True
    cfg.ea_gated = True
    cfg.ea_slots = slots
    cfg.use_edge_injection = stream["use_edge_injection"]

    cfg.hidden_dim = 128
    cfg.gnn_layers = 3
    cfg.mlp_hidden = 128

    cfg.max_epochs = stream["max_epochs"]
    cfg.patience = stream["patience"]
    cfg.grad_clip = 5.0
    cfg.init_checkpoint = stream["init_checkpoint"]
    cfg.resume_checkpoint = None

    # Fixed hyperparameters
    cfg.lr = FIXED_HPARAMS["lr"]
    cfg.weight_decay = FIXED_HPARAMS["weight_decay"]
    cfg.dropout = FIXED_HPARAMS["dropout"]
    cfg.batch_size = FIXED_HPARAMS["batch_size"]
    cfg.eval_batch_size = FIXED_HPARAMS["eval_batch_size"]
    cfg.ea_gate_init = FIXED_HPARAMS["ea_gate_init"]
    cfg.lambda_gamma = FIXED_HPARAMS["lambda_gamma"]

    
    cfg.core_csv = str(pc.DEFAULT["core_csv"])

    cfg.experiment = f"{stream['label'].split('·')[0].split(':')[0]}/{slots}"
    return cfg


def run_stream(stream_key: str, device: str, yes: bool) -> list[dict]:
    stream = STREAMS[stream_key]
    out_root = stream["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for slots in SLOTS:
        home = out_root / str(slots)

        
        cfg = build_cfg(stream, slots, device)
        print(
            json.dumps(
                {
                    "stream": stream["label"],
                    "slots": slots,
                    "cfg_summary": {
                        "use_edge_injection": cfg.use_edge_injection,
                        "init_checkpoint": cfg.init_checkpoint,
                        "max_epochs": cfg.max_epochs,
                        "patience": cfg.patience,
                        "lr": cfg.lr,
                        "solvent_mode": cfg.solvent_mode,
                        "ea_slots": cfg.ea_slots,
                        "core_csv": cfg.core_csv,
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        if not yes:
            rows.append(
                {
                    "stream": stream_key,
                    "slots": slots,
                    "dry_run": True,
                }
            )
            continue

        summary_path = home / "results" / "summary.json"
        if summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            lb = existing.get("leaderboard", {})
            rows.append(
                {
                    "stream": stream_key,
                    "slots": slots,
                    "test_MAE_LogS": lb.get("test_MAE_LogS"),
                    "test_r2": lb.get("test_r2"),
                    "best_epoch": existing.get("best_epoch"),
                    "skipped": True,
                }
            )
            print(f"  [skip] {slots} already has result → {lb.get('test_MAE_LogS')}", flush=True)
            continue

        home.mkdir(parents=True, exist_ok=True)
        _bind_run_dirs(home)

        assert cfg.solvent_mode == "x1_weighted", (
            f"solvent_mode must be x1_weighted, got {cfg.solvent_mode}"
        )

        # Check warm-start checkpoint exists
        if cfg.init_checkpoint is not None:
            ckpt_path = ROOT / cfg.init_checkpoint
            assert ckpt_path.exists(), f"Warm-start checkpoint missing: {ckpt_path}"

        try:
            summary = train_model(cfg)
        finally:
            _restore_default_run_dirs()

        saved_cfg = summary.get("config", {})
        assert saved_cfg.get("solvent_mode") == "x1_weighted", (
            f"solvent_mode was unexpectedly overwritten after training: {saved_cfg.get('solvent_mode')}"
        )

        lb = summary.get("leaderboard", {})
        rows.append(
            {
                "stream": stream_key,
                "slots": slots,
                "test_MAE_LogS": lb.get("test_MAE_LogS"),
                "test_RMSE_LogS": lb.get("test_RMSE_LogS"),
                "test_r2": lb.get("test_r2"),
                "test_hard_MAE_LogS": lb.get("test_hard_MAE_LogS"),
                "best_epoch": summary.get("best_epoch"),
                "stopped_early": summary.get("stopped_early"),
                "elapsed_sec": summary.get("elapsed_sec"),
            }
        )
        print(f"  [done] {slots} → MAE={lb.get('test_MAE_LogS'):.4f}", flush=True)

    return rows


# Summary
def write_summary(all_rows: list[dict]) -> None:
    out = ROOT / "protocol_compare_summary" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "description": "EA protocol comparison — fixed memory_size=64",
                "streams": {k: v["label"] for k, v in STREAMS.items()},
                "fixed_hparams": {
                    **FIXED_HPARAMS,
                    "seed": SEED,
                },
                "ea_slots": 64,
                "rows": all_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Write Markdown table
    lines = [
        "# EA protocol comparison (fixed memory_size=64)",
        "",
        f"**Fixed hparams**: lr={FIXED_HPARAMS['lr']}, wd={FIXED_HPARAMS['weight_decay']}, "
        f"dropout={FIXED_HPARAMS['dropout']}, batch={FIXED_HPARAMS['batch_size']}, "
        f"gate={FIXED_HPARAMS['ea_gate_init']}, λ_γ={FIXED_HPARAMS['lambda_gamma']}",
        f"**solvent_mode**: x1_weighted (assert)",
        "",
        "## E2: +Edge (edge injection) · from scratch",
        "",
        "| Test MAE | Test R² | Best Epoch |",
        "|----------|---------|------------|",
    ]
    for r in [row for row in all_rows if row.get("stream") == "edge"]:
        lines.append(
            f"| {r.get('test_MAE_LogS', '—')} | {r.get('test_r2', '—')} | {r.get('best_epoch', '—')} |"
        )
    lines.extend(
        [
            "",
            "## E3: +Warm (pretrained) · no edge injection",
            "",
            "| Test MAE | Test R² | Best Epoch |",
            "|----------|---------|------------|",
        ]
    )
    for r in [row for row in all_rows if row.get("stream") == "warm"]:
        lines.append(
            f"| {r.get('test_MAE_LogS', '—')} | {r.get('test_r2', '—')} | {r.get('best_epoch', '—')} |"
        )

    md = out.parent / "SUMMARY.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[summary] → {out}", flush=True)
    print(f"[summary] → {md}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="EA Memory_size sensitivity — two supplementary protocol chains")
    p.add_argument("--stream", choices=["edge", "warm", "both"], default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    print(
        json.dumps(
            {
                "stream": args.stream,
                "slots": list(SLOTS),
                "device": args.device,
                "fixed_hparams": FIXED_HPARAMS,
                "solvent_mode": "x1_weighted (assert)",
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    if not args.yes:
        print("[dry-run] pass --yes to run", flush=True)
        return

    all_rows = []
    for key in ("edge", "warm") if args.stream == "both" else (args.stream,):
        print(f"\n{'='*60}\n===== {STREAMS[key]['label']} =====\n{'='*60}", flush=True)
        rows = run_stream(key, device=args.device, yes=True)
        all_rows.extend(rows)

    write_summary(all_rows)
    print("\n[all done]", flush=True)


if __name__ == "__main__":
    main()
