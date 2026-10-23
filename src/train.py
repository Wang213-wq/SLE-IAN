# SLE_IAN training loop and entrypoints
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.project_config import DEFAULT, run_dirs
from src.data import build_datasets, collate_batch, save_split_artifacts
from src.evaluate import evaluate_loader, save_metrics
from src.model import SLE_IANModel
from src.sle_loss import total_loss


@dataclass
class TrainConfig:
    experiment: str = "exp_geat_phys_group"
    hidden_dim: int = DEFAULT["hidden_dim"]
    gnn_layers: int = DEFAULT["gnn_layers"]
    ea_slots: int = DEFAULT["ea_slots"]
    mlp_hidden: int = DEFAULT["mlp_hidden"]
    dropout: float = DEFAULT["dropout"]
    solvent_mode: str = "concat"
    batch_size: int = DEFAULT["batch_size"]
    eval_batch_size: int = DEFAULT["eval_batch_size"]
    lr: float = DEFAULT["lr"]
    weight_decay: float = DEFAULT["weight_decay"]
    max_epochs: int = DEFAULT["max_epochs"]
    patience: int = DEFAULT["patience"]
    grad_clip: float = DEFAULT["grad_clip"]
    seed: int = DEFAULT["split_seed"]
    train_frac: float = DEFAULT["train_frac"]
    val_frac: float = DEFAULT["val_frac"]
    test_frac: float = DEFAULT["test_frac"]
    num_workers: int = DEFAULT["num_workers"]
    device: str = "auto"
    core_csv: str | None = None
    val_hard_beta: float = 0.4
    use_group_split: bool = True
    split_indices_csv: str | None = None
    split_mode_name: str | None = None
    # Overrides DEFAULT["group_split_csv"] when provided
    group_split_csv: str | None = None
    readout_mode: str = "direct"
    init_checkpoint: str | None = None
    # Resume from last.pt / given ckpt (restores epoch/opt/history/best/stale)
    resume_checkpoint: str | None = None
    ea_gated: bool = False
    ea_gate_init: float = -2.0
    lambda_gamma: float = 0.0
    use_ea: bool = True
    use_edge_injection: bool = False


def cfg_from_preset(preset: dict, *, device: str = "auto") -> TrainConfig:
    """Convert a preset dict (may contain Path / non-dataclass fields) into TrainConfig."""
    fields = TrainConfig.__dataclass_fields__
    kwargs = {}
    for k, v in preset.items():
        if k == "split_seed":
            kwargs["seed"] = int(v)
        elif k in fields:
            if isinstance(v, Path):
                v = str(v)
            kwargs[k] = v
    kwargs.setdefault("experiment", preset.get("experiment", "exp_geat_phys_group"))
    kwargs["device"] = device
    if kwargs.get("core_csv") in ("", None):
        kwargs["core_csv"] = None
    elif kwargs.get("core_csv") is not None:
        kwargs["core_csv"] = str(kwargs["core_csv"])
    bool_keys = (
        "use_group_split",
        "ea_gated",
        "use_ea",
        "use_edge_injection",
    )
    for k in bool_keys:
        if k in kwargs:
            kwargs[k] = bool(kwargs[k])
    if kwargs.get("init_checkpoint") in ("", None):
        kwargs["init_checkpoint"] = None
    return TrainConfig(**{k: v for k, v in kwargs.items() if k in fields})


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_LOG_FILE = None  # Persistent training log handle; one open per process


def _configure_log_file(experiment: str) -> None:
    """Open logs/train_<experiment>.log for append; called once per process."""
    global _LOG_FILE
    if _LOG_FILE is not None:
        return
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{experiment}.log"
    _LOG_FILE = open(log_path, "a", encoding="utf-8")


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def _close_log_file() -> None:
    global _LOG_FILE
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.flush()
            _LOG_FILE.close()
        except Exception:  # noqa: BLE001
            pass
        _LOG_FILE = None


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:04.1f}s"
    if m > 0:
        return f"{m}m{s:04.1f}s"
    return f"{s:.1f}s"


def val_selection_score(val_mae: float, val_hard_mae: float, beta: float) -> float:
    if math.isnan(val_hard_mae):
        return float(val_mae)
    return float(val_mae + float(beta) * float(val_hard_mae))


def _make_train_loader(train_ds, cfg: TrainConfig) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )


def train_model(cfg: TrainConfig, *, epoch_callback=None) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dirs = run_dirs(cfg.experiment)
    res_dir = dirs["results"]
    ckpt_dir = dirs["checkpoint"]

    _configure_log_file(cfg.experiment)
    _log("Recipe: SLE_IAN | dual-stream GNN + External Attention + SLE physics readout")
    _log(
        f"  experiment={cfg.experiment} | device={device} | "
        f"lr={cfg.lr} batch={cfg.batch_size} patience={cfg.patience} | "
        f"val β={cfg.val_hard_beta}"
    )
    _log(
        f"  split: group_split={cfg.use_group_split} | "
        f"arch: solvent={cfg.solvent_mode} "
        f"ea_gated={cfg.ea_gated} use_ea={cfg.use_ea} readout={cfg.readout_mode}"
    )

    train_ds, val_ds, test_ds, manifest = build_datasets(
        cfg.core_csv,
        seed=cfg.seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        artifact_dir=res_dir,
        use_group_split=cfg.use_group_split,
        split_indices_csv=cfg.split_indices_csv,
        split_mode_name=cfg.split_mode_name,
        group_split_csv=cfg.group_split_csv,
    )
    save_split_artifacts(
        train_ds.records, val_ds.records, test_ds.records, manifest, out_dir=res_dir
    )
    _log(
        f"[split] mode={manifest.mode} train={manifest.n_train} val={manifest.n_val} "
        f"test={manifest.n_test} | "
        f"systems={manifest.n_systems_train}/{manifest.n_systems_val}/{manifest.n_systems_test} "
        f"| overlap={manifest.train_system_overlap_with_test}"
    )

    train_loader = _make_train_loader(train_ds, cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )

    model = SLE_IANModel(
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
        ea_slots=cfg.ea_slots,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
        solvent_mode=cfg.solvent_mode,
        readout_mode=cfg.readout_mode,
        ea_gated=cfg.ea_gated,
        ea_gate_init=cfg.ea_gate_init,
        use_ea=cfg.use_ea,
        use_edge_injection=cfg.use_edge_injection,
    ).to(device)
    if cfg.init_checkpoint and not cfg.resume_checkpoint:
        ck0 = Path(cfg.init_checkpoint)
        if not ck0.exists():
            raise FileNotFoundError(f"init_checkpoint not found: {ck0}")
        state0 = torch.load(ck0, map_location=device, weights_only=False)
        raw = state0["model_state"]
        model_sd = model.state_dict()
        filtered = {}
        skipped_shape = []
        for k, v in raw.items():
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped_shape.append(k)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        _log(
            f"warm-start: {ck0} | loaded={len(filtered)} "
            f"skip_shape={len(skipped_shape)} missing={len(missing)} unexpected={len(unexpected)}")
        if skipped_shape:
            _log(f"  skip_shape keys (e.g. ea_slots changed): {skipped_shape[:8]}")
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Trainable params: {n_trainable:,}")

    best_val_mae = float("inf")
    best_val_score = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    start_epoch = 1
    train_start = time.perf_counter()
    stopped_early = False

    if cfg.resume_checkpoint:
        rk = Path(cfg.resume_checkpoint)
        if not rk.exists():
            raise FileNotFoundError(f"resume_checkpoint not found: {rk}")
        rs = torch.load(rk, map_location=device, weights_only=False)
        model.load_state_dict(rs["model_state"], strict=True)
        if rs.get("optimizer_state"):
            try:
                opt.load_state_dict(rs["optimizer_state"])
            except Exception as exc:  # noqa: BLE001
                _log(f"[resume] optimizer state not restored ({exc}); starting fresh Adam")
        history = list(rs.get("history") or [])
        best_val_mae = float(rs.get("best_val_mae", float("inf")))
        best_val_score = float(rs.get("best_val_score", best_val_mae))
        best_epoch = int(rs.get("best_epoch", 0))
        stale = int(rs.get("stale", 0))
        start_epoch = int(rs.get("epoch", 0)) + 1
        _log(
            f"[resume] {rk} → resume ep{start_epoch} → ep{cfg.max_epochs} | "
            f"best@{best_epoch} sel={best_val_score:.4f} stale={stale}/{cfg.patience} "
            f"| hist={len(history)}"
        )
        if start_epoch > cfg.max_epochs:
            _log(f"[resume] already at max_epochs={cfg.max_epochs}; skipping loop, evaluating best directly")
            start_epoch = cfg.max_epochs + 1

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        ep_stats = {
            "loss": 0.0,
            "L_data": 0.0,
            "n": 0,
            "skip": 0,
        }
        for batch in train_loader:
            solute = batch["solute"].to(device)
            s1 = batch["solvent1"].to(device)
            s2 = batch["solvent2"].to(device)
            cond = batch["cond"].to(device)
            y = batch["logS"].to(device)
            ideal = batch["ideal_ln_x"].to(device)
            mw_u = batch["mw_solute"].to(device)
            mw_v = batch["mw_solvent"].to(device)

            out = model(
                solute,
                s1,
                s2,
                cond,
                mw_solute=mw_u,
                mw_solvent=mw_v,
                ideal_ln_x=ideal,
            )
            loss, stats = total_loss(
                out["logS"],
                y,
                out["ln_gamma"],
                lambda_gamma=cfg.lambda_gamma,
            )
            if not torch.isfinite(loss):
                # Skip batches with NaN/Inf loss to avoid polluting grads
                ep_stats["skip"] += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                # Gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = int(y.size(0))
            ep_stats["loss"] += stats["loss"] * bs
            ep_stats["L_data"] += stats["L_data"] * bs
            ep_stats["n"] += bs

        n = max(ep_stats["n"], 1)
        train_loss = ep_stats["loss"] / n
        if ep_stats["skip"] > 0:
            _log(f"  [info] ep{epoch}: skipped {ep_stats['skip']} non-finite-loss batches")

        val_bundle, _ = evaluate_loader(model, val_loader, device)
        val_mae = val_bundle.mae
        val_hard = val_bundle.hard_mae
        val_score = val_selection_score(val_mae, val_hard, cfg.val_hard_beta)
        select_metric = val_mae
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - train_start

        improved = select_metric < best_val_score - 1e-8
        if improved:
            best_val_mae = val_mae
            best_val_score = select_metric
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_L_data": ep_stats["L_data"] / n,
            "val_MAE_LogS": val_mae,
            "val_RMSE_LogS": val_bundle.rmse,
            "val_R2_LogS": val_bundle.r2,
            "val_hard_MAE_LogS": val_hard,
            "val_score": val_score,
            "best_val_MAE_LogS": best_val_mae,
            "best_val_score": best_val_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "epoch_seconds": epoch_elapsed,
            "elapsed_seconds": total_elapsed,
        }
        history.append(row)

        if epoch_callback is not None:
            epoch_callback(epoch, row)

        _log(
            f"ep{epoch:03d} | loss={train_loss:.4f} "
            f"(data={row['train_L_data']:.4f}) | "
            f"val MAE={val_mae:.4f} R2={val_bundle.r2:.4f} RMSE={val_bundle.rmse:.4f} "
            f"hard={val_hard:.4f} score={val_score:.4f} | "
            f"best@{best_epoch} sel={best_val_score:.4f} | "
            f"stale={stale}/{cfg.patience} | {_format_duration(epoch_elapsed)}"
        )

        ckpt_cfg = asdict(cfg)
        for k, v in list(ckpt_cfg.items()):
            if isinstance(v, Path):
                ckpt_cfg[k] = str(v)
            elif isinstance(v, tuple):
                ckpt_cfg[k] = list(v)
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "config": ckpt_cfg,
            "val_mae": val_mae,
            "val_hard_mae": val_hard,
            "best_val_mae": best_val_mae,
            "best_val_score": best_val_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "history": history,
            "elapsed_seconds": total_elapsed,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if improved:
            torch.save(ckpt, ckpt_dir / "best.pt")

        if stale >= cfg.patience:
            stopped_early = True
            _log(f"Early stopping @ ep{epoch}")
            break

    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])

    _log(f"\n=== SLE_IAN training done @ ep{best_epoch} ===")
    metrics = {}
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        # train loader may have a sampler; re-build with shuffle=False for deterministic eval
        if name == "train":
            loader = DataLoader(
                train_ds,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                collate_fn=collate_batch,
            )
        bundle, pred_df = evaluate_loader(model, loader, device)
        score = val_selection_score(bundle.mae, bundle.hard_mae, cfg.val_hard_beta)
        payload = bundle.to_dict()
        payload["score"] = score
        save_metrics(bundle, res_dir / f"metrics_{name}.json", extra={"score": score})
        pred_df.to_csv(res_dir / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")
        metrics[name] = payload
        _log(
            f"[{name}] MAE={bundle.mae:.4f} RMSE={bundle.rmse:.4f} "
            f"R2={bundle.r2:.4f} Pearson={bundle.pearson:.4f} "
            f"hard={bundle.hard_mae:.4f} score={score:.4f} n={bundle.n}"
        )

    # Primary leaderboard metrics over the test split
    leaderboard = {
        "primary_metric": "test_MAE_LogS_on_train_split",
        "test_MAE_LogS": metrics["test"]["mae"],
        "test_RMSE_LogS": metrics["test"]["rmse"],
        "test_r2": metrics["test"]["r2"],
        "test_pearson": metrics["test"]["pearson"],
        "test_hard_MAE_LogS": metrics["test"]["hard_mae"],
        "test_score": metrics["test"]["score"],
        "n_test": metrics["test"]["n"],
    }

    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        elif isinstance(v, tuple):
            cfg_dict[k] = list(v)
    summary = {
        "experiment": cfg.experiment,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_val_score": best_val_score,
        "stopped_early": stopped_early,
        "elapsed_sec": time.perf_counter() - train_start,
        "metrics": metrics,
        "leaderboard": leaderboard,
        "config": cfg_dict,
        "split": manifest.to_dict(),
    }
    with open(res_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(res_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(res_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)
    _log(
        f"[leaderboard] test MAE={leaderboard['test_MAE_LogS']:.4f} "
        f"RMSE={leaderboard['test_RMSE_LogS']:.4f} "
        f"R2={leaderboard['test_r2']:.4f} "
        f"hard={leaderboard['test_hard_MAE_LogS']:.4f} "
        f"n={leaderboard['n_test']}"
    )
    _log(
        f"[done] best_val_MAE={best_val_mae:.4f} score={best_val_score:.4f} "
        f"@ep{best_epoch} | wall-clock {_format_duration(time.perf_counter() - train_start)} → {res_dir}"
    )
    _close_log_file()
    return summary


def _build_arg_parser():
    """CLI for single-preset training: `python -m src.train --preset <name>`."""
    import argparse

    from config.train_presets import PRESETS

    p = argparse.ArgumentParser(description="SLE_IAN train")
    p.add_argument(
        "--preset",
        default="exp_geat_phys_group_edge_ea64",
        choices=sorted(PRESETS.keys()),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--core-csv", default=None)
    return p


def main() -> None:
    """Entry point for a single preset run. Invoke from SLE_IAN/ via `python -m src.train --preset <name>`."""
    from config.console import configure_console_utf8

    configure_console_utf8()

    import torch_geometric

    torch_geometric.typing.WITH_TORCH_SCATTER = False

    from config.train_presets import resolve_preset

    args = _build_arg_parser().parse_args()
    cfg = cfg_from_preset(resolve_preset(args.preset), device=args.device)
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.core_csv is not None:
        cfg.core_csv = args.core_csv

    print(f"[train] experiment={cfg.experiment} device={cfg.device}", flush=True)
    train_model(cfg)


if __name__ == "__main__":
    main()
