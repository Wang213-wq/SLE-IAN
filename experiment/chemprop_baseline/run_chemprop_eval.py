# Chemprop baseline — test-set evaluation
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "chemprop_baseline"

# chemprop is installed as a CLI entry point, not an importable module; default to PATH chemprop,
CHEMPROP_EXE = Path(os.environ.get("CHEMPROP_EXE", "chemprop"))


def _require_chemprop() -> None:
    if str(CHEMPROP_EXE) != "chemprop" and not CHEMPROP_EXE.exists():
        raise FileNotFoundError(
            f"chemprop CLI not found: {CHEMPROP_EXE} (set env var CHEMPROP_EXE to full path)"
        )


def run_eval(ckpt_dir_name: str, label: str) -> dict:
    ckpt_dir = BASE / ckpt_dir_name
    ckpts = sorted(ckpt_dir.glob("**/best.pt")) + sorted(ckpt_dir.glob("**/model.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no chemprop checkpoint found: {ckpt_dir}")
    ckpt = ckpts[0]
    print(f"\n=== [{label}] checkpoint: {ckpt}")

    pred_path = BASE / f"test_preds_{ckpt_dir_name}.csv"
    cmd = [
        str(CHEMPROP_EXE), "predict",
        "-i", str(BASE / "test.csv"),
        "--model-path", str(ckpt),
        "-o", str(pred_path),
        "--descriptors-columns", "temperature_K", "x1",
        "--accelerator", "gpu",
        "-n", "0",
    ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=False)

    pred_df = pd.read_csv(pred_path)
    # chemprop predict overwrites the target column (LogS) with predictions in regression mode
    y_pred = pred_df["LogS"].to_numpy(dtype=float)

    test_df = pd.read_csv(BASE / "test.csv")
    y_true = test_df["LogS"].to_numpy(dtype=float)
    assert len(y_pred) == len(y_true)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    hard_mask = y_true < 0.0
    hard_mae = float(np.mean(np.abs(y_true[hard_mask] - y_pred[hard_mask])))
    hard_n = int(hard_mask.sum())

    return {
        "label": label,
        "checkpoint_dir": ckpt_dir_name,
        "mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson,
        "hard_mae": hard_mae, "hard_n": hard_n,
        "n": int(len(y_true)),
    }


def main():
    _require_chemprop()
    import json

    # Check whether training finished: MSE loss (run) and MAE loss (run_mae)
    results = []
    for ckpt_dir, label in [
        ("run", "Chemprop (MSE loss)"),
        ("run_mae", "Chemprop (MAE loss)"),
    ]:
        if (BASE / ckpt_dir).exists():
            try:
                results.append(run_eval(ckpt_dir, label))
            except FileNotFoundError as e:
                print(f"[skip] {e}")

    if not results:
        raise SystemExit("no chemprop training artifacts found — train first")

    print("Test-set metrics (n=3922, group_split seed=42, LogS space)")
    print(f"{'Model':<30s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'Pearson':>8s} {'hard_MAE':>9s}")
    for r in results:
        print(f"{r['label']:<30s} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>8.4f} {r['pearson']:>8.4f} {r['hard_mae']:>9.4f}")

    out_json = BASE / "chemprop_test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {out_json}")


if __name__ == "__main__":
    main()
