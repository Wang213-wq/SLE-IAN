# Chemprop strong OOD four-split experiment: data prep + test-set evaluation
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CORE_CSV = ROOT / "dataset" / "public dataset and split" / "Bao2024.csv"
# Frozen neutral split (shared between SLE-IAN and ChemProp)
SPLITS_DIR = ROOT / "dataset" / "strong_ood_splits"
BASE = ROOT / "chemprop_baseline" / "strong_ood"
# Default to PATH chemprop; env var CHEMPROP_EXE can override with full path
CHEMPROP_EXE = Path(os.environ.get("CHEMPROP_EXE", "chemprop"))

MODES = ["solvent", "solute", "scaffold", "temperature"]


# Data preparation
def build_chemprop_df(core_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    # Rows from core table picked by split_indices row_idx, assembled into ChemProp format
    # split_df row_idx maps to the core DataFrame row index
    sub = core_df.iloc[split_df["row_idx"].values].reset_index(drop=True)
    smiles = (
        sub["Drug_SMILES"].astype(str) + "." +
        sub["Solvent_1_SMILES"].astype(str) + "." +
        sub["Solvent_2_SMILES"].astype(str)
    )
    out = pd.DataFrame({
        "smiles": smiles,
        "temperature_K": sub["Temperature(K)"].astype(float),
        "x1": sub["Solvent_mol_fraction"].astype(float),
        "LogS": sub["LogS"].astype(float),
        "split": split_df["split"].values,  # train / val / test
    })
    # Drop any NaN rows just in case
    out = out.dropna(subset=["smiles", "temperature_K", "x1", "LogS"]).reset_index(drop=True)
    return out


def run_prepare() -> None:
    core_df = pd.read_csv(CORE_CSV, encoding="gbk")
    print(f"[core] {CORE_CSV.name}: {len(core_df)} rows")

    for mode in MODES:
        split_csv = SPLITS_DIR / mode / "split_indices.csv"
        if not split_csv.exists():
            print(f"[skip] {mode}: split_indices.csv not found")
            continue

        split_df = pd.read_csv(split_csv)
        print(f"\n[{mode}] split_indices: {len(split_df)} rows")
        print(f"  split counts: {split_df['split'].value_counts().to_dict()}")
        print(f"  row_idx range: [{split_df['row_idx'].min()}, {split_df['row_idx'].max()}]")

        chem_df = build_chemprop_df(core_df, split_df)
        print(f"  chemprop rows after dropna: {len(chem_df)}")
        print(f"  final split counts: {chem_df['split'].value_counts().to_dict()}")

        out_dir = BASE / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        chem_df.to_csv(out_dir / "all.csv", index=False)
        for s in ["train", "val", "test"]:
            chem_df[chem_df["split"] == s].to_csv(out_dir / f"{s}.csv", index=False)
        print(f"  [save] {out_dir}/")

    print("\n[done] 4 modes prepared.")


# Test-set evaluation
def eval_mode(mode: str) -> dict:
    mode_dir = BASE / mode
    ckpts = sorted((mode_dir / "run").glob("**/best.pt")) + sorted((mode_dir / "run").glob("**/model.pt"))
    if not ckpts:
        raise FileNotFoundError(f"{mode}: no checkpoint found")
    ckpt = ckpts[0]

    test_csv = mode_dir / "test.csv"
    pred_csv = mode_dir / "test_preds.csv"
    cmd = [
        str(CHEMPROP_EXE), "predict",
        "-i", str(test_csv),
        "--model-path", str(ckpt),
        "-o", str(pred_csv),
        "--descriptors-columns", "temperature_K", "x1",
        "--accelerator", "gpu",
        "-n", "0",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    pred_df = pd.read_csv(pred_csv)
    y_pred = pred_df["LogS"].to_numpy(dtype=float)
    test_df = pd.read_csv(test_csv)
    y_true = test_df["LogS"].to_numpy(dtype=float)
    assert len(y_pred) == len(y_true)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    hard_mask = y_true < 0.0
    hard_mae = float(np.mean(np.abs(y_true[hard_mask] - y_pred[hard_mask]))) if hard_mask.sum() > 0 else float("nan")

    return {
        "mode": mode,
        "n": int(len(y_true)),
        "mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson,
        "hard_mae": hard_mae, "hard_n": int(hard_mask.sum()),
    }


def run_eval() -> None:
    if str(CHEMPROP_EXE) != "chemprop" and not CHEMPROP_EXE.exists():
        raise FileNotFoundError(
            f"chemprop CLI not found: {CHEMPROP_EXE} (set env var CHEMPROP_EXE to full path)"
        )

    results = []
    for mode in MODES:
        try:
            r = eval_mode(mode)
            results.append(r)
            print(f"[ok] {mode}: MAE={r['mae']:.4f}")
        except Exception as e:
            print(f"[skip] {mode}: {e}")

    if not results:
        raise SystemExit("no results available")

    # Print summary table
    print("Strong OOD experiment (ChemProp, test MAE(LogS))")
    print(f"{'Mode':<14s} {'n':>6s} {'MAE':>10s} {'RMSE':>10s} {'R2':>10s} {'Pearson':>10s} {'hard_MAE':>10s}")
    for r in results:
        print(f"{r['mode']:<14s} {r['n']:>6d} {r['mae']:>10.4f} {r['rmse']:>10.4f} "
              f"{r['r2']:>10.4f} {r['pearson']:>10.4f} {r['hard_mae']:>10.4f}")

    # Save
    out_json = BASE / "chemprop_strong_ood_eval.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {out_json}")


# Entry point
def main() -> None:
    p = argparse.ArgumentParser(description="ChemProp strong OOD four-split experiment (prepare / eval)")
    p.add_argument("step", choices=["prepare", "eval"], help="prepare=build four CSVs; eval=evaluate four checkpoints")
    args = p.parse_args()

    if args.step == "prepare":
        run_prepare()
    else:
        run_eval()


if __name__ == "__main__":
    main()
