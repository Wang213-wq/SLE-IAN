# Prepare train/val/test splits for Chemprop baseline
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import load_core_table, filter_core, parse_records, group_split_records

OUT_DIR = ROOT / "chemprop_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def records_to_df(records) -> pd.DataFrame:
    rows = []
    for r in records:
        # Three-component SMILES: solute.solvent1.solvent2
        smiles = f"{r.drug_smiles}.{r.solvent1_smiles}.{r.solvent2_smiles}"
        rows.append(
            {
                "smiles": smiles,
                "LogS": r.logS,
                "temperature_K": r.temperature_k,
                "x1": r.x1,
            }
        )
    return pd.DataFrame(rows)


def write_split(df: pd.DataFrame, name: str) -> None:
    # Data columns: smiles + target + extra descriptors (temperature, x1)
    df[["smiles", "LogS", "temperature_K", "x1"]].to_csv(
        OUT_DIR / f"{name}.csv", index=False, encoding="utf-8"
    )
    print(f"[{name}] n={len(df)}  ->  {name}.csv")


def main():
    core_csv = ROOT / "dataset" / "public dataset and split" / "Bao2024.csv"
    df = pd.read_csv(core_csv, encoding="gbk")
    # Normalize temperature column if named "Temperature(K)" → "Temperature (K)"
    if "Temperature(K)" in df.columns and "Temperature (K)" not in df.columns:
        df = df.rename(columns={"Temperature(K)": "Temperature (K)"})
    df = filter_core(df)
    records = parse_records(df)
    train, val, test, manifest = group_split_records(records, seed=42)

    print(f"total records: {len(records)}")
    print(f"split: train={len(train)} val={len(val)} test={len(test)}")
    print(f"systems: train={manifest.n_systems_train} val={manifest.n_systems_val} test={manifest.n_systems_test}")
    print(f"train∩test systems: {manifest.train_system_overlap_with_test}")
    assert (len(train), len(val), len(test)) == (18756, 4028, 3922), "Split differs from main model!"

    frames = []
    for recs, name in [(train, "train"), (val, "val"), (test, "test")]:
        part = records_to_df(recs)
        write_split(part, name)
        part["split"] = name
        frames.append(part)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all.csv", index=False, encoding="utf-8")
    print(f"[all]   n={len(all_df)}  ->  all.csv (with split column)")
    print(f"\nOutput dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
