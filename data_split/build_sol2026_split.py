# Sol2026 external-dataset format normalisation + group_split table builder
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config.console import configure_console_utf8
from config.project_config import DEFAULT

configure_console_utf8()

from src.data import filter_core, load_core_table, parse_records  # noqa: E402

# HFUS source path + external-dataset output path
SOL2026_HFUS = ROOT / "dataset" / "Sol2026_hfus.csv"
OUT_DIR = ROOT / "dataset" / "private dataset and split"
OUT_CORE = OUT_DIR / "Sol2026.csv"
OUT_SPLIT = OUT_DIR / "binary_group_split_sol2026.csv"

# Column rename map: Sol2026 HFUS → Bao2024-compatible schema
RENAME_MAP = {
    "Compound_Name": "Drug",
    "Solvent1": "Solvent_1",
    "SMILES_Solvent1": "Solvent_1_SMILES",
    "Solvent2": "Solvent_2",
    "SMILES_Solvent2": "Solvent_2_SMILES",
    "Temperature_K": "Temperature (K)",
    "LogS(g/100g)": "LogS", 
    "Solubility(g/100g)": "Solubility (g/100g)",
    "Fraction_Solvent1": "Solvent_mass_fraction",
    "Tm_corr_K": "Drug_Collected_Melting_temp (K)",
    "SMILES_Solute": "Drug_SMILES",
}

# Target Bao2024.csv columns
CORE_COLUMNS = [
    "Drug",
    "Solvent_1",
    "Solvent_1_SMILES",
    "Solvent_2",
    "Solvent_2_SMILES",
    "Temperature (K)",
    "Solvent_mass_fraction",
    "Solvent_mol_fraction",
    "Solubility (g/100g)",
    "LogS",
    "Class",
    "Drug-solvent system",
    "Type",
    "Drug_SMILES",
    "Hfus_pred_kJ_mol",
    "Drug_Collected_Melting_temp (K)",
    "Drug_MolWt",
    "Solvent_1_MolWt",
    "Solvent_2_MolWt",
]


def build_core_csv() -> pd.DataFrame:
    """Load Sol2026_hfus.csv, normalise columns, write Sol2026.csv."""
    if not SOL2026_HFUS.exists():
        raise FileNotFoundError(f"Sol2026_hfus.csv not found: {SOL2026_HFUS}")

    df = pd.read_csv(SOL2026_HFUS, encoding="utf-8-sig")
    print(f"[info] Sol2026 HFUS raw: {len(df)} rows, {len(df.columns)} cols", flush=True)

    df = df.rename(columns=RENAME_MAP).copy()

    # Fallback: if Tm_corr_K missing but Melting_temp (K) available
    if "Drug_Collected_Melting_temp (K)" not in df.columns or df["Drug_Collected_Melting_temp (K)"].isna().all():
        df["Drug_Collected_Melting_temp (K)"] = df.get("Melting_temp (K)")

    # Construct Drug-solvent system grouping key
    df["Drug-solvent system"] = (
        df["Drug"].astype(str) + "-"
        + df["Solvent_2"].astype(str) + "/"
        + df["Solvent_1"].astype(str)
    )

    df["Type"] = "Train"

    if "Class" not in df.columns:
        df["Class"] = ""

    # Verify all required columns are present
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after conversion: {missing}")

    df_out = df[CORE_COLUMNS].copy()
    print(f"[info] Converted Sol2026: {len(df_out)} rows", flush=True)

    # Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CORE, index=False, encoding="utf-8")
    print(f"[done] Wrote: {OUT_CORE}", flush=True)
    return df_out


def build_split_csv(df_core: pd.DataFrame) -> pd.DataFrame:
    """Run SLE_IAN filter_core → parse_records, then produce group_split train/val/test."""
    # Use SLE_IAN's filter_core + parse_records pipeline (identical to training-time preprocessing)
    df_filtered = filter_core(df_core.copy())
    records = parse_records(df_filtered)
    if not records:
        raise RuntimeError("parse_records returned empty list — check field completeness or NaN rate")
    print(f"[info] records after parse_records: {len(records)}", flush=True)

    # GroupShuffleSplit 7:1.5:1.5 (train vs val+test, then val vs test)
    systems = [r.system for r in records]
    n = len(records)
    n_train_target = int(round(n * 0.70))
    n_val_target = int(round(n * 0.15))
    _ = n - n_train_target - n_val_target  # test

    seed = DEFAULT["split_seed"]
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, rest_idx = next(gss1.split(range(n), groups=systems))

    # Second pass on held-out portion: val:test = 0.5:0.5 → 15%:15% overall
    rest_groups = [systems[i] for i in rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(rest_idx, groups=rest_groups))
    val_idx = [rest_idx[i] for i in val_rel]
    test_idx = [rest_idx[i] for i in test_rel]

    split_arr = [""] * n
    for i in train_idx:
        split_arr[i] = "train"
    for i in val_idx:
        split_arr[i] = "val"
    for i in test_idx:
        split_arr[i] = "test"

    # Verify zero system overlap
    tr_sys = {systems[i] for i in train_idx}
    va_sys = {systems[i] for i in val_idx}
    te_sys = {systems[i] for i in test_idx}
    print(
        f"[info] split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} | "
        f"systems={len(tr_sys)}/{len(va_sys)}/{len(te_sys)} | "
        f"train∩test={len(tr_sys & te_sys)}",
        flush=True,
    )

    record_indices = [r.idx for r in records]
    out = df_filtered.iloc[record_indices].reset_index(drop=True).copy()
    out["split"] = split_arr
    assert len(out) == len(records), f"Length mismatch: out={len(out)} records={len(records)}"
    out.to_csv(OUT_SPLIT, index=False, encoding="utf-8")
    print(f"[done] Wrote: {OUT_SPLIT} (rows={len(out)})", flush=True)
    return out


def main() -> None:
    print(f"=== build_sol2026_split ===", flush=True)
    df_core = build_core_csv()
    build_split_csv(df_core)
    print("\n[summary]", flush=True)
    print(f"  core : {OUT_CORE}", flush=True)
    print(f"  split: {OUT_SPLIT}", flush=True)


if __name__ == "__main__":
    main()
