# Zero-overlap Drug-solvent system grouping split builder
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.console import configure_console_utf8

configure_console_utf8()

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config.project_config import DEFAULT


def load_core_table(core_path: Path) -> pd.DataFrame:
    data = pd.read_csv(core_path, encoding="gbk")
    if "Type" in data.columns:
        exclude = set(DEFAULT["exclude_types"])
        data = data[~data["Type"].isin(exclude)].copy()
    return data.reset_index(drop=True)


def split_binary_core(
    core_path: Path,
    *,
    random_state: int | None = None,
) -> pd.DataFrame:
    train_frac = DEFAULT["train_frac"]
    val_frac = DEFAULT["val_frac"]
    test_frac = DEFAULT["test_frac"]
    seed = DEFAULT["split_seed"] if random_state is None else random_state

    # Reset index only — no end-point filtering
    data = load_core_table(core_path).reset_index(drop=True)

    # Normalise temperature column name: current core file uses "Temperature(K)" but the rest
    # of the pipeline (parse_records, build_sol2026_split.py, chemprop preprocessing) expects
    # "Temperature (K)".
    if "Temperature(K)" in data.columns and "Temperature (K)" not in data.columns:
        data = data.rename(columns={"Temperature(K)": "Temperature (K)"})

    gcol = DEFAULT["split_group_col"]
    groups = data[gcol]
    test_val_frac = val_frac + test_frac

    gss = GroupShuffleSplit(n_splits=1, test_size=test_val_frac, random_state=seed)
    train_idx, hold_idx = next(gss.split(data, groups=groups))
    hold = data.iloc[hold_idx]

    val_ratio_in_hold = val_frac / test_val_frac
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=1.0 - val_ratio_in_hold, random_state=seed
    )
    val_rel, test_rel = next(gss2.split(hold, groups=hold[gcol]))

    split = pd.Series("train", index=data.index, dtype=str)
    split.iloc[hold_idx[val_rel]] = "val"
    split.iloc[hold_idx[test_rel]] = "test"

    out = data.copy()
    out["split"] = split.to_numpy()
    return out


def summarize(df: pd.DataFrame) -> None:
    gcol = DEFAULT["split_group_col"]
    counts = df["split"].value_counts().to_dict()
    sys_by = df.groupby("split")[gcol].nunique().to_dict()
    print(f"[info] rows={len(df)} | systems={df[gcol].nunique()}", flush=True)
    print(
        f"[info] train={counts.get('train', 0)} val={counts.get('val', 0)} "
        f"test={counts.get('test', 0)}",
        flush=True,
    )
    print(
        f"[info] systems train/val/test = {sys_by.get('train', 0)}/"
        f"{sys_by.get('val', 0)}/{sys_by.get('test', 0)}",
        flush=True,
    )

    tr = set(df.loc[df["split"] == "train", gcol])
    va = set(df.loc[df["split"] == "val", gcol])
    te = set(df.loc[df["split"] == "test", gcol])
    overlap = len(tr & va) + len(tr & te) + len(va & te)
    print(f"[check] system overlap = {overlap} (expected 0)", flush=True)
    if overlap:
        raise RuntimeError("System overlap detected — split is not valid")


def main() -> None:
    p = argparse.ArgumentParser(description="Build zero-overlap Drug-solvent system group split")
    p.add_argument("--core-csv", type=Path, default=Path(DEFAULT["core_csv"]))
    p.add_argument("--out", type=Path, default=Path(DEFAULT["group_split_csv"]))
    args = p.parse_args()

    core_path: Path = args.core_csv
    out_path: Path = args.out
    if not core_path.exists():
        raise FileNotFoundError(f"Core table not found: {core_path}")

    print(f"[in ] core : {core_path}", flush=True)
    df = split_binary_core(core_path)
    summarize(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[out] split: {out_path}", flush=True)


if __name__ == "__main__":
    main()
