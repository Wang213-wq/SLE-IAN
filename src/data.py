# Core data pipeline: graph building + cond_13 + ideal_ln_x / Hfus / Tm / MW features
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from config.project_config import DEFAULT, RESULTS_DIR
from src.graph import smiles_to_graph
from src.sle_loss import ideal_ln_x_np
from src.solvent import build_condition_vector


@dataclass
class RowRecord:
    idx: int
    drug: str
    drug_smiles: str
    solvent1_smiles: str
    solvent2_smiles: str
    temperature_k: float
    x1: float
    logS: float
    system: str
    split: str
    hfus_j_per_mol: float
    tm_k: float
    mw_solute: float
    mw_solvent1: float
    mw_solvent2: float
    ideal_ln_x: float
    sle_valid: bool


@dataclass
class SplitManifest:
    mode: str
    seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    group_col: str
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    n_systems_train: int
    n_systems_val: int
    n_systems_test: int
    train_system_overlap_with_test: int
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_core_table(csv_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(csv_path or DEFAULT["core_csv"])
    df = None
    for enc in ("utf-8", "gbk", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        df = pd.read_csv(path, encoding="gbk")
    # Normalise temperature column name: core CSV uses "Temperature(K)" but the rest of the
    # pipeline (parse_records, build_sol2026_split, chemprop preprocessing) expects "Temperature (K)".
    if "Temperature(K)" in df.columns and "Temperature (K)" not in df.columns:
        df = df.rename(columns={"Temperature(K)": "Temperature (K)"})
    return df


def filter_core(df: pd.DataFrame) -> pd.DataFrame:
    exclude = set(DEFAULT["exclude_types"])
    out = df[~df["Type"].astype(str).isin(exclude)].copy()
    return out.reset_index(drop=True)


def _mixture_mw(x1: float, mw1: float, mw2: float) -> float:
    x1 = float(np.clip(x1, 0.0, 1.0))
    return x1 * float(mw1) + (1.0 - x1) * float(mw2)


def parse_records(df: pd.DataFrame) -> list[RowRecord]:
    records: list[RowRecord] = []
    for idx, row in df.iterrows():
        smi = str(row.get("Drug_SMILES", "")).strip()
        s1 = str(row.get("Solvent_1_SMILES", "")).strip()
        s2 = str(row.get("Solvent_2_SMILES", "")).strip()
        if not smi or not s1 or not s2:
            continue
        try:
            t = float(row["Temperature (K)"])
            x1 = float(row["Solvent_mol_fraction"])
            logS = float(row["LogS"])
            hfus_kj = float(row["Hfus_pred_kJ_mol"])
            tm = float(row["Drug_Collected_Melting_temp (K)"])
            mw_u = float(row["Drug_MolWt"])
            mw1 = float(row["Solvent_1_MolWt"])
            mw2 = float(row["Solvent_2_MolWt"])
        except (TypeError, ValueError, KeyError):
            continue
        if not all(np.isfinite(v) for v in (t, x1, logS, hfus_kj, tm, mw_u, mw1, mw2)):
            continue
        hfus_j = hfus_kj * 1000.0
        ideal = ideal_ln_x_np(t, tm, hfus_j)
        sle_ok = bool(np.isfinite(ideal))
        records.append(
            RowRecord(
                idx=int(idx),
                drug=str(row["Drug"]),
                drug_smiles=smi,
                solvent1_smiles=s1,
                solvent2_smiles=s2,
                temperature_k=t,
                x1=x1,
                logS=logS,
                system=str(row.get("Drug-solvent system", "")),
                split=str(row.get("Type", "")),
                hfus_j_per_mol=hfus_j,
                tm_k=tm,
                mw_solute=mw_u,
                mw_solvent1=mw1,
                mw_solvent2=mw2,
                ideal_ln_x=float(ideal) if sle_ok else 0.0,
                sle_valid=sle_ok,
            )
        )
    return records


def group_split_records(
    records: list[RowRecord],
    *,
    seed: int = DEFAULT["split_seed"],
    train_frac: float = DEFAULT["train_frac"],
    val_frac: float = DEFAULT["val_frac"],
    test_frac: float = DEFAULT["test_frac"],
    group_col: str = DEFAULT["split_group_col"],
) -> tuple[list[RowRecord], list[RowRecord], list[RowRecord], SplitManifest]:
    """GroupShuffleSplit on Drug-solvent system at 7:1.5:1.5 (zero-system-overlap partition)."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    df = pd.DataFrame(
        {
            "i": np.arange(len(records)),
            "system": [r.system for r in records],
        }
    )
    groups = df["system"]
    hold_frac = val_frac + test_frac
    gss = GroupShuffleSplit(n_splits=1, test_size=hold_frac, random_state=seed)
    train_pos, hold_pos = next(gss.split(df, groups=groups))
    hold = df.iloc[hold_pos]
    val_ratio_in_hold = val_frac / hold_frac
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=1.0 - val_ratio_in_hold, random_state=seed
    )
    val_rel, test_rel = next(gss2.split(hold, groups=hold["system"]))
    train_idx = train_pos.tolist()
    val_idx = hold_pos[val_rel].tolist()
    test_idx = hold_pos[test_rel].tolist()

    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]
    for r in train:
        r.split = "train"
    for r in val:
        r.split = "val"
    for r in test:
        r.split = "test"

    tr_sys = {r.system for r in train}
    va_sys = {r.system for r in val}
    te_sys = {r.system for r in test}
    manifest = SplitManifest(
        mode="group",
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        group_col=group_col,
        n_total=len(records),
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        n_systems_train=len(tr_sys),
        n_systems_val=len(va_sys),
        n_systems_test=len(te_sys),
        train_system_overlap_with_test=len(tr_sys & te_sys),
        description=(
            f"GroupShuffleSplit on {group_col}; "
            f"{train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%}; seed={seed}"
        ),
    )
    return train, val, test, manifest


def save_split_artifacts(
    train: list[RowRecord],
    val: list[RowRecord],
    test: list[RowRecord],
    manifest: SplitManifest,
    *,
    out_dir: Path | None = None,
) -> None:
    out = Path(out_dir or RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)

    def _rows(name: str, recs: list[RowRecord]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "split": name,
                "row_idx": [r.idx for r in recs],
                "drug": [r.drug for r in recs],
                "system": [r.system for r in recs],
                "logS": [r.logS for r in recs],
                "sle_valid": [r.sle_valid for r in recs],
            }
        )

    pd.concat(
        [_rows("train", train), _rows("val", val), _rows("test", test)],
        ignore_index=True,
    ).to_csv(out / "split_indices.csv", index=False, encoding="utf-8-sig")
    with open(out / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)


class GEATDataset(Dataset):
    def __init__(self, records: list[RowRecord]) -> None:
        self.records = records
        self._cache: dict[str, Data | None] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _graph(self, smiles: str) -> Data:
        if smiles not in self._cache:
            self._cache[smiles] = smiles_to_graph(smiles)
        g = self._cache[smiles]
        if g is None:
            raise ValueError(f"Cannot build graph for SMILES: {smiles}")
        return g.clone()

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        cond = build_condition_vector(
            temperature_k=rec.temperature_k,
            x1=rec.x1,
            solvent1_smiles=rec.solvent1_smiles,
            solvent2_smiles=rec.solvent2_smiles,
        )
        mw_mix = _mixture_mw(rec.x1, rec.mw_solvent1, rec.mw_solvent2)
        return {
            "solute": self._graph(rec.drug_smiles),
            "solvent1": self._graph(rec.solvent1_smiles),
            "solvent2": self._graph(rec.solvent2_smiles),
            "cond": torch.tensor(cond, dtype=torch.float32),
            "logS": torch.tensor(rec.logS, dtype=torch.float32),
            "ideal_ln_x": torch.tensor(rec.ideal_ln_x, dtype=torch.float32),
            "sle_valid": torch.tensor(rec.sle_valid, dtype=torch.bool),
            "mw_solute": torch.tensor(rec.mw_solute, dtype=torch.float32),
            "mw_solvent": torch.tensor(mw_mix, dtype=torch.float32),
            "tm_k": torch.tensor(rec.tm_k, dtype=torch.float32),
            "hfus_j_per_mol": torch.tensor(rec.hfus_j_per_mol, dtype=torch.float32),
            "rec": rec,
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "solute": Batch.from_data_list([b["solute"] for b in batch]),
        "solvent1": Batch.from_data_list([b["solvent1"] for b in batch]),
        "solvent2": Batch.from_data_list([b["solvent2"] for b in batch]),
        "cond": torch.stack([b["cond"] for b in batch]),
        "logS": torch.stack([b["logS"] for b in batch]),
        "ideal_ln_x": torch.stack([b["ideal_ln_x"] for b in batch]),
        "sle_valid": torch.stack([b["sle_valid"] for b in batch]),
        "mw_solute": torch.stack([b["mw_solute"] for b in batch]),
        "mw_solvent": torch.stack([b["mw_solvent"] for b in batch]),
        "tm_k": torch.stack([b["tm_k"] for b in batch]),
        "hfus_j_per_mol": torch.stack([b["hfus_j_per_mol"] for b in batch]),
        "recs": [b["rec"] for b in batch],
    }


def _canonical(smi: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol is None:
        return str(smi).strip()
    return Chem.MolToSmiles(mol)


def row_key(
    *,
    drug_smi: str,
    s1: str,
    s2: str,
    t: float,
    x1: float,
    logS: float,
) -> tuple:
    return (
        _canonical(drug_smi),
        _canonical(s1),
        _canonical(s2),
        round(float(t), 6),
        round(float(x1), 8),
        round(float(logS), 8),
    )


def apply_group_split_labels(
    records: list[RowRecord],
    split_csv: Path | str | None = None,
) -> list[RowRecord]:
    """Overwrite each record's .split label from a pre-built binary_group_split table."""
    path = Path(split_csv or DEFAULT["group_split_csv"])
    split_df = pd.read_csv(path)
    if len(split_df) != len(records):
        raise RuntimeError(
            f"Split table length mismatch: {len(split_df)} split labels vs {len(records)} records"
        )
    for i, r in enumerate(records):
        r.split = str(split_df.loc[i, "split"])
    return records


def records_to_split_buckets(
    records: list[RowRecord],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    mode: str,
) -> tuple[list[RowRecord], list[RowRecord], list[RowRecord], SplitManifest]:
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    test = [r for r in records if r.split == "test"]
    if not (train and val and test):
        raise RuntimeError(
            f"Split buckets empty: train={len(train)} val={len(val)} test={len(test)}"
        )
    tr_sys = {r.system for r in train}
    va_sys = {r.system for r in val}
    te_sys = {r.system for r in test}
    manifest = SplitManifest(
        mode=mode,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        group_col=DEFAULT["split_group_col"],
        n_total=len(records),
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        n_systems_train=len(tr_sys),
        n_systems_val=len(va_sys),
        n_systems_test=len(te_sys),
        train_system_overlap_with_test=len(tr_sys & te_sys),
        description=mode,
    )
    return train, val, test, manifest


def build_datasets_from_split_indices(
    split_indices_csv: Path | str,
    *,
    csv_path: Path | str | None = None,
    mode_name: str = "external_indices",
    seed: int = DEFAULT["split_seed"],
    artifact_dir: Path | None = None,
) -> tuple[GEATDataset, GEATDataset, GEATDataset, SplitManifest]:
    """Build datasets from an external split_indices.csv (strong OOD split mode)."""
    path = Path(split_indices_csv)
    if not path.is_file():
        raise FileNotFoundError(f"split_indices not found: {path}")
    idx_df = pd.read_csv(path)
    if not {"split", "row_idx"}.issubset(idx_df.columns):
        raise ValueError(f"{path} requires columns split,row_idx")

    records = parse_records(filter_core(load_core_table(csv_path)))
    by_idx = {r.idx: r for r in records}
    buckets: dict[str, list[RowRecord]] = {"train": [], "val": [], "test": []}
    missing = 0
    for _, row in idx_df.iterrows():
        split = str(row["split"]).strip().lower()
        if split not in buckets:
            continue
        rid = int(row["row_idx"])
        rec = by_idx.get(rid)
        if rec is None:
            missing += 1
            continue
        buckets[split].append(rec)
    if missing:
        raise RuntimeError(f"{missing} rows in split_indices absent from GEAT records: {path}")
    train, val, test = buckets["train"], buckets["val"], buckets["test"]
    if not train or not test:
        raise RuntimeError(f"Empty split train={len(train)} test={len(test)}: {path}")

    tr_sys, va_sys, te_sys = (
        {r.system for r in train},
        {r.system for r in val},
        {r.system for r in test},
    )
    n = len(train) + len(val) + len(test)
    manifest = SplitManifest(
        mode=mode_name,
        seed=seed,
        train_frac=len(train) / max(n, 1),
        val_frac=len(val) / max(n, 1),
        test_frac=len(test) / max(n, 1),
        group_col="external_split_indices",
        n_total=n,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        n_systems_train=len(tr_sys),
        n_systems_val=len(va_sys),
        n_systems_test=len(te_sys),
        train_system_overlap_with_test=len(tr_sys & te_sys),
        description=f"external indices from {path.name} ({mode_name})",
    )
    if artifact_dir is not None:
        save_split_artifacts(train, val, test, manifest, out_dir=artifact_dir)
    return GEATDataset(train), GEATDataset(val), GEATDataset(test), manifest


def build_datasets(
    csv_path: Path | str | None = None,
    *,
    seed: int = DEFAULT["split_seed"],
    train_frac: float = DEFAULT["train_frac"],
    val_frac: float = DEFAULT["val_frac"],
    test_frac: float = DEFAULT["test_frac"],
    artifact_dir: Path | None = None,
    use_group_split: bool = True,
    group_split_csv: Path | str | None = None,
    split_indices_csv: Path | str | None = None,
    split_mode_name: str | None = None,
) -> tuple[GEATDataset, GEATDataset, GEATDataset, SplitManifest]:
    if split_indices_csv is not None:
        return build_datasets_from_split_indices(
            split_indices_csv,
            csv_path=csv_path,
            mode_name=split_mode_name or "external_indices",
            seed=seed,
            artifact_dir=artifact_dir,
        )

    df = filter_core(load_core_table(csv_path))
    records = parse_records(df)
    if not records:
        raise RuntimeError("No valid records parsed")

    if use_group_split:
        records = apply_group_split_labels(records, group_split_csv)
        mode = "group_full"
        train, val, test, manifest = records_to_split_buckets(
            records,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            mode=mode,
        )
    else:
        train, val, test, manifest = group_split_records(
            records,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )

    if artifact_dir is not None:
        save_split_artifacts(train, val, test, manifest, out_dir=artifact_dir)
    return GEATDataset(train), GEATDataset(val), GEATDataset(test), manifest
