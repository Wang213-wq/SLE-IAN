# GNNExplainer atom & bond feature importance analysis (per-test-sample explanation + aggregation)
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.utils import to_dense_batch

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.console import configure_console_utf8
configure_console_utf8()

import torch_geometric
torch_geometric.typing.WITH_TORCH_SCATTER = False

from src.graph import smiles_to_graph
from src.solvent import build_condition_vector
from src.model import SLE_IANModel


CORE_CSV = ROOT / "dataset" / "public dataset and split" / "Bao2024.csv"
GEAT_CKPT = ROOT / "checkpoints" / "external_sensitivity_group_edge" / "64" / "best.pt"

# Raw feature order
NODE_FEATURE_NAMES = [
    "Atomic_Number",       # 0
    "Period",              # 1
    "Hybridization_SP",   # 2
    "Hybridization_SP2",  # 3
    "Hybridization_SP3",  # 4
    "Formal_Charge",      # 5
    "Total_Valence",      # 6
    "Aromaticity",        # 7
    "Chirality",          # 8
    "HBD",                # 9
    "HBA",                # 10
    "Degree",             # 11
    "Num_Hs",             # 12
]
assert len(NODE_FEATURE_NAMES) == 13

BOND_FEATURE_NAMES = ["Single", "Double", "Triple", "Aromatic", "Conjugated", "In_Ring"]
assert len(BOND_FEATURE_NAMES) == 6

# Group features by physical semantics (sum masked dims within group → group importance, then pct-normalize)
NODE_FEATURE_GROUPS = [
    ("Atomic_Symbol",   [0, 1, 6]),   # Atomic_Number + Period + Total_Valence: intrinsic elemental properties
    ("Hybridization",   [2, 3, 4]),    # SP / SP2 / SP3 three one-hot merged
    ("Formal_Charge",   [5]),
    ("Aromaticity",     [7]),
    ("Chirality",       [8]),
    ("HBD",             [9]),
    ("HBA",             [10]),
    ("Neighbor_Count",  [11]),           # Degree
    ("Hydrogen_Count",   [12]),           # Num_Hs
]
BOND_FEATURE_GROUPS = [
    ("Bond_Type",   [0, 1, 2, 3]),      # Single / Double / Triple / Aromatic four one-hot merged
    ("Conjugated", [4]),
    ("Ring_Part",  [5]),                 # In_Ring
]

R_kJ = 8.314e-3  # kJ/(mol·K)

BOND_TYPE_TO_IDX = {0: "SINGLE", 1: "DOUBLE", 2: "TRIPLE", 3: "AROMATIC"}


# Wrap SLE_IAN's three-graph forward into a GNNExplainer-compatible single-graph interface
class SoluteExplainerWrapper(torch.nn.Module):

    def __init__(
        self,
        model: SLE_IANModel,
        solv1_data: Data,
        solv2_data: Data,
        cond: torch.Tensor,
        mw_solute: torch.Tensor,
        mw_solvent: torch.Tensor,
        ideal_ln_x: torch.Tensor,
    ):
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Precompute solvent embeddings (fixed, no gradient needed)
        with torch.no_grad():
            solv1_batch = Batch.from_data_list([solv1_data]).to(cond.device)
            solv2_batch = Batch.from_data_list([solv2_data]).to(cond.device)
            h_v1, m_v1 = model._encode_dense(solv1_batch)
            h_v2, m_v2 = model._encode_dense(solv2_batch)
            t_k = cond[:, 0]
            x1 = cond[:, 1]
            h_v, m_v = model._merge_solvent(h_v1, m_v1, h_v2, m_v2, x1)

        self.register_buffer("_h_v", h_v)
        self.register_buffer("_m_v", m_v)
        self.register_buffer("_cond", cond)
        self.register_buffer("_mw_s", mw_solute)
        self.register_buffer("_mw_v", mw_solvent)
        self.register_buffer("_ideal_lnx", ideal_ln_x)

    def _encode_solute(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.model.encoder

        # Derive edge_weight from edge_attr (mean of masked 6-dim features)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_weight = edge_attr.mean(dim=1)  # [num_edges]
        else:
            edge_weight = None

        # edge_injection: use (masked) edge_attr
        if edge_attr is not None and enc.edge_proj is not None and edge_attr.numel() > 0:
            ef = enc.edge_proj(edge_attr)
            src = edge_index[0]
            N = x.size(0)
            agg = x.new_zeros(N, ef.size(1))
            agg = agg.index_add(0, src, ef)
            ones = x.new_ones(src.size(0), 1)
            count = x.new_zeros(N, 1)
            count = count.index_add(0, src, ones)
            count = torch.clamp(count, min=1.0)
            x_inj = x + agg / count
        else:
            x_inj = x

        # GCNConv layers (pass edge_weight)
        h = x_inj
        for i, layer in enumerate(enc.layers):
            h = layer(h, edge_index, edge_weight=edge_weight)
            if i < len(enc.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=enc.dropout, training=enc.training)

        # to_dense_batch (single graph, all-zero batch)
        num_nodes = h.size(0)
        batch_idx = torch.zeros(num_nodes, dtype=torch.long, device=h.device)
        dense, mask = to_dense_batch(h, batch_idx)
        return dense, mask

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from src.sle_loss import ln_x_to_logS

        # 1) Encode solute (edge_attr already element-wise masked by edge_mask)
        h_u, m_u = self._encode_solute(x, edge_index, edge_attr)

        # 2) Use precomputed solvent embeddings
        h_v = self._h_v
        m_v = self._m_v

        # 3) External Attention
        if self.model.ea is not None:
            h_u, h_v = self.model.ea(h_u, m_u, h_v, m_v)

        # 4) Pool
        z_u = self.model._pool(h_u, m_u)
        z_v = self.model._pool(h_v, m_v)

        # 5) Cond + Fuse
        z_c = self.model.cond_mlp(self._cond)
        fused = self.model.fuse(torch.cat([z_u, z_v, z_c], dim=-1))

        # 6) Readout
        if self.model.readout_mode == "from_gamma":
            ln_gamma = self.model.head_ln_gamma(fused).squeeze(-1)
            ln_x = self._ideal_lnx - ln_gamma
            logS = ln_x_to_logS(ln_x, self._mw_s, self._mw_v)
        else:
            logS = self.model.head_logS(fused).squeeze(-1)

        return logS



def load_model(ckpt_path: Path, device: torch.device) -> SLE_IANModel:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck = state.get("config", {})
    model = SLE_IANModel(
        hidden_dim=ck.get("hidden_dim", 128),
        gnn_layers=ck.get("gnn_layers", 3),
        ea_slots=ck.get("ea_slots", 64),
        mlp_hidden=ck.get("mlp_hidden", 128),
        dropout=ck.get("dropout", 0.15),
        solvent_mode=ck.get("solvent_mode", "x1_weighted"),
        readout_mode=ck.get("readout_mode", "from_gamma"),
        ea_gated=ck.get("ea_gated", True),
        ea_gate_init=ck.get("ea_gate_init", -2.0),
        use_ea=ck.get("use_ea", True),
        use_edge_injection=ck.get("use_edge_injection", True),
    ).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    print(f"  [load] {ckpt_path}")
    print(f"         epoch={state.get('epoch','?')}  best_val_MAE={state.get('best_val_mae','?')}")
    print(f"         readout={ck.get('readout_mode')}  solvent={ck.get('solvent_mode')}  ea_slots={ck.get('ea_slots')}")
    return model


def build_sample_from_row(row: pd.Series, device: torch.device):
    # Build triple graph + cond + mw + ideal_ln_x from one row of data.
    drug_smi = str(row["Drug_SMILES"]).strip()
    s1_smi = str(row["Solvent_1_SMILES"]).strip()
    s2_smi = str(row["Solvent_2_SMILES"]).strip()
    Tm = float(row["Drug_Collected_Melting_temp (K)"])
    Hfus = float(row["Hfus_pred_kJ_mol"])
    mw_drug = float(row["Drug_MolWt"])
    mw_1 = float(row["Solvent_1_MolWt"])
    mw_2 = float(row["Solvent_2_MolWt"])
    temp_k = float(row["Temperature(K)"])
    x1 = float(row["Solvent_mol_fraction"])

    solute_data = smiles_to_graph(drug_smi)
    solv1_data = smiles_to_graph(s1_smi)
    solv2_data = smiles_to_graph(s2_smi)
    if solute_data is None or solv1_data is None or solv2_data is None:
        return None

    cond_np = build_condition_vector(
        temperature_k=temp_k, x1=x1,
        solvent1_smiles=s1_smi, solvent2_smiles=s2_smi,
    ).astype(np.float32)
    cond = torch.tensor(cond_np, device=device).unsqueeze(0)

    ideal_lnx = (Hfus / R_kJ) * (1.0 / Tm - 1.0 / temp_k)
    ideal_ln_x = torch.tensor([ideal_lnx], dtype=torch.float32, device=device)
    mw_solute = torch.tensor([mw_drug], dtype=torch.float32, device=device)
    mw_solvent = torch.tensor([x1 * mw_1 + (1 - x1) * mw_2], dtype=torch.float32, device=device)

    return {
        "solute_data": solute_data,
        "solv1_data": solv1_data,
        "solv2_data": solv2_data,
        "cond": cond,
        "mw_solute": mw_solute,
        "mw_solvent": mw_solvent,
        "ideal_ln_x": ideal_ln_x,
        "drug_smiles": drug_smi,
        "system": str(row["Drug-solvent system"]),
        "temp_k": temp_k,
        "x1": x1,
        "logS_true": float(row["LogS"]),
    }


def run_gnn_explainer(
    wrapper: SoluteExplainerWrapper,
    solute_data: Data,
    epochs: int,
    device: torch.device,
) -> tuple:
    x = solute_data.x.to(device)
    edge_index = solute_data.edge_index.to(device)
    edge_attr = solute_data.edge_attr.to(device) if solute_data.edge_attr is not None else None

    with torch.no_grad():
        orig_pred = wrapper(x, edge_index, edge_attr=edge_attr)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    explanation = explainer(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return explanation, orig_pred.item()


# Per-sample mask extraction
def extract_masks(explanation, solute_data: Data, drug_smiles: str) -> dict:
    mol = Chem.MolFromSmiles(drug_smiles)
    assert mol is not None

    node_mask = explanation.node_mask  # [n_atoms, 13]
    if hasattr(node_mask, "detach"):
        node_mask = node_mask.detach().cpu().numpy()

    edge_mask = explanation.edge_mask  # [num_edges]
    if hasattr(edge_mask, "detach"):
        edge_mask = edge_mask.detach().cpu().numpy()

    edge_attr = solute_data.edge_attr.cpu().numpy() if solute_data.edge_attr is not None else None

    return {
        "node_mask": node_mask,     # [n_atoms, 13]
        "edge_mask": edge_mask,     # [num_directed_edges]
        "edge_attr": edge_attr,     # [num_directed_edges, 6]
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
    }


# Visualization: 13 atom + 6 bond dims, merged into one panel figure
def plot_feature_ranking(node_df: pd.DataFrame, bond_df: pd.DataFrame,
                         out_path: Path, n_samples: int):
    def _panel(ax, df, title, colors, bar_h):
        n = len(df)
        df = df.sort_values("importance_mean")  # most important on top
        total = df["importance_mean"].sum()
        pct = (df["importance_mean"] / total * 100.0) if total > 0 else df["importance_mean"] * 0.0

        y = range(len(df))
        bars = ax.barh(y, pct, color=colors, edgecolor="black", linewidth=1.2, height=bar_h)
        ax.set_yticks(list(y))
        ax.set_yticklabels(df["feature"], fontsize=11)
        ax.set_ylim(-0.5, n - 0.5)  # each panel fills its space
        ax.set_xlabel("Relative Importance (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        xmax = max(pct.max() * 1.18, 5.0)
        ax.set_xlim(0, xmax)
        for bar, v in zip(bars, pct):
            ax.text(bar.get_width() + xmax * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}%", va="center", ha="left", fontsize=9, color="#222222")

        ax.xaxis.grid(True, linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        return pct

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(16, 7.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.32},
    )

    # Atom side (13 dims): viridis gradient, bottom (least) = bright yellow-green → top (most) = deep purple
    n_n = len(node_df)
    node_colors = [plt.cm.viridis(0.88 - 0.80 * i / max(n_n - 1, 1)) for i in range(n_n)]
    # Bond side (6 dims): same viridis ramp
    n_b = len(bond_df)
    bond_colors = [plt.cm.viridis(0.88 - 0.80 * i / max(n_b - 1, 1)) for i in range(n_b)]

    node_bar_h = 0.7
    bond_bar_h = 0.7 * n_b / n_n  # 6/13 ratio, matches atom-side visual thickness
    _panel(ax_a, node_df, "Atom Feature Importance Ranking", node_colors, node_bar_h)
    _panel(ax_b, bond_df, "Bond Feature Importance Ranking", bond_colors, bond_bar_h)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description="GNNExplainer test-set atom/bond feature importance analysis")
    parser.add_argument("--checkpoint", type=str, default=str(GEAT_CKPT), help="Model weights path")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Test-set sample count (-1 = all test samples, very slow)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling random seed")
    parser.add_argument("--epochs", type=int, default=200, help="GNNExplainer optimization epochs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--replot", action="store_true",
                        help="Skip re-inference; replot from existing CSVs only")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "gnn_explainer_testset"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --replot: replot from existing CSVs only
    if args.replot:
        print("GNNExplainer: replotting from existing CSVs")
        node_df = pd.read_csv(out_dir / "atom_feature_importance.csv")
        bond_df = pd.read_csv(out_dir / "bond_feature_importance.csv")
        n_samples = len(pd.read_csv(out_dir / "per_sample_summary.csv"))
        plot_feature_ranking(node_df, bond_df, out_dir / "feature_importance_ranking.png", n_samples)
        print(f"  n_samples = {n_samples}")
        print(f"  [save] {out_dir / 'feature_importance_ranking.png'}")
        print("\n13-dim atom feature importance:")
        ng = node_df.sort_values("importance_mean", ascending=False)
        for r in ng.itertuples():
            print(f"  {r.feature:<20s} {r.importance_mean / ng['importance_mean'].sum() * 100:6.2f}%")
        print(f"\n6-dim bond feature importance:")
        bg = bond_df.sort_values("importance_mean", ascending=False)
        for r in bg.itertuples():
            print(f"  {r.feature:<20s} {r.importance_mean / bg['importance_mean'].sum() * 100:6.2f}%")
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("GNNExplainer: test-set atom & bond feature importance analysis")

    # 1. Load model
    print("\n[1] Load model")
    model = load_model(Path(args.checkpoint), device)

    # 2. Read test set
    print("\n[2] Read test set")
    core_df = pd.read_csv(CORE_CSV, encoding="gbk")
    test_df = core_df[core_df["Type"] == "Test"].reset_index(drop=True)
    print(f"  Test split has {len(test_df)} rows")
    if args.max_samples > 0 and args.max_samples < len(test_df):
        test_df = test_df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"  Random sample {args.max_samples} rows (seed={args.seed})")

    # 3. Run GNNExplainer per sample + aggregate
    print(f"\n[3] Per-sample GNNExplainer ({len(test_df)} samples × {args.epochs} epochs)")
    node_feat_sum = np.zeros(13)          # 13-dim atom importance accumulator
    node_feat_sq = np.zeros(13)
    bond_feat_sum = np.zeros(6)           # 6-dim bond importance accumulator
    bond_feat_sq = np.zeros(6)
    per_sample_rows = []
    n_ok, n_fail = 0, 0

    out_dir = ROOT / "results" / "gnn_explainer_testset"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in test_df.iterrows():
        sample = build_sample_from_row(row, device)
        if sample is None:
            n_fail += 1
            continue
        try:
            wrapper = SoluteExplainerWrapper(
                model, sample["solv1_data"], sample["solv2_data"],
                sample["cond"], sample["mw_solute"], sample["mw_solvent"], sample["ideal_ln_x"],
            ).to(device)
            wrapper.eval()

            explanation, orig_pred = run_gnn_explainer(
                wrapper, sample["solute_data"], args.epochs, device
            )
            masks = extract_masks(explanation, sample["solute_data"], sample["drug_smiles"])
        except Exception as e:
            print(f"  [skip] {sample['system']} @ {sample['temp_k']}K x1={sample['x1']:.3f}: {type(e).__name__}: {e}")
            n_fail += 1
            continue

        node_mask = masks["node_mask"]                       # [n_atoms, 13]
        edge_mask = masks["edge_mask"]                       # [num_edges]
        edge_attr = masks["edge_attr"]                       # [num_edges, 6]

        # 13-dim atom features: average across atoms → per-sample 13-dim importance
        nf = node_mask.mean(axis=0)                          # [13]
        node_feat_sum += nf
        node_feat_sq += nf ** 2

        # 6-dim bond features: for each feature j, aggregate with edge_attr[:,j] weighting
        if edge_attr is not None and edge_attr.shape[0] > 0:
            bf = np.zeros(6)
            for j in range(6):
                col = edge_attr[:, j]
                denom = col.sum()
                if denom > 0:
                    bf[j] = float(np.sum(edge_mask * col) / denom)
                else:
                    bf[j] = 0.0
        else:
            bf = np.zeros(6)
        bond_feat_sum += bf
        bond_feat_sq += bf ** 2

        n_ok += 1

        per_sample_rows.append({
            "system": sample["system"],
            "temperature_K": sample["temp_k"],
            "x1": sample["x1"],
            "drug_SMILES": sample["drug_smiles"],
            "num_atoms": masks["num_atoms"],
            "num_bonds": masks["num_bonds"],
            "logS_true": sample["logS_true"],
            "logS_pred": orig_pred,
            "abs_error": abs(orig_pred - sample["logS_true"]),
        })

        if (idx + 1) % 10 == 0 or idx == len(test_df) - 1:
            print(f"  progress {idx+1}/{len(test_df)}  ok {n_ok}  skipped {n_fail}")

    if n_ok == 0:
        print("No successful samples — exiting")
        return

    # 4. Aggregate results
    print(f"\n[4] Aggregate ({n_ok} ok / {n_fail} skipped)")
    node_mean = node_feat_sum / n_ok
    node_std = np.sqrt(np.maximum(node_feat_sq / n_ok - node_mean ** 2, 0.0))
    bond_mean = bond_feat_sum / n_ok
    bond_std = np.sqrt(np.maximum(bond_feat_sq / n_ok - bond_mean ** 2, 0.0))

    node_df = pd.DataFrame({
        "feature": NODE_FEATURE_NAMES,
        "importance_mean": node_mean,
        "importance_std": node_std,
    }).sort_values("importance_mean", ascending=False)
    node_df.to_csv(out_dir / "atom_feature_importance.csv", index=False, encoding="utf-8-sig")
    print(f"  [save] {out_dir / 'atom_feature_importance.csv'}")

    bond_df = pd.DataFrame({
        "feature": BOND_FEATURE_NAMES,
        "importance_mean": bond_mean,
        "importance_std": bond_std,
    }).sort_values("importance_mean", ascending=False)
    bond_df.to_csv(out_dir / "bond_feature_importance.csv", index=False, encoding="utf-8-sig")
    print(f"  [save] {out_dir / 'bond_feature_importance.csv'}")

    per_df = pd.DataFrame(per_sample_rows)
    per_df.to_csv(out_dir / "per_sample_summary.csv", index=False, encoding="utf-8-sig")
    print(f"  [save] {out_dir / 'per_sample_summary.csv'}")

    # JSON summary
    summary = {
        "n_test_total": int(len(core_df[core_df['Type'] == 'Test'])),
        "n_analyzed": n_ok,
        "n_failed": n_fail,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "epochs": args.epochs,
        "checkpoint": str(args.checkpoint),
        "atom_feature_importance": node_df.to_dict(orient="records"),
        "bond_feature_importance": bond_df.to_dict(orient="records"),
        "per_sample_MAE": float(per_df["abs_error"].mean()),
    }
    (out_dir / "explainer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [save] {out_dir / 'explainer_summary.json'}")

    # 5. Visualization (13 atom + 6 bond dims, merged panel figure)
    print("\n[5] Visualize")
    plot_feature_ranking(node_df, bond_df, out_dir / "feature_importance_ranking.png", n_ok)
    print(f"  [save] {out_dir / 'feature_importance_ranking.png'}")

    # 6. Console summary
    print("Summary")
    print(f"Samples analyzed: {n_ok} (skipped {n_fail})  sampling seed={args.seed}")
    print(f"Per-sample prediction MAE: {per_df['abs_error'].mean():.4f}")
    print(f"\n13-dim atom feature importance (%):")
    ng = node_df.sort_values("importance_mean", ascending=False)
    for r in ng.itertuples():
        print(f"  {r.feature:<20s} {r.importance_mean / ng['importance_mean'].sum() * 100:6.2f}%")
    print(f"\n6-dim bond feature importance (%):")
    bg = bond_df.sort_values("importance_mean", ascending=False)
    for r in bg.itertuples():
        print(f"  {r.feature:<20s} {r.importance_mean / bg['importance_mean'].sum() * 100:6.2f}%")
    print(f"\nOutput dir: {out_dir}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
