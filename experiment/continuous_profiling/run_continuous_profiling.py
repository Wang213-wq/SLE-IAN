# Continuous-composition solubility profiling: SLE_IAN inference + visualization on 101-point x1 grid per system
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from torch_geometric.data import Batch

RDLogger.DisableLog("rdApp.*")

# Paths and env
GEAT_ROOT = Path(__file__).resolve().parents[2]

if str(GEAT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEAT_ROOT))

import src.graph as _geat_graph_mod
import src.solvent as _geat_solvent_mod
import src.model as _geat_model_mod

geat_smiles_to_graph = _geat_graph_mod.smiles_to_graph
geat_build_cond = _geat_solvent_mod.build_condition_vector
SLE_IANModel = _geat_model_mod.SLE_IANModel

CORE_CSV = GEAT_ROOT / "dataset" / "public dataset and split" / "Bao2024.csv"
GEAT_CKPT = GEAT_ROOT / "checkpoints" / "external_sensitivity_group_edge" / "64" / "best.pt"

SYSTEM = ["Ketoconazole-1,4-Dioxane/Water", "Thiabendazole-Methanol/Water"]
TEMP_K = 298.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R_kJ = 8.314e-3   

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def geat_build_model(ckpt_path: Path) -> SLE_IANModel:
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
    ).to(DEVICE)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    print(f"  [load] {ckpt_path}  ep={state.get('epoch', '?')}  best_val_MAE={state.get('best_val_mae', '?')}")
    return model


# Step 1: Load raw experimental data and fix physical constants
print("Step 1: Load raw data for target system")
for sys_name in SYSTEM:
    core_df = pd.read_csv(CORE_CSV, encoding="gbk")
    sys_df = core_df[(core_df["Drug-solvent system"] == sys_name) & (core_df["Temperature(K)"] == TEMP_K)].reset_index(drop=True)
    print(f"System {sys_name} @ {TEMP_K}K: {len(sys_df)} expt points")
    assert len(sys_df) >= 2, "Need at least 2 experimental points"

    # Fix constants (take first row — same system + T means SMILES/MW/Tm/Hfus all identical)
    ROW = sys_df.iloc[0]
    DRUG_SMILES = str(ROW["Drug_SMILES"]).strip()
    SOLV1_SMILES = str(ROW["Solvent_1_SMILES"]).strip()
    SOLV2_SMILES = str(ROW["Solvent_2_SMILES"]).strip()
    SOLV1_NAME = str(ROW["Solvent_1"]).strip()
    SOLV2_NAME = str(ROW["Solvent_2"]).strip()
    DRUG_NAME = str(ROW["Drug"]).strip()
    Tm = float(ROW["Drug_Collected_Melting_temp (K)"])
    Hfus_kJ = float(ROW["Hfus_pred_kJ_mol"])
    MW_drug = float(ROW["Drug_MolWt"])
    MW_1 = float(ROW["Solvent_1_MolWt"])
    MW_2 = float(ROW["Solvent_2_MolWt"])

    print(f"  Drug:   {DRUG_NAME}  Tm={Tm:.2f}K  ΔHfus={Hfus_kJ:.3f}kJ/mol  MW={MW_drug:.2f}")
    print(f"  Solv1:  {SOLV1_NAME}  MW={MW_1:.3f}")
    print(f"  Solv2:  {SOLV2_NAME}  MW={MW_2:.3f}")

    # Ideal dissolution (van't Hoff)
    ideal_ln_x = (Hfus_kJ / R_kJ) * (1.0 / Tm - 1.0 / TEMP_K)
    ideal_x = np.exp(ideal_ln_x)
    print(f"  ideal_ln_x = {ideal_ln_x:.4f}  (x_ideal = {ideal_x:.6f})")

    # Experimental ground truth (for comparison)
    x1_true = sys_df["Solvent_mol_fraction"].values.astype(float)
    logS_true = sys_df["LogS"].values.astype(float)
    print(f"  x1 expt range: [{x1_true.min():.4f}, {x1_true.max():.4f}]")
    print(f"  LogS expt range: [{logS_true.min():.4f}, {logS_true.max():.4f}]")

    # Step 2: Build 101-point continuous grid
    print("Step 2: Build 101-point x1 grid")

    N_GRID = 101
    x1_grid_np = np.linspace(0.0, 1.0, N_GRID)
    x2_grid_np = 1.0 - x1_grid_np

    # 13-dim condition vector (temperature + x1 + x2 + 5-dim solvent1 descriptors + 5-dim solvent2 descriptors)
    cond_np = np.vstack([
        geat_build_cond(
            temperature_k=TEMP_K,
            x1=float(x1i),
            solvent1_smiles=SOLV1_SMILES,
            solvent2_smiles=SOLV2_SMILES,
        )
        for x1i in x1_grid_np
    ]).astype(np.float32)

    # Compute MW_solvent per row (average solvent MW)
    MW_solvent_np = x1_grid_np * MW_1 + x2_grid_np * MW_2

    # Mass fraction
    w1_np = x1_grid_np * MW_1 / MW_solvent_np

    # ideal_ln_x, mw_solute are shared across all samples
    ideal_ln_x_tensor = torch.tensor([ideal_ln_x] * N_GRID, dtype=torch.float32)
    mw_solute_tensor = torch.tensor([MW_drug] * N_GRID, dtype=torch.float32)
    mw_solvent_tensor = torch.tensor(MW_solvent_np, dtype=torch.float32)

    print(f"  x1_grid: {N_GRID} pts [{x1_grid_np[0]:.2f}, {x1_grid_np[-1]:.2f}]")
    print(f"  cond shape: {cond_np.shape}")
    print(f"  MW_solvent range: [{MW_solvent_np.min():.3f}, {MW_solvent_np.max():.3f}]")

    # Step 3: SLE_IAN inference
    print("Step 3: SLE_IAN inference")

    geat_model = geat_build_model(GEAT_CKPT)

    solute_graph = geat_smiles_to_graph(DRUG_SMILES)
    solv1_graph = geat_smiles_to_graph(SOLV1_SMILES)
    solv2_graph = geat_smiles_to_graph(SOLV2_SMILES)
    assert solute_graph is not None and solv1_graph is not None and solv2_graph is not None, "Molecular graph construction failed"

    # Clone 101x → three Batches
    solute_batch = Batch.from_data_list([solute_graph.clone() for _ in range(N_GRID)]).to(DEVICE)
    solv1_batch = Batch.from_data_list([solv1_graph.clone() for _ in range(N_GRID)]).to(DEVICE)
    solv2_batch = Batch.from_data_list([solv2_graph.clone() for _ in range(N_GRID)]).to(DEVICE)
    cond_tensor = torch.tensor(cond_np, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        geat_out = geat_model(
            solute_batch, solv1_batch, solv2_batch, cond_tensor,
            mw_solute=mw_solute_tensor.to(DEVICE),
            mw_solvent=mw_solvent_tensor.to(DEVICE),
            ideal_ln_x=ideal_ln_x_tensor.to(DEVICE),
        )

    logS_geat = geat_out["logS"].cpu().numpy()
    ln_gamma_geat = geat_out["ln_gamma"].cpu().numpy()
    ln_x_geat = geat_out["ln_x"].cpu().numpy()

    print(f"  SLE_IAN logS range: [{logS_geat.min():.4f}, {logS_geat.max():.4f}]")
    print(f"  SLE_IAN ln_gamma range: [{ln_gamma_geat.min():.4f}, {ln_gamma_geat.max():.4f}]")
    print(f"  SLE_IAN ln_x range: [{ln_x_geat.min():.4f}, {ln_x_geat.max():.4f}]")

    # Step 4: Aggregate and output
    print("Step 4: Aggregate + output")

    sys_rename = sys_name.replace("/", ",")
    out_dir = GEAT_ROOT / "results" / "continuous_profiling" / f"{sys_rename}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Align expt truth 
    logS_true_on_grid = np.full(N_GRID, np.nan)
    for i, x1g in enumerate(x1_grid_np):
        j = np.argmin(np.abs(x1_true - x1g))
        if abs(x1_true[j] - x1g) < 0.005:
            logS_true_on_grid[i] = logS_true[j]

    result_df = pd.DataFrame({
        "x1_Dioxane": x1_grid_np,
        "x2_Water": x2_grid_np,
        "w1_mass_frac_Dioxane": w1_np,
        "T_K": [TEMP_K] * N_GRID,
        "MW_solvent_avg": MW_solvent_np,
        "ideal_ln_x": [ideal_ln_x] * N_GRID,
        "ln_gamma_SLE_IAN": ln_gamma_geat,
        "ln_x_SLE_IAN": ln_x_geat,
        "logS_SLE_IAN": logS_geat,
        "logS_true_expt": logS_true_on_grid,
    })
    result_df.to_csv(out_dir / "grid_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"  [save] {out_dir / 'grid_predictions.csv'}  ({N_GRID} rows)")

    # Summary metrics
    summary = {
        "system": sys_name,
        "temperature_K": TEMP_K,
        "n_grid_points": N_GRID,
        "n_expt_points": len(sys_df),
        "drug": DRUG_NAME,
        "drug_SMILES": DRUG_SMILES,
        "Tm_K": Tm,
        "Hfus_kJ_mol": Hfus_kJ,
        "MW_drug": MW_drug,
        "MW_solvent1": MW_1,
        "MW_solvent2": MW_2,
        "ideal_ln_x": ideal_ln_x,
        "peak_logS_sle_ian": float(logS_geat.max()),
        "peak_x1_sle_ian": float(x1_grid_np[np.argmax(logS_geat)]),
        "expt_peak_logS": float(logS_true.max()),
        "expt_peak_x1": float(x1_true[np.argmax(logS_true)]),
    }
    # Compute MAE at experimental-point positions
    mask_valid = ~np.isnan(logS_true_on_grid)
    if mask_valid.sum() >= 2:
        mae = float(np.mean(np.abs(logS_geat[mask_valid] - logS_true_on_grid[mask_valid])))
        summary[f"MAE_on_{mask_valid.sum()}_expt_pts_SLE_IAN"] = mae
        print(f"  MAE (on {mask_valid.sum()} expt pts) SLE_IAN: {mae:.4f}")

    (out_dir / "profiling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [save] {out_dir / 'profiling_summary.json'}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(x1_grid_np, logS_geat, "b-", lw=2.5, label="SLE_IAN (ea_slots=64)")
    ax1.scatter(x1_true, logS_true, c="black", s=60, zorder=10, marker="o", label="Experimental value")
    ax1.set_ylabel("LogS (g/100g)", fontsize=12)
    ax1.set_title("Continuous Composition Solubility Curve", fontsize=13)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.01, 1.02, "(a)", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="bottom")

    # Mark SLE_IAN predicted peak
    idx_p = np.argmax(logS_geat)
    ax1.scatter([x1_grid_np[idx_p]], [logS_geat[idx_p]], color="b", marker="v", s=80, zorder=11)

    # Experimental peak
    idx_ep = np.argmax(logS_true)
    ax1.scatter([x1_true[idx_ep]], [logS_true[idx_ep]], c="black", marker="*", s=250, zorder=12)

    ax2 = axes[1]
    ax2.plot(x1_grid_np, ln_gamma_geat, "b-", lw=2.2, label="SLE_IAN ln γ")
    ax2.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax2.fill_between(x1_grid_np, 0, ln_gamma_geat, where=ln_gamma_geat > 0, alpha=0.2, color="red", label="positive deviation (excess G > 0)")
    ax2.fill_between(x1_grid_np, 0, ln_gamma_geat, where=ln_gamma_geat < 0, alpha=0.2, color="green", label="negative deviation (excess G < 0)")
    ax2.set_xlabel("mole fraction", fontsize=12)
    ax2.set_ylabel("ln γ", fontsize=12)
    ax2.set_title("Activity coefficient curve", fontsize=11)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.01, 1.02, "(b)", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="bottom")

    plt.tight_layout()
    png_path = out_dir / "grid_predictions.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] {png_path}")

    # Console summary
    print("Summary")
    print(f"{'Model':<12} {'Peak LogS':<12} {'Peak x₁':<10} {'LogS range':<28} {'MAE@expt':<10}")
    idx_p = np.argmax(logS_geat)
    peak_mae = summary.get(f"MAE_on_{mask_valid.sum()}_expt_pts_SLE_IAN", float("nan"))
    print(f"{'SLE_IAN':<12} {logS_geat[idx_p]:<12.3f} {x1_grid_np[idx_p]:<10.3f} [{logS_geat.min():.3f}, {logS_geat.max():.3f}]{'':<12} {peak_mae:<10.3f}")
    print(f"{'Expt (truth)':<12} {logS_true.max():<12.3f} {x1_true[np.argmax(logS_true)]:<10.3f} [{logS_true.min():.3f}, {logS_true.max():.3f}]")

    print("\nDone ✓")
