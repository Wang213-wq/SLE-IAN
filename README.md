# SLE-IAN

Code and data release for the paper **SLE-IAN**: a physics-informed graph neural network for solubility prediction in binary solvent systems. SLE-IAN (Solubility via External Attention Network) combines a dual-channel GNN (solute and solvent graphs share one `TwinGraphEncoder`), gated bidirectional External Attention, and a thermodynamics readout layer (lnγ → hard SLE constraint → LogS). Six-dimensional bond features (bond-type one-hot: single / double / triple / aromatic + conjugation + ring membership) are injected into node features via a residual `edge_proj`.

* Internal dataset: 26,706 binary-solvent solubility records, system-level group split by `Drug-solvent system` (train 18,756 / val 4,028 / test 3,922; 56 held-out test systems with zero overlap), seed=42.
* External dataset: Sol2026 (18,058 records for zero-shot evaluation; partition: train 12,738 / val 2,765 / test 2,555).

## Project structure

```
SLE_IAN/
├── src/                 # model / encoder / external_attention / graph / data / train / sle_loss / evaluate / solvent
├── config/              # project_config (default hyperparameters & paths) / train_presets / console
├── data_split/          # split construction scripts
├── experiment/          # experiment entry scripts (one folder per experiment)
│   ├── ablation/architecture/run_phys_ablation.py
│   ├── ablation/train_strategy/run_ea_protocol_compare.py
│   └── sensitivity_analysis/run_phys_ea_slots_sensitivity.py
│   ├── strong_ood/
│   │   ├── SLE-IAN/run_strong_ood.py
│   │   └── chemprop/run_chemprop_strong_ood.py
│   ├── external_validation/run_external_validation.py
│   ├── feature_importance_analysis/run_gnn_explainer.py
│   ├── continuous_profiling/run_continuous_profiling.py
│   └── chemprop_baseline/
│       ├── run_chemprop_prepare.py
│       └── run_chemprop_eval.py
├── dataset/
│   ├── public dataset and split/
│   │   ├── Bao2024.csv                 # internal main table (26,706 records)
│   │   └── binary_group_split.csv      # system-level group split table
│   ├── private dataset and split/
│   │   ├── Sol2026.csv                 # Sol2026 external table
│   │   └── binary_group_split_sol2026.csv
│   └── strong_ood_splits/              # frozen strong-OOD splits (solvent / solute / scaffold / temperature)
├── checkpoints/
│   ├── external_sensitivity_group/32/best.pt                   # Phase-I best (S=32)
│   └── external_sensitivity_group_edge/64/best.pt              # final model (S=64)
├── requirements.txt
└── README.md
```

Each training run directory contains `results/summary.json` (final metrics + full config + split manifest), `history.json` (per-epoch), `predictions_test.csv`, and `split_manifest.json`. The sensitivity scan root additionally writes `ea_slots_sensitivity.json` (summary rows).

## Environment

```bash
pip install -r requirements.txt
# torch>=2.0, torch-geometric>=2.3, rdkit>=2023.3.1, numpy, pandas, scikit-learn, matplotlib
# Chemprop baseline additionally requires `pip install chemprop`.
# Set the CHEMPROP_EXE environment variable to point to the chemprop executable if it is not on PATH.
```

## Reproduction

Run all commands from the `SLE_IAN/` directory. Data tables live under `dataset/public dataset and split/` and `dataset/private dataset and split/`; their paths are configured in `config/project_config.py`.

|Experiment|Command|Output|
|-|-|-|
|Memory_size sensitivity — Phase I (scratch, no edge injection)|`python experiment/sensitivity_analysis/run_phys_ea_slots_sensitivity.py --yes`|`checkpoints/external_sensitivity_group/{16..256}/`|
|Architecture ablation (edge injection, S=64)|`python experiment/ablation/architecture/run_phys_ablation.py --yes`|`ablation_group_m64_edge/`|
|Ablation: edge-injection scratch vs. warm-start (no edge)|`python experiment/ablation/train_strategy/run_ea_protocol_compare.py --stream both --yes`|`protocol_compare_edge_scratch/`, `protocol_compare_warm_noedge/`, `protocol_compare_summary/`|
|Strong out-of-distribution (solvent / solute / scaffold / temperature)|`python experiment/strong_ood/SLE-IAN/run_strong_ood.py --yes`|`checkpoints/exp_geat_phys_ood_*/`|
|Sol2026 zero-shot evaluation|`python experiment/external_validation/run_external_validation.py`|`results/`|
|Continuous composition profiling|`python experiment/continuous_profiling/run_continuous_profiling.py`|`results/continuous_profiling/`|
|Feature importance (GNN-Explainer)|`python experiment/feature_importance_analysis/run_gnn_explainer.py`|`results/gnn_explainer_testset/`|

Phase II of the Memory_size sensitivity (edge injection + warm start) shares the final model configuration; it selected Memory_size=64, i.e. the released main model.

The final model is trained with `python -m src.train --preset exp_geat_phys_group_edge_ea64`. Chemprop baseline preparation and evaluation live under `experiment/chemprop_baseline/` and `experiment/strong_ood/chemprop/`.

## Baselines

All comparison models were re-trained by us using the original implementations released with their respective papers, on the same main table (`dataset/public dataset and split/Bao2024.csv`) and **our** system-level group split (`dataset/strong_ood_splits/`), not each paper's own split. None of their source code is bundled in this release.

* **Chemprop** — wrapper scripts included (`experiment/chemprop_baseline/`, `experiment/strong_ood/chemprop/`). Install with `pip install chemprop`. The scripts cover data preparation and evaluation only — the training step is not included, so supply your own checkpoints.
* **GCN** — Amiri & Khaleseh, "Predicting Drug Solubility in Binary Solvent Mixtures Using Graph Convolutional Networks: A Comprehensive Deep Learning Approach", *Sci Rep* (2025), doi:10.1038/s41598-025-28272-3.
* **XGBoost / LightGBM** — Bao et al., "Towards the prediction of drug solubility in binary solvent mixtures at various temperatures using machine learning", *J Cheminform* 16 (2024), doi:10.1186/s13321-024-00911-3.

## Data

* `dataset/public dataset and split/Bao2024.csv` — internal main table, from the same Bao et al. (2024) paper listed under Baselines.
* `dataset/private dataset and split/Sol2026.csv` — external dataset; 

## Two-stage weight loading

`train.py` loads the Phase-I checkpoint with `strict=False`, migrating only the shared `TwinGraphEncoder` (`encoder.*`); the EA memory shape mismatch (32→64) and the absent `edge_proj` are skipped automatically.

Final model loading example:

```python
import torch
from src.model import SLE_IANModel

model = SLE_IANModel(
    ea_slots=64, readout_mode="from_gamma", ea_gated=True,
    solvent_mode="x1_weighted", use_edge_injection=True,
)
state = torch.load("checkpoints/external_sensitivity_group_edge/64/best.pt", map_location="cpu")
model.load_state_dict(state["model_state"], strict=False)
```

