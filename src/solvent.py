# Solvent condition vector (13 dims: 5 per solvent + x1/x2 + temperature)
from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

REICHARDT_POL: dict[str, float] = {
    "O": 1.00,
    "CO": 0.76,
    "CCO": 0.65,
    "CC(C)O": 0.55,
    "CCCO": 0.52,
    "CCCCO": 0.47,
    "CC(C)CO": 0.50,
    "CC#N": 0.46,
    "CC(=O)C": 0.36,
    "CCC(=O)C": 0.34,
    "CC(C)CC(=O)C": 0.32,
    "CC(=O)O": 0.65,
    "C(=O)O": 0.72,
    "CCOC(=O)C": 0.30,
    "CCCCOC(=O)C": 0.28,
    "CC(=O)OC": 0.29,
    "CCCOC(=O)C": 0.28,
    "CN(C)C=O": 0.39,
    "CC(=O)N(C)C": 0.38,
    "CS(=O)C": 0.44,
    "C1CCOC1": 0.22,
    "C1COCCO1": 0.20,
    "C(CO)O": 0.51,
    "CC(CO)O": 0.48,
    "COCCO": 0.42,
    "COCCOC": 0.35,
    "CCOCCO": 0.38,
    "CCCOCCO": 0.34,
    "CCCCOCCO": 0.32,
    "COCCOCCOC": 0.30,
    "C(Cl)Cl": 0.31,
    "C(Cl)(Cl)Cl": 0.25,
    "C(CCl)Cl": 0.27,
    "CCCCCC": 0.00,
    "CCCCCCC": 0.00,
    "C1CCCCC1": 0.02,
    "CC1=CC=CC=C1": 0.10,
    "c1ccccc1": 0.11,
    "C1=CC=C(C=C1)CO": 0.42,
    "CC(=O)OCC1=CC=CC=C1": 0.28,
    "C1=CC=C(C=C1)COC(=O)C2=CC=CC=C2": 0.18,
    "CN1CCCC1=O": 0.38,
    "C1CCC(=O)C1": 0.33,
    "CC1=CC(=O)CC(C1)(C)C": 0.26,
    "C1COCCN1C=O": 0.40,
}


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return str(smiles).strip()
    return Chem.MolToSmiles(mol)


def estimate_polarity(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.5
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Crippen.MolLogP(mol)
    return float(min(1.0, (tpsa / 100.0) * (1.0 / (1.0 + max(logp, 0.0)))))


@lru_cache(maxsize=512)
def solvent_descriptor_vector(smiles: str) -> tuple[float, float, float, float, float]:
    smi = canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (0.0, 0.5, 0.0, 0.0, 0.0)

    logp = Crippen.MolLogP(mol) / 5.0
    hbd = rdMolDescriptors.CalcNumHBD(mol) / 5.0
    hba = rdMolDescriptors.CalcNumHBA(mol) / 10.0
    mw = Descriptors.MolWt(mol) / 200.0
    pol = REICHARDT_POL.get(smi, estimate_polarity(smi))
    return (float(logp), float(pol), float(hbd), float(hba), float(mw))


def build_condition_vector(
    *,
    temperature_k: float,
    x1: float,
    solvent1_smiles: str,
    solvent2_smiles: str,
) -> np.ndarray:
    x1 = float(np.clip(x1, 0.0, 1.0))
    x2 = 1.0 - x1
    t_norm = (float(temperature_k) - 273.0) / 100.0
    d1 = solvent_descriptor_vector(solvent1_smiles)
    d2 = solvent_descriptor_vector(solvent2_smiles)
    return np.array([t_norm, x1, x2, *d1, *d2], dtype=np.float32)
