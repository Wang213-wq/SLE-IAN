# RDKit molecular graph builder: 13-dim node features + 6-dim edge features
from __future__ import annotations

from functools import lru_cache

import torch
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from torch_geometric.data import Data

_PT = GetPeriodicTable()

HYBRID_MAP = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
}

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def _atom_is_hbd(atom: Chem.Atom) -> float:
    sym = atom.GetSymbol()
    if sym in {"O", "N", "S"} and atom.GetTotalNumHs() > 0:
        return 1.0
    return 0.0


def _atom_is_hba(atom: Chem.Atom) -> float:
    return 1.0 if atom.GetSymbol() in {"N", "O", "F", "S"} else 0.0


def _atom_features(atom: Chem.Atom) -> list[float]:
    hyb = HYBRID_MAP.get(atom.GetHybridization(), 2)
    hyb_onehot = [0.0, 0.0, 0.0]
    hyb_onehot[hyb] = 1.0
    return [
        atom.GetAtomicNum() / 100.0,
        float(_PT.GetRow(atom.GetAtomicNum())) / 7.0,
        *hyb_onehot,
        atom.GetFormalCharge() / 2.0,
        float(atom.GetTotalValence()) / 8.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0,
        _atom_is_hbd(atom),
        _atom_is_hba(atom),
        atom.GetDegree() / 6.0,
        atom.GetTotalNumHs() / 4.0,
    ]


def _bond_features(bond: Chem.Bond) -> list[float]:
    bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
    one_hot = [0.0, 0.0, 0.0, 0.0]
    one_hot[bt] = 1.0
    return one_hot + [
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]


@lru_cache(maxsize=16384)
def smiles_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None

    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    if x.numel() == 0:
        return None

    edge_index: list[list[int]] = [[], []]
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        edge_attr += [feat, feat]

    if not edge_index[0]:
        edge_index = [[0], [0]]
        edge_attr = [[0.0] * 6]

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    data.num_nodes = x.size(0)
    return data
