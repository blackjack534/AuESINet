# requirements: rdkit, dgllife, dgl, torch
from typing import List, Optional, Set
import numpy as np
import torch
from rdkit import Chem

# 一些核心官能团（可按需增删）
SMARTS_CORE = [
    "C=O",          # 羰基
    "NC=O",         # 酰胺
    "c1ccccc1",     # 苯环
    "P(=O)(O)O",    # 磷酸
    "S(=O)(=O)O",   # 磺酸
]

def _core_atom_mask_from_smiles(smiles: str, smarts_list: List[str]) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    core = np.zeros(n, dtype=bool)
    for s in smarts_list:
        patt = Chem.MolFromSmarts(s)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            for idx in match:
                core[idx] = True
    return core  # True 表示“核心原子”，不做mask

class DGLGraphFeatureMaskAug:
    """
    在不改变图拓扑的前提下，对节点/边特征进行随机掩蔽（置零）。
    - 保护核心官能团：避免掩蔽活性位点附近原子/键
    - 仅操作 g.ndata['h'], g.edata['e'] (Canonical{Atom,Bond}Featurizer 默认写入)
    """
    def __init__(self,
                 atom_mask_ratio: float = 0.1,
                 bond_mask_ratio: float = 0.1,
                 protect_core: bool = True,
                 smarts_core: Optional[List[str]] = None,
                 mask_value: float = 0.0,
                 inplace: bool = False):
        self.atom_mask_ratio = atom_mask_ratio
        self.bond_mask_ratio = bond_mask_ratio
        self.protect_core = protect_core
        self.smarts_core = smarts_core if smarts_core is not None else SMARTS_CORE
        self.mask_value = mask_value
        self.inplace = inplace  # True: 原地改；False: clone 一份改

    @torch.no_grad()
    def __call__(self, g, smiles: Optional[str] = None):
        # 准备图
        if not self.inplace:
            g = g.clone()
        assert 'h' in g.ndata, "node features not found (expect g.ndata['h'])."
        assert 'e' in g.edata, "edge features not found (expect g.edata['e'])."

        N = g.num_nodes()
        E = g.num_edges()
        node_feat = g.ndata['h']          # (N, F_node)
        edge_feat = g.edata['e']          # (E, F_edge)

        # 保护核心原子（需要 SMILES）
        protected = np.zeros(N, dtype=bool)
        if self.protect_core and smiles is not None:
            m = _core_atom_mask_from_smiles(smiles, self.smarts_core)
            if m is not None and len(m) == N:
                protected = m

        # 采样节点掩蔽集合（只在非核心）
        cand_nodes = np.where(~protected)[0]
        k_nodes = int(round(self.atom_mask_ratio * len(cand_nodes)))
        if k_nodes > 0:
            mask_nodes = torch.from_numpy(
                np.random.choice(cand_nodes, size=k_nodes, replace=False)
            ).long()
            node_feat[mask_nodes] = self.mask_value

        # 采样边掩蔽集合：避免核心原子相关边
        if self.bond_mask_ratio > 0 and E > 0:
            src, dst = g.edges()  # (E,), (E,)
            src = src.cpu().numpy()
            dst = dst.cpu().numpy()
            safe_edge_mask = ~(protected[src] | protected[dst])
            cand_edges = np.where(safe_edge_mask)[0]
            k_edges = int(round(self.bond_mask_ratio * len(cand_edges)))
            if k_edges > 0:
                mask_edges = torch.from_numpy(
                    np.random.choice(cand_edges, size=k_edges, replace=False)
                ).long()
                edge_feat[mask_edges] = self.mask_value

        # 写回
        g.ndata['h'] = node_feat
        g.edata['e'] = edge_feat
        return g
