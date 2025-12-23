import torch.utils.data as data
import torch
from functools import partial
from typing import Optional
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from tqdm import tqdm
from dgl import DGLGraph
from GraphAug import DGLGraphFeatureMaskAug

# 在ESIDataset.__init__里新增一个参数，或者直接写死一个增强器
class ESIDataset(data.Dataset):
    def __init__(self, list_IDs, df, task='binary',
                 graph_aug: Optional[DGLGraphFeatureMaskAug] = None,
                 graph_aug_on_orig: bool = False,
                 graph_aug_on_aug: bool = True):
        self.list_IDs = list_IDs
        self.df = df
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.task = task
        self.graph_aug = graph_aug
        self.graph_aug_on_orig = graph_aug_on_orig
        self.graph_aug_on_aug = graph_aug_on_aug

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]

        # === 分子图（原） ===
        smi = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=smi,
                      node_featurizer=self.atom_featurizer,
                      edge_featurizer=self.bond_featurizer)

        # === 分子图（来自 SMILES_aug 的“字符串级”增强视图） ===
        smi_aug = self.df.iloc[index]['SMILES_aug']
        v_d_aug = self.fc(smiles=smi_aug,
                          node_featurizer=self.atom_featurizer,
                          edge_featurizer=self.bond_featurizer)

        # === 在图上再做一次“特征级”增强（可开关） ===
        if self.graph_aug is not None:
            if self.graph_aug_on_orig:
                v_d = self.graph_aug(v_d, smiles=smi)             # 叠加图增强
            if self.graph_aug_on_aug:
                v_d_aug = self.graph_aug(v_d_aug, smiles=smi_aug) # 叠加图增强

        # === 蛋白向量 ===
        v_p = torch.load(self.df.iloc[index]['Protein_Path'])
        v_p_aug = torch.load(self.df.iloc[index]['Protein_Path_aug'])

        # === 标签 ===
        if self.task == 'binary':
            y = self.df.iloc[index]["Y"]
            # y_env = 0  # 若没有 y_env，这里放占位；或改成返回 None
        else:
            y = self.df.iloc[index]["Score"]
            # y_env = self.df.iloc[index]["y_env"]

        return v_d, v_p, v_d_aug, v_p_aug, y
        # return v_d, v_p, y
