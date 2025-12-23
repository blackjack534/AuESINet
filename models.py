import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from module.Encoder import *
from module.CN import *
from module.Transformer import CrossTransformerLayer

# PyTorch内置平均值池化模块使用示例
# 
# 1. torch.mean() - 简单平均值池化
# tensor1_pooled = torch.mean(tensor1, dim=1)  # (batch, dim)
# tensor2_pooled = torch.mean(tensor2, dim=1)  # (batch, dim)
#
# 2. torch.nn.AdaptiveAvgPool1d - 自适应平均值池化
# adaptive_pool = nn.AdaptiveAvgPool1d(1)
# tensor1_pooled = adaptive_pool(tensor1.transpose(1, 2)).squeeze(-1)  # (batch, dim)
#
# 3. torch.nn.functional.avg_pool1d - 1D平均值池化
# tensor1_pooled = F.avg_pool1d(tensor1.transpose(1, 2), kernel_size=tensor1.size(1)).squeeze(-1)
#
# 4. 带mask的平均值池化（推荐用于序列数据）
# mask_expanded = mask.unsqueeze(-1)
# x_masked = x.masked_fill(mask_expanded, 0)
# seq_lengths = mask.size(1) - mask.float().sum(dim=1, keepdim=True)
# avg_pool = x_masked.sum(dim=1) / torch.clamp(seq_lengths, min=1)

class MESI(nn.Module):
    def __init__(self, **config):
        super(MESI, self).__init__()
        """
        drug: features related to substrate;
        protein: features related to enzyme;     
        """
        
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_padding = config["DRUG"]["PADDING"]

        protein_in_dim = config["PROTEIN"]["IN_DIM"]
        protein_hidden_dim = config["PROTEIN"]["HIDDEN_DIM"]
        protein_target_dim = config["PROTEIN"]["TARGET_DIM"]
        
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        
        self.stage_num = config["STAGE"]["NUM"]
        self.ccfm_flag = config["STAGE"]["CCFM"]
        self.bcfm_flag = config["STAGE"]["BCFM"]

        self.fusion = SimpleFusion()
        
        self.ccfm_dim = config["CCFM"]["DIM"]    
        self.bcfm_dim = config["BCFM"]["DIM"]
        
        self.drug_extractor = Encoder_drug(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)        
        self.protein_extractor = Encoder_protein(protein_in_dim, protein_hidden_dim, protein_target_dim)
        self.cross_attn = CrossTransformerLayer(dim_model=protein_target_dim, n_head=4)
        
        if self.bcfm_flag:
            self.bcfm_list = nn.ModuleList([BCFM(dim_model=self.bcfm_dim) for i in range(self.stage_num)])
    
        if self.ccfm_flag:
            self.fusion = CCFM(dim_model=self.ccfm_dim)
        else:
            self.fusion = SimpleFusion()
        self.env_mlp = EnvironmentMLP(protein_target_dim, mlp_out_dim)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        # 对比学习投影头 - 减少中间层维度以节省内存
        self.f_proj = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_hidden_dim // 2),  # 减少中间层维度
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(mlp_hidden_dim // 2, mlp_out_dim)
        )
        self.temperature = config["CONTRASTIVE_TEMPERATURE"] if "CONTRASTIVE_TEMPERATURE" in config else 0.1

    def forward(self, v_d, v_p, v_d_aug, v_p_aug, v_d_mask, v_p_mask):
        # 使用torch.cuda.amp.autocast()进行混合精度训练以节省内存
        
        v_d = self.drug_extractor(v_d) #F_m
        v_p = self.protein_extractor(v_p)#F_p

        # 对augmented data进行处理
        v_d_aug = self.drug_extractor(v_d_aug)
        v_p_aug = self.protein_extractor(v_p_aug)
        
        if self.bcfm_flag:
            for i in range(self.stage_num):
                v_p, v_d = self.bcfm_list[i](v_p, v_d, v_p_mask, v_d_mask)
                # v_p_aug, v_d_aug = self.bcfm_list[i](v_p_aug, v_d_aug, v_p_mask, v_d_mask)
        if self.ccfm_flag:
            f = self.fusion(v_p, v_d, v_p_mask, v_d_mask)
        #     # f_aug = self.fusion(v_p_aug, v_d_aug, v_p_mask, v_d_mask)
        else:
            f = self.fusion(v_d, v_p, v_d_mask, v_p_mask)
            # f_aug = self.fusion(v_d_aug, v_p_aug, v_d_mask, v_p_mask)
        # p_env = self.env_mlp(f)
        # f = self.fusion(v_p, v_d, v_p_mask, v_d_mask)
        f_aug = self.fusion(v_d_aug, v_p_aug, v_d_mask, v_p_mask)
        score = self.mlp_classifier(f) 
        # 对比学习投影
        z = self.f_proj(f)
        z = F.normalize(z, p=2, dim=1)  # L2归一化
        z_aug = self.f_proj(f_aug)
        z_aug = F.normalize(z_aug, p=2, dim=1)  # L2归一化
        
        return v_d, v_p, f, score, z, z_aug

    @staticmethod
    def contrastive_loss(z, labels, temperature=0.1):
        """
        z: (batch, dim) 已L2归一化
        labels: (batch,) int, 环境类别标签
        temperature: float
        """
        device = z.device
        batch_size = z.size(0)
        sim_matrix = torch.matmul(z, z.T) / temperature  # (batch, batch)
        labels = labels.contiguous().view(-1, 1)  # (batch, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (batch, batch)
        # 排除自身
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        # 计算分母（所有非自身的样本）
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        # 只对正样本求平均
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss
    @staticmethod
    def simclr_nt_xent_loss(z, temperature=0.1):
        """
        z: (2N, dim) 已L2归一化，前N为原始，后N为增广
        """
        batch_size = z.size(0) // 2
        z = torch.cat([z[:batch_size], z[batch_size:]], dim=0)  # shape: (2N, dim)
        sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels, labels], dim=0)
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        logits_mask = torch.ones_like(mask) - torch.eye(2*batch_size, device=z.device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

class EnvironmentMLP(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(EnvironmentMLP, self).__init__()
        self.env_fc = nn.Linear(hidden_dim, out_dim)#enviroment_classfier
    
    def forward(self, pred_output):
        return F.softmax(self.env_fc(pred_output), dim=1)

class SimpleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = MaskedAveragePooling()

    def forward(self, v_d, v_p, v_d_mask, v_p_mask):
        tensor1_pooled = self.avgpool(v_d, v_d_mask)
        tensor2_pooled = self.avgpool(v_p, v_p_mask)

        concatenated = tensor1_pooled+ tensor2_pooled

        return concatenated