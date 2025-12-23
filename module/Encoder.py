import torch.nn as nn
import torch
from dgllife.model.gnn import GCN
from .SMILES_Transformer import SMILESEncoder


class Encoder_drug(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(Encoder_drug, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]
    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class Encoder_drug_smiles(nn.Module):
    """基于SMILES Transformer的药物编码器"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=512, max_len=500, dropout=0.1, 
                 output_dim=128, pooling='mean'):
        super(Encoder_drug_smiles, self).__init__()
        
        self.smiles_encoder = SMILESEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout,
            output_dim=output_dim,
            pooling=pooling
        )
        
        self.output_dim = output_dim
    
    def forward(self, smiles_tokens, padding_mask=None):
        """
        前向传播
        Args:
            smiles_tokens: (batch_size, seq_len) token索引
            padding_mask: (batch_size, seq_len) padding mask
        Returns:
            features: (batch_size, output_dim) 提取的特征
        """
        return self.smiles_encoder(smiles_tokens, padding_mask)


class HybridEncoder_drug(nn.Module):
    """混合药物编码器，结合图神经网络和SMILES Transformer"""
    
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, 
                 activation=None, vocab_size=None, use_smiles_transformer=True,
                 smiles_d_model=256, smiles_nhead=8, smiles_num_layers=6,
                 smiles_dim_feedforward=512, smiles_max_len=500, smiles_dropout=0.1,
                 smiles_output_dim=128, smiles_pooling='mean', fusion_method='concat'):
        super(HybridEncoder_drug, self).__init__()
        
        # 图神经网络编码器
        self.gnn_encoder = Encoder_drug(in_feats, dim_embedding, padding, hidden_feats, activation)
        
        self.use_smiles_transformer = use_smiles_transformer
        self.fusion_method = fusion_method
        
        if use_smiles_transformer and vocab_size is not None:
            # SMILES Transformer编码器
            self.smiles_encoder = Encoder_drug_smiles(
                vocab_size=vocab_size,
                d_model=smiles_d_model,
                nhead=smiles_nhead,
                num_layers=smiles_num_layers,
                dim_feedforward=smiles_dim_feedforward,
                max_len=smiles_max_len,
                dropout=smiles_dropout,
                output_dim=smiles_output_dim,
                pooling=smiles_pooling
            )
            
            # 特征融合层
            gnn_output_dim = hidden_feats[-1] if hidden_feats else dim_embedding
            if fusion_method == 'concat':
                fusion_input_dim = gnn_output_dim + smiles_output_dim
            elif fusion_method == 'add':
                fusion_input_dim = gnn_output_dim
                assert gnn_output_dim == smiles_output_dim, "Add fusion requires same dimensions"
            elif fusion_method == 'attention':
                fusion_input_dim = gnn_output_dim
                self.attention_fusion = nn.MultiheadAttention(
                    embed_dim=gnn_output_dim, num_heads=4, batch_first=True
                )
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_input_dim, gnn_output_dim)
            )
            
            self.output_feats = gnn_output_dim
        else:
            self.output_feats = hidden_feats[-1] if hidden_feats else dim_embedding
    
    def forward(self, batch_graph=None, smiles_tokens=None, smiles_mask=None):
        """
        前向传播
        Args:
            batch_graph: 图神经网络输入
            smiles_tokens: SMILES token索引
            smiles_mask: SMILES padding mask
        """
        if not self.use_smiles_transformer or smiles_tokens is None:
            # 只使用图神经网络
            return self.gnn_encoder(batch_graph)
        
        # 获取图神经网络特征
        gnn_features = self.gnn_encoder(batch_graph)  # (batch_size, max_nodes, gnn_dim)
        
        # 获取SMILES Transformer特征
        smiles_features = self.smiles_encoder(smiles_tokens, smiles_mask)  # (batch_size, smiles_dim)
        
        # 特征融合
        if self.fusion_method == 'concat':
            # 对图神经网络特征进行全局平均池化
            gnn_pooled = torch.mean(gnn_features, dim=1)  # (batch_size, gnn_dim)
            # 拼接特征
            fused_features = torch.cat([gnn_pooled, smiles_features], dim=1)
            # 通过融合层
            output = self.fusion_layer(fused_features)
            # 扩展回原来的形状
            output = output.unsqueeze(1).expand(-1, gnn_features.size(1), -1)
            
        elif self.fusion_method == 'add':
            # 对图神经网络特征进行全局平均池化
            gnn_pooled = torch.mean(gnn_features, dim=1)
            # 相加
            fused_features = gnn_pooled + smiles_features
            # 扩展回原来的形状
            output = fused_features.unsqueeze(1).expand(-1, gnn_features.size(1), -1)
            
        elif self.fusion_method == 'attention':
            # 使用注意力机制融合
            smiles_features_expanded = smiles_features.unsqueeze(1)  # (batch_size, 1, smiles_dim)
            attended_features, _ = self.attention_fusion(
                gnn_features, smiles_features_expanded, smiles_features_expanded
            )
            output = attended_features
        
        return output


"""
ESM Feature For Protein 
"""
class Encoder_protein(nn.Module):
    def __init__(self, embedding_dim=320, hidden_dim=320, target_dim=128):
        super(Encoder_protein, self).__init__()
        self.output_layer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        x = self.output_layer(x)
        return x