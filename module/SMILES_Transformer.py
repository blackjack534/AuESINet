import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from collections import Counter
import numpy as np


class SMILESVocabulary:
    """SMILES词汇表，用于tokenization"""
    
    def __init__(self, smiles_list=None, max_vocab_size=1000, min_freq=1):
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        if smiles_list is not None:
            self.build_vocab(smiles_list)
        else:
            # 使用预定义的常见SMILES tokens
            self._build_default_vocab()
    
    def _build_default_vocab(self):
        """构建默认的SMILES词汇表"""
        # 常见的原子符号
        atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'H']
        
        # 数字
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # 化学键符号
        bonds = ['-', '=', '#', ':', '\\', '/']
        
        # 环闭合符号
        rings = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '%10', '%11', '%12']
        
        # 分支符号
        branches = ['(', ')']
        
        # 芳香性符号
        aromatic = ['c', 'n', 'o', 's', 'p']
        
        # 立体化学符号
        stereo = ['@', '@@', '\\', '/']
        
        # 电荷符号
        charge = ['+', '-']
        
        # 组合所有tokens
        all_tokens = (atoms + numbers + bonds + rings + branches + 
                     aromatic + stereo + charge + self.special_tokens)
        
        self.token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
    
    def build_vocab(self, smiles_list):
        """从SMILES列表构建词汇表"""
        token_counter = Counter()
        
        for smiles in smiles_list:
            tokens = self._tokenize_smiles(smiles)
            token_counter.update(tokens)
        
        # 保留频率大于min_freq的tokens
        vocab_tokens = [token for token, count in token_counter.most_common() 
                       if count >= self.min_freq]
        
        # 添加特殊tokens
        vocab_tokens = self.special_tokens + vocab_tokens
        
        # 限制词汇表大小
        if len(vocab_tokens) > self.max_vocab_size:
            vocab_tokens = vocab_tokens[:self.max_vocab_size]
        
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(vocab_tokens)}
        self.vocab_size = len(vocab_tokens)
    
    def _tokenize_smiles(self, smiles):
        """将SMILES字符串分解为tokens"""
        tokens = []
        i = 0
        while i < len(smiles):
            # 处理百分号开头的环编号 (如%10, %11)
            if smiles[i] == '%' and i + 2 < len(smiles):
                tokens.append(smiles[i:i+3])
                i += 3
            # 处理方括号中的原子
            elif smiles[i] == '[':
                bracket_end = smiles.find(']', i)
                if bracket_end != -1:
                    tokens.append(smiles[i:bracket_end+1])
                    i = bracket_end + 1
                else:
                    tokens.append(smiles[i])
                    i += 1
            # 处理其他字符
            else:
                tokens.append(smiles[i])
                i += 1
        return tokens
    
    def encode(self, smiles, max_length=None):
        """将SMILES字符串编码为token索引"""
        tokens = self._tokenize_smiles(smiles)
        
        # 添加BOS和EOS tokens
        tokens = ['<BOS>'] + tokens + ['<EOS>']
        
        # 转换为索引
        indices = [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) 
                  for token in tokens]
        
        # 截断或填充到指定长度
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.token_to_idx['<PAD>']] * (max_length - len(indices)))
        
        return indices
    
    def decode(self, indices):
        """将token索引解码为SMILES字符串"""
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in indices]
        # 移除特殊tokens
        tokens = [token for token in tokens if token not in ['<PAD>', '<BOS>', '<EOS>']]
        return ''.join(tokens)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SMILESTransformerBlock(nn.Module):
    """SMILES Transformer块"""
    
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(SMILESTransformerBlock, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class SMILESTransformer(nn.Module):
    """SMILES Transformer模型"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=512, max_len=500, dropout=0.1):
        super(SMILESTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            SMILESTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, src, src_key_padding_mask=None, return_attention=False):
        """
        前向传播
        Args:
            src: (batch_size, seq_len) token索引
            src_key_padding_mask: (batch_size, seq_len) padding mask
            return_attention: 是否返回注意力权重
        """
        batch_size, seq_len = src.size()
        
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src.transpose(0, 1)).transpose(0, 1)
        src = self.dropout(src)
        
        # 生成因果掩码（可选）
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)
        
        # 通过Transformer层
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            if return_attention:
                # 如果需要返回注意力权重，这里需要修改MultiheadAttention的实现
                # 为了简化，这里只返回最终的特征
                pass
            src = transformer_block(src, src_mask=causal_mask, 
                                  src_key_padding_mask=src_key_padding_mask)
        
        src = self.layer_norm(src)
        
        if return_attention:
            return src, attention_weights
        else:
            return src


class SMILESFeatureExtractor(nn.Module):
    """SMILES特征提取器"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=512, max_len=500, dropout=0.1, 
                 output_dim=128, pooling='mean'):
        super(SMILESFeatureExtractor, self).__init__()
        
        self.transformer = SMILESTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )
        
        self.pooling = pooling
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def forward(self, smiles_tokens, padding_mask=None):
        """
        前向传播
        Args:
            smiles_tokens: (batch_size, seq_len) token索引
            padding_mask: (batch_size, seq_len) padding mask
        Returns:
            features: (batch_size, output_dim) 提取的特征
        """
        # 通过Transformer
        transformer_output = self.transformer(smiles_tokens, src_key_padding_mask=padding_mask)
        
        # 池化
        if self.pooling == 'mean':
            # 平均池化，忽略padding
            if padding_mask is not None:
                # padding_mask为True的位置是padding
                mask_expanded = padding_mask.unsqueeze(-1).expand_as(transformer_output)
                transformer_output = transformer_output.masked_fill(mask_expanded, 0)
                seq_lengths = (~padding_mask).float().sum(dim=1, keepdim=True)
                pooled_features = transformer_output.sum(dim=1) / torch.clamp(seq_lengths, min=1)
            else:
                pooled_features = transformer_output.mean(dim=1)
        elif self.pooling == 'max':
            # 最大池化
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(-1).expand_as(transformer_output)
                transformer_output = transformer_output.masked_fill(mask_expanded, float('-inf'))
            pooled_features = transformer_output.max(dim=1)[0]
        elif self.pooling == 'cls':
            # 使用第一个token作为特征
            pooled_features = transformer_output[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # 输出投影
        features = self.output_projection(pooled_features)
        
        return features


class SMILESEncoder(nn.Module):
    """SMILES编码器，用于替代原有的drug encoder"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=512, max_len=500, dropout=0.1, 
                 output_dim=128, pooling='mean'):
        super(SMILESEncoder, self).__init__()
        
        self.feature_extractor = SMILESFeatureExtractor(
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
        
        self.max_len = max_len
    
    def forward(self, smiles_tokens, padding_mask=None):
        """
        前向传播
        Args:
            smiles_tokens: (batch_size, seq_len) token索引
            padding_mask: (batch_size, seq_len) padding mask
        Returns:
            features: (batch_size, output_dim) 提取的特征
        """
        return self.feature_extractor(smiles_tokens, padding_mask)


def create_smiles_tokenizer(smiles_list=None, max_vocab_size=1000, min_freq=1):
    """创建SMILES tokenizer"""
    return SMILESVocabulary(smiles_list, max_vocab_size, min_freq)


def create_smiles_encoder(vocab_size, config=None):
    """创建SMILES编码器"""
    if config is None:
        config = {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 512,
            'max_len': 500,
            'dropout': 0.1,
            'output_dim': 128,
            'pooling': 'mean'
        }
    
    return SMILESEncoder(vocab_size=vocab_size, **config)


# 使用示例
if __name__ == "__main__":
    # 创建示例SMILES数据
    sample_smiles = [
        "CCO",  # 乙醇
        "CC(=O)O",  # 乙酸
        "C1=CC=CC=C1",  # 苯
        "CC(C)O",  # 异丙醇
        "CCOC",  # 乙醚
    ]
    
    # 创建tokenizer
    tokenizer = create_smiles_tokenizer(sample_smiles)
    
    # 编码SMILES
    encoded_smiles = []
    for smiles in sample_smiles:
        tokens = tokenizer.encode(smiles, max_length=100)
        encoded_smiles.append(tokens)
    
    # 转换为tensor
    batch_tokens = torch.tensor(encoded_smiles)
    
    # 创建编码器
    encoder = create_smiles_encoder(tokenizer.vocab_size)
    
    # 前向传播
    with torch.no_grad():
        features = encoder(batch_tokens)
        print(f"输入形状: {batch_tokens.shape}")
        print(f"输出特征形状: {features.shape}")
        print(f"词汇表大小: {tokenizer.vocab_size}")



