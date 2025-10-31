import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoModel

class TransformerDTAWithGNN(nn.Module):
    """结合Transformer和GNN的药物-靶点亲和力预测模型"""
    
    def __init__(self, drug_dim=2048, protein_dim=1024, hidden_dim=256, 
                 gnn_layers=3, heads=8):
        super().__init__()
        
        # 药物编码器
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim)
        )
        
        # 蛋白质编码器（使用ESM）
        self.protein_encoder = PeptideEncoder(model_type='esm', hidden_dim=hidden_dim)
        
        # GNN层用于知识图谱信息
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // heads, heads=heads) 
            for _ in range(gnn_layers)
        ])
        
        # 注意力融合机制
        self.drug_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.protein_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, drug_data, protein_data, kg_data=None):
        # 编码药物和蛋白质
        drug_emb = self.drug_encoder(drug_data['features'])
        protein_emb = self.protein_encoder(protein_data['sequences'])
        
        # 如果提供知识图谱数据，应用GNN
        if kg_data is not None:
            drug_emb = self._apply_gnn(drug_emb, kg_data)
            protein_emb = self._apply_gnn(protein_emb, kg_data)
        
        # 注意力机制
        drug_attn, _ = self.drug_attention(drug_emb.unsqueeze(0), 
                                         drug_emb.unsqueeze(0), 
                                         drug_emb.unsqueeze(0))
        protein_attn, _ = self.protein_attention(protein_emb.unsqueeze(0),
                                               protein_emb.unsqueeze(0),
                                               protein_emb.unsqueeze(0))
        
        # 特征融合
        combined = torch.cat([
            drug_emb.mean(dim=0, keepdim=True),
            protein_emb.mean(dim=0, keepdim=True),
            drug_attn.mean(dim=0, keepdim=True),
            protein_attn.mean(dim=0, keepdim=True)
        ], dim=-1)
        
        return self.predictor(combined).squeeze()
    
    def _apply_gnn(self, embeddings, kg_data):
        """应用GNN处理知识图谱"""
        x, edge_index = kg_data.x, kg_data.edge_index
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        
        return x
