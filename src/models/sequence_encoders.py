import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import esm
import re

class PeptideEncoder(nn.Module):
    def __init__(self, model_type='esm', hidden_dim=512):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'esm':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.tokenizer = self.alphabet.get_batch_converter()
        elif model_type == 'protbert':
            self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
            self.model = AutoModel.from_pretrained("Rostlab/prot_bert")
        
        self.projection = nn.Linear(1280 if model_type == 'esm' else 1024, hidden_dim)
        
    def forward(self, sequences):
        if self.model_type == 'esm':
            return self._encode_esm(sequences)
        else:
            return self._encode_protbert(sequences)
    
    def _encode_esm(self, sequences):
        batch_labels, batch_strs, batch_tokens = self.tokenizer(sequences)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33]
        
        # 平均池化获取序列表示
        sequence_embeddings = embeddings.mean(dim=1)
        return self.projection(sequence_embeddings)
    
    def _encode_protbert(self, sequences):
        inputs = self.tokenizer(sequences, return_tensors='pt', padding=True, 
                               truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        sequence_embeddings = outputs.last_hidden_state.mean(dim=1)
        return self.projection(sequence_embeddings)

class MultiModalDrugEncoder(nn.Module):
    """多模态药物编码器：处理小分子和肽药物"""
    def __init__(self, drug_dim=2048, peptide_dim=512, hidden_dim=256):
        super().__init__()
        self.small_molecule_encoder = nn.Sequential(
            nn.Linear(drug_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        
        self.peptide_encoder = PeptideEncoder(hidden_dim=hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, drug_features, peptide_sequences=None):
        if peptide_sequences is not None:
            # 肽药物编码
            peptide_emb = self.peptide_encoder(peptide_sequences)
            molecule_emb = self.small_molecule_encoder(drug_features)
            fused = torch.cat([peptide_emb, molecule_emb], dim=1)
            return self.fusion(fused)
        else:
            # 小分子药物编码
            return self.small_molecule_encoder(drug_features)
