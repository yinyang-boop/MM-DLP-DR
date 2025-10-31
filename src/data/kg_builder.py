import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_mapping = {}
        self.edge_types = {}
    
    def add_drugs(self, drugs_df: pd.DataFrame):
        """添加药物节点"""
        for _, drug in drugs_df.iterrows():
            node_id = f"drug_{drug['drugbank_id']}"
            self.node_mapping[node_id] = len(self.node_mapping)
            self.graph.add_node(node_id, 
                              type='drug',
                              name=drug['name'],
                              atc_codes=drug['atc_codes'])
    
    def add_proteins(self, uniprot_df: pd.DataFrame):
        """添加蛋白质节点"""
        for _, protein in uniprot_df.iterrows():
            node_id = f"protein_{protein['uniprot_id']}"
            self.node_mapping[node_id] = len(self.node_mapping)
            self.graph.add_node(node_id,
                              type='protein',
                              name=protein['name'],
                              sequence=protein['sequence'])
    
    def add_diseases(self, disgenet_df: pd.DataFrame):
        """添加疾病节点"""
        for _, disease in disgenet_df.iterrows():
            node_id = f"disease_{disease['disease_id']}"
            self.node_mapping[node_id] = len(self.node_mapping)
            self.graph.add_node(node_id,
                              type='disease',
                              name=disease['name'])
    
    def add_interactions(self, interactions_df: pd.DataFrame):
        """添加各种关系边"""
        for _, interaction in interactions_df.iterrows():
            source_id = f"{interaction['source_type']}_{interaction['source_id']}"
            target_id = f"{interaction['target_type']}_{interaction['target_id']}"
            
            if source_id in self.node_mapping and target_id in self.node_mapping:
                self.graph.add_edge(source_id, target_id, 
                                  relation_type=interaction['relation_type'])
    
    def to_pytorch_geometric(self) -> Data:
        """转换为PyTorch Geometric格式"""
        # 节点特征矩阵
        node_features = []
        for node_id, attr in self.graph.nodes(data=True):
            features = self._get_node_features(attr)
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # 边索引和边属性
        edge_index = []
        edge_attr = []
        
        for i, (src, dst, attr) in enumerate(self.graph.edges(data=True)):
            edge_index.append([self.node_mapping[src], self.node_mapping[dst]])
            edge_attr.append(self._get_edge_attributes(attr))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def _get_node_features(self, attributes: Dict) -> List[float]:
        """根据节点类型生成特征向量"""
        # 实现特征提取逻辑
        if attributes['type'] == 'drug':
            return self._encode_drug_features(attributes)
        elif attributes['type'] == 'protein':
            return self._encode_protein_features(attributes)
        elif attributes['type'] == 'disease':
            return self._encode_disease_features(attributes)
        return [0.0] * 128  # 默认特征维度
    
    def _encode_drug_features(self, drug_attr: Dict) -> List[float]:
        """编码药物特征"""
        # 使用分子指纹或学习到的嵌入
        return [0.0] * 256
    
    def _encode_protein_features(self, protein_attr: Dict) -> List[float]:
        """编码蛋白质特征"""
        # 使用ESM或ProtBERT嵌入
        return [0.0] * 512
