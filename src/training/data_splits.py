import numpy as np
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader

class DrugTargetDataset(Dataset):
    def __init__(self, drug_features, protein_features, affinities):
        self.drug_features = drug_features
        self.protein_features = protein_features
        self.affinities = affinities
    
    def __len__(self):
        return len(self.affinities)
    
    def __getitem__(self, idx):
        return {
            'drug': self.drug_features[idx],
            'protein': self.protein_features[idx],
            'affinity': self.affinities[idx]
        }

def create_data_splits(drug_data, protein_data, affinity_data, config):
    """创建训练、验证、测试分割"""
    # 初始分割：训练+验证 vs 测试
    train_val_drug, test_drug, train_val_protein, test_protein, train_val_affinity, test_affinity = train_test_split(
        drug_data, protein_data, affinity_data,
        test_size=config['training']['test_split'],
        random_state=42
    )
    
    # 二次分割：训练 vs 验证
    train_drug, val_drug, train_protein, val_protein, train_affinity, val_affinity = train_test_split(
        train_val_drug, train_val_protein, train_val_affinity,
        test_size=config['training']['validation_split'],
        random_state=42
    )
    
    # 创建数据集
    train_dataset = DrugTargetDataset(train_drug, train_protein, train_affinity)
    val_dataset = DrugTargetDataset(val_drug, val_protein, val_affinity)
    test_dataset = DrugTargetDataset(test_drug, test_protein, test_affinity)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])
    
    return train_loader, val_loader, test_loader

def create_cv_folds(drug_data, protein_data, affinity_data, n_folds=5):
    """创建交叉验证折叠"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []
    
    for train_index, test_index in kf.split(drug_data):
        train_drug, test_drug = drug_data[train_index], drug_data[test_index]
        train_protein, test_protein = protein_data[train_index], protein_data[test_index]
        train_affinity, test_affinity = affinity_data[train_index], affinity_data[test_index]
        
        train_dataset = DrugTargetDataset(train_drug, train_protein, train_affinity)
        test_dataset = DrugTargetDataset(test_drug, test_protein, test_affinity)
        
        folds.append((train_dataset, test_dataset))
    
    return folds
