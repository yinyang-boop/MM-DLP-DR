#!/usr/bin/env python3
"""
模型训练脚本：训练和评估药物重定位模型
"""

import argparse
import torch
from src.models.transformer_dta import TransformerDTAWithGNN
from src.training.trainer import DrugRepurposingTrainer
from src.data.data_loader import load_training_data

def main(config_path):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    train_loader, val_loader, test_loader = load_training_data(config)
    
    # 初始化模型
    model = TransformerDTAWithGNN(
        drug_dim=config['model']['drug_dim'],
        protein_dim=config['model']['protein_dim'],
        hidden_dim=config['model']['hidden_dim']
    )
    
    # 初始化训练器
    trainer = DrugRepurposingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # 训练模型
    trainer.train()
    
    # 保存模型
    torch.save(model.state_dict(), './models/trained_model.pth')
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
