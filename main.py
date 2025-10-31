import argparse
import yaml
import torch
from src.data.drugbank_loader import DrugBankLoader
from src.data.kg_builder import KnowledgeGraphBuilder
from src.models.transformer_dta import TransformerDTAWithGNN
from src.training.trainer import DrugRepurposingTrainer
from src.evaluation.metrics import evaluate_model

def main(config_path):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化数据加载器
    drugbank_loader = DrugBankLoader(
        username="your_username",  # 替换为实际用户名
        password="your_password",  # 替换为实际密码
        data_dir=config['data']['output_dir']
    )
    
    # 下载和解析DrugBank数据
    print("Downloading DrugBank data...")
    xml_path = drugbank_loader.download_drugbank()
    drugs_df, targets_df = drugbank_loader.parse_drugbank_xml(xml_path)
    
    # 构建知识图谱
    print("Building knowledge graph...")
    kg_builder = KnowledgeGraphBuilder()
    kg_builder.add_drugs(drugs_df)
    # 添加其他实体和关系...
    kg_data = kg_builder.to_pytorch_geometric()
    
    # 初始化模型
    model = TransformerDTAWithGNN(
        drug_dim=config['model']['drug_dim'],
        protein_dim=config['model']['protein_dim'],
        hidden_dim=config['model']['hidden_dim'],
        gnn_layers=config['model']['gnn_layers'],
        heads=config['model']['attention_heads']
    )
    
    # 训练模型
    trainer = DrugRepurposingTrainer(
        model=model,
        train_loader=train_loader,  # 需要实现数据加载
        val_loader=val_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    )
    
    # 训练循环
    for epoch in range(config['training']['epochs']):
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # 评估模型
    evaluation_results = evaluate_model(model, test_loader)
    print("Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
