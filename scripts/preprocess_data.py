#!/usr/bin/env python3
"""
数据预处理脚本：下载、清洗和准备所有必要数据
"""

import argparse
import yaml
from src.data.drugbank_loader import DrugBankLoader
from src.data.bindingdb_loader import BindingDBLoader
from src.data.uniprot_loader import UniProtLoader

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理DrugBank数据
    print("Processing DrugBank data...")
    drugbank_loader = DrugBankLoader(
        config['drugbank']['username'],
        config['drugbank']['password']
    )
    drugbank_loader.process()
    
    # 处理BindingDB数据
    print("Processing BindingDB data...")
    bindingdb_loader = BindingDBLoader()
    bindingdb_loader.download_and_process()
    
    # 处理UniProt数据
    print("Processing UniProt data...")
    uniprot_loader = UniProtLoader()
    uniprot_loader.download_swissprot()
    
    print("Data preprocessing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
