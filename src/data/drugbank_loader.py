import xml.etree.ElementTree as ET
import pandas as pd
import requests
import os

class DrugBankLoader:
    def __init__(self, username, password, data_dir="./data/raw"):
        self.username = username
        self.password = password
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_drugbank(self):
        """下载DrugBank XML数据"""
        url = "https://go.drugbank.com/releases/5-1-13/downloads/all-full-database"
        local_path = os.path.join(self.data_dir, "drugbank.xml")
        
        # 使用您的认证信息下载
        response = requests.get(url, auth=(self.username, self.password), stream=True)
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    
    def parse_drugbank_xml(self, xml_file):
        """解析DrugBank XML文件"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        drugs_data = []
        targets_data = []
        interactions_data = []
        
        for drug in root.findall('drug'):
            # 解析药物基本信息
            drug_id = drug.find('drugbank-id').text
            name = drug.find('name').text
            
            # 解析靶点信息
            for target in drug.findall('.//targets/target'):
                target_id = target.find('id').text
                target_name = target.find('name').text
                uniprot_id = target.find('polypeptide/accession').text if target.find('polypeptide/accession') is not None else None
                
                targets_data.append({
                    'drugbank_id': drug_id,
                    'target_id': target_id,
                    'target_name': target_name,
                    'uniprot_id': uniprot_id
                })
            
            # 解析ATC代码
            atc_codes = []
            for atc in drug.findall('.//atc-codes/atc-code'):
                atc_codes.append(atc.get('code'))
            
            drugs_data.append({
                'drugbank_id': drug_id,
                'name': name,
                'atc_codes': atc_codes,
                'type': drug.get('type'),
                'groups': [group.text for group in drug.findall('groups/group')]
            })
        
        return pd.DataFrame(drugs_data), pd.DataFrame(targets_data)
