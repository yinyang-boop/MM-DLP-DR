import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class DrugRepurposingTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.MSELoss()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # 前向传播
            pred = self.model(batch['drug'], batch['protein'], batch.get('kg_data'))
            loss = self.criterion(pred, batch['affinity'])
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                pred = self.model(batch['drug'], batch['protein'], batch.get('kg_data'))
                loss = self.criterion(pred, batch['affinity'])
                
                total_loss += loss.item()
                predictions.extend(pred.cpu().numpy())
                targets.extend(batch['affinity'].cpu().numpy())
