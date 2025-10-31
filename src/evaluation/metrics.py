import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

def evaluate_model(model, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """全面评估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 移动数据到设备
            drug_data = batch['drug'].to(device)
            protein_data = batch['protein'].to(device)
            targets = batch['affinity'].to(device)
            
            # 预测
            predictions = model(drug_data, protein_data)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def calculate_binding_affinity_metrics(predictions, targets):
    """计算结合亲和力特定指标"""
    # 转换为numpy数组
    pred_np = np.array(predictions)
    target_np = np.array(targets)
    
    # 计算误差指标
    metrics = {
        'mse': mean_squared_error(target_np, pred_np),
        'rmse': np.sqrt(mean_squared_error(target_np, pred_np)),
        'mae': mean_absolute_error(target_np, pred_np),
        'r2': r2_score(target_np, pred_np)
    }
    
    # 计算分类指标（如果设置了阈值）
    if len(np.unique(target_np)) == 2:  # 二分类
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics.update({
            'accuracy': accuracy_score(target_np, pred_np > 0.5),
            'precision': precision_score(target_np, pred_np > 0.5),
            'recall': recall_score(target_np, pred_np > 0.5),
            'f1': f1_score(target_np, pred_np > 0.5)
        })
    
    return metrics
