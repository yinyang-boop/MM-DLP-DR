import optuna
import torch
import torch.nn as nn
from functools import partial

def objective(trial, model_class, train_loader, val_loader):
    """Optuna优化目标函数"""
    # 建议超参数
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    
    # 初始化模型
    model = model_class(
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        num_layers=num_layers
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(50):  # 缩短训练轮数用于调优
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch['affinity'])
            loss.backward()
            optimizer.step()
    
    # 验证损失
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            val_loss += criterion(outputs, batch['affinity']).item()
    
    return val_loss / len(val_loader)

def optimize_hyperparameters(model_class, train_loader, val_loader, n_trials=100):
    """执行超参数优化"""
    study = optuna.create_study(direction='minimize')
    objective_func = partial(objective, model_class=model_class, 
                           train_loader=train_loader, val_loader=val_loader)
    
    study.optimize(objective_func, n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params
