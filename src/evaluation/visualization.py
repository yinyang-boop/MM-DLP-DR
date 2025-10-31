import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_training_history(train_losses, val_losses):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/training_history.png')
    plt.close()

def plot_predictions_vs_actuals(predictions, actuals):
    """绘制预测值 vs 实际值"""
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actuals')
    plt.savefig('./results/predictions_vs_actuals.png')
    plt.close()

def plot_feature_importance(feature_names, importance_scores):
    """绘制特征重要性"""
    indices = np.argsort(importance_scores)[::-1]
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importance_scores)), importance_scores[indices])
    plt.xticks(range(len(importance_scores)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('./results/feature_importance.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('./results/confusion_matrix.png')
    plt.close()
