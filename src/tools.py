# src/tools.py

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def infer_chembl(model, smiles_list, featurizer):
    X = featurizer(smiles_list)
    return model.predict_proba(X)[:,1]

def infer_drugbank(model, mechanisms_list, vectorizer):
    X = vectorizer.transform(mechanisms_list)
    return model.predict_proba(X)[:,1]

def infer_gdsc(model, df, feat_cols):
    return model.predict_proba(df[feat_cols])[:,1]

def evaluate_binary(y_true, y_pred, name, save_path=None):
    m = {"roc_auc": roc_auc_score(y_true, y_pred),
         "pr_auc": average_precision_score(y_true, y_pred)}
    if save_path:
        import json
        with open(save_path, "w") as f: json.dump(m, f, indent=2)
    print(f"[{name}] ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}")
    return m
