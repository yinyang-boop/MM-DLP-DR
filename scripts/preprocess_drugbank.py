# scripts/preprocess_drugbank.py

import pandas as pd
import os

def preprocess_drugbank(in_path="data/raw/drugbank.csv", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_path)

    # Drop duplicates
    if "smiles" in df.columns:
        df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    # Proxy label: inhibitor vs non-inhibitor
    if "mechanisms" in df.columns:
        df["label"] = df["mechanisms"].fillna("").str.lower().apply(
            lambda x: 1 if "inhibitor" in x else 0
        )

    out_path = os.path.join(out_dir, "drugbank_processed.csv")
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    return df

if __name__ == "__main__":
    preprocess_drugbank()
