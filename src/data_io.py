import pandas as pd
def load_split(dataset, split):
    return pd.read_csv(f"data/processed/{dataset}_{split}_stratified.csv")
