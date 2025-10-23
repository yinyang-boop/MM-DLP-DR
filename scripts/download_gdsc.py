# scripts/download_gdsc.py

import requests
import pandas as pd
from io import StringIO
import os

def download_gdsc(dataset="GDSC1", out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    url = f"https://www.cancerrxgene.org/gdsc1000/{dataset}_fitted_dose_response.csv"
    print("Downloading:", url)
    r = requests.get(url, timeout=60)
    df = pd.read_csv(StringIO(r.text))
    out_path = os.path.join(out_dir, f"{dataset.lower()}_response.csv")
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    return df

if __name__ == "__main__":
    download_gdsc()
