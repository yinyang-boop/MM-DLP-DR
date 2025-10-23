from chembl_webresource_client.new_client import new_client
import pandas as pd

def fetch_chembl(target_id="CHEMBL203"):
    df = pd.DataFrame.from_records(list(new_client.activity.filter(
        target_chembl_id=target_id, assay_type="B"
    )))
    df.to_csv(f"data/raw/{target_id}.csv", index=False)
    print("Saved:", f"data/raw/{target_id}.csv")

if __name__ == "__main__":
    fetch_chembl()
