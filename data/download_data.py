"""
Download IEEE-CIS Fraud Detection dataset from Kaggle.

Setup:
  1. Go to https://www.kaggle.com/account → Create API Token → downloads kaggle.json
  2. Place kaggle.json at ~/.kaggle/kaggle.json
  3. Run: python data/download_data.py
"""

import os
import zipfile
from pathlib import Path


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download():
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Install kaggle: pip install kaggle")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "kaggle.json not found.\n"
            "Go to https://www.kaggle.com/account → Create API Token\n"
            f"Place the downloaded file at: {kaggle_json}"
        )
        return

    print("Downloading IEEE-CIS Fraud Detection dataset...")
    os.system(
        f"kaggle competitions download -c ieee-fraud-detection -p {RAW_DIR}"
    )

    zip_path = RAW_DIR / "ieee-fraud-detection.zip"
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        zip_path.unlink()
        print(f"Done. Files in {RAW_DIR}:")
        for f in RAW_DIR.iterdir():
            print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
    else:
        print("Download failed — check your Kaggle credentials.")


def create_sample():
    """
    Create a small synthetic sample so you can run the project
    even without the Kaggle dataset.
    """
    import pandas as pd
    import numpy as np

    print("Creating synthetic sample dataset...")
    np.random.seed(42)
    n = 5000

    df_transaction = pd.DataFrame({
        "TransactionID":  range(2987000, 2987000 + n),
        "isFraud":        np.random.choice([0, 1], size=n, p=[0.965, 0.035]),
        "TransactionDT":  np.random.randint(86400, 86400 * 180, size=n),
        "TransactionAmt": np.abs(np.random.lognormal(mean=4.5, sigma=1.2, size=n)).round(2),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], size=n),
        "card1":          np.random.randint(1000, 18000, size=n),
        "card2":          np.random.choice([np.nan, 100.0, 200.0, 300.0, 500.0], size=n),
        "card4":          np.random.choice(["visa", "mastercard", "discover", "amex"], size=n),
        "card6":          np.random.choice(["debit", "credit"], size=n),
        "P_emaildomain":  np.random.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", np.nan], size=n
        ),
        "dist1":          np.abs(np.random.randn(n) * 100),
        "dist2":          np.abs(np.random.randn(n) * 200),
        "C1":             np.random.randint(0, 10, size=n).astype(float),
        "C2":             np.random.randint(0, 5, size=n).astype(float),
        "C6":             np.random.randint(0, 8, size=n).astype(float),
        "V1":             np.random.randn(n),
        "V2":             np.random.randn(n),
        "V3":             np.random.randn(n),
    })

    # Inject a fraud ring: 4 cards all sending to same destinations
    ring_cards = [13926, 41029, 2190, 88341]
    ring_mask = np.random.choice(range(n), size=40, replace=False)
    df_transaction.loc[ring_mask[:10], "card1"] = 13926
    df_transaction.loc[ring_mask[10:20], "card1"] = 41029
    df_transaction.loc[ring_mask[20:30], "card1"] = 2190
    df_transaction.loc[ring_mask[30:], "card1"] = 88341
    df_transaction.loc[ring_mask, "isFraud"] = 1
    df_transaction.loc[ring_mask, "TransactionAmt"] = np.random.uniform(9000, 50000, size=40)

    out = RAW_DIR / "train_transaction.csv"
    df_transaction.to_csv(out, index=False)
    print(f"Saved {len(df_transaction)} rows → {out}")
    print(f"Fraud rate: {df_transaction['isFraud'].mean():.3f}")
    return df_transaction


if __name__ == "__main__":
    csv_path = RAW_DIR / "train_transaction.csv"
    if csv_path.exists():
        print(f"Dataset already exists at {csv_path}")
    else:
        download()
        if not csv_path.exists():
            print("\nKaggle download skipped — generating synthetic sample instead.")
            create_sample()
