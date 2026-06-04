"""
Graph Builder — Convert IEEE-CIS transaction CSV to PyTorch Geometric graph.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


RAW_DIR   = Path("data/raw")
GRAPH_DIR = Path("data/processed")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def load_csv() -> pd.DataFrame:
    path = RAW_DIR / "train_transaction.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python data/download_data.py"
        )
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}  |  Fraud rate: {df['isFraud'].mean():.4f}")
    return df


def engineer_user_features(df: pd.DataFrame):
    grp = df.groupby("card1")
    user_stats = grp["TransactionAmt"].agg(["mean", "std", "count"]).fillna(0)
    user_stats["fraud_rate"] = grp["isFraud"].mean()
    user_stats["degree"]     = grp["TransactionID"].count()

    feat = user_stats.values.astype(np.float32)
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)

    user_idx = {card: i for i, card in enumerate(user_stats.index)}
    return user_idx, feat


def engineer_txn_features(df: pd.DataFrame) -> np.ndarray:
    cols = ["TransactionAmt", "dist1", "dist2", "C1", "C2", "C6", "V1", "V2", "V3"]

    if "TransactionDT" in df.columns:
        df = df.copy()
        df["hour"] = (df["TransactionDT"] // 3600) % 24
        cols = ["hour"] + cols

    feat = df[cols].fillna(0).values.astype(np.float32)
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
    return feat


def build_graph(df: pd.DataFrame):
    print("Building heterogeneous graph...")

    user_idx, user_feat = engineer_user_features(df)
    txn_feat            = engineer_txn_features(df)
    labels              = df["isFraud"].values.astype(np.int64)

    data = HeteroData()

    data["user"].x     = torch.tensor(user_feat, dtype=torch.float)
    data["txn"].x      = torch.tensor(txn_feat,  dtype=torch.float)
    data["txn"].y      = torch.tensor(labels,    dtype=torch.long)
    data["txn"].txn_id = torch.tensor(df["TransactionID"].values, dtype=torch.long)

    src_user = torch.tensor([user_idx[c] for c in df["card1"]], dtype=torch.long)
    dst_txn  = torch.arange(len(df), dtype=torch.long)

    data["user", "sends", "txn"].edge_index     = torch.stack([src_user, dst_txn])
    data["txn", "rev_sends", "user"].edge_index = torch.stack([dst_txn, src_user])

    n         = len(df)
    perm      = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end   = int(0.8 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:train_end]]      = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]]         = True

    data["txn"].train_mask = train_mask
    data["txn"].val_mask   = val_mask
    data["txn"].test_mask  = test_mask

    print(f"  User nodes : {data['user'].num_nodes:,}")
    print(f"  Txn  nodes : {data['txn'].num_nodes:,}")
    print(f"  Edges      : {data['user','sends','txn'].edge_index.shape[1]:,}")
    print(f"  Train / Val / Test : {train_mask.sum()} / {val_mask.sum()} / {test_mask.sum()}")

    return data, user_idx


def save_artifacts(data: HeteroData, user_idx: dict, df: pd.DataFrame):
    torch.save(data, GRAPH_DIR / "graph.pt")
    with open(GRAPH_DIR / "user_idx.pkl", "wb") as f:
        pickle.dump(user_idx, f)
    # CSV fallback — parquet nahi chahiye
    df.to_csv(GRAPH_DIR / "transactions.csv", index=False)
    print(f"Saved graph → {GRAPH_DIR}/graph.pt")


if __name__ == "__main__":
    df = load_csv()
    data, user_idx = build_graph(df)
    save_artifacts(data, user_idx, df)
