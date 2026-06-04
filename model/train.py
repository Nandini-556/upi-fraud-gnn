"""
Training script — GraphSAGE + GAT on UPI fraud graph.
Full-graph training (no NeighborLoader) — works without torch-sparse.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from model.gnn import FraudPredictor


CFG = {
    "hidden_dim"  : 64,
    "lr"          : 1e-3,
    "weight_decay": 1e-4,
    "epochs"      : 100,
    "patience"    : 10,
    "pos_weight"  : 29.0,
    "threshold"   : 0.50,
    "model_dir"   : "model/checkpoints",
    "graph_path"  : "data/processed/graph.pt",
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",
}


def load_graph():
    path = Path(CFG["graph_path"])
    if not path.exists():
        raise FileNotFoundError("Graph not found. Run: python data\\graph_builder.py")
    data = torch.load(path, weights_only=False)
    print(f"Graph loaded | device: {CFG['device']}")
    return data


def train_epoch(model, data, optimiser, pos_weight, device):
    model.train()
    optimiser.zero_grad()

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    out    = model(data.x_dict, data.edge_index_dict)
    logits = out["txn"][data["txn"].train_mask]
    labels = data["txn"].y[data["txn"].train_mask].float()

    loss = criterion(logits[:, 1], labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device, threshold=0.50):
    model.eval()

    out    = model(data.x_dict, data.edge_index_dict)
    logits = out["txn"][mask]
    probs  = logits.softmax(dim=-1)[:, 1].cpu().numpy()
    labels = data["txn"].y[mask].cpu().numpy()

    auc   = roc_auc_score(labels, probs)
    ap    = average_precision_score(labels, probs)
    preds = (probs >= threshold).astype(int)

    return {"auc": auc, "ap": ap, "preds": preds, "probs": probs, "labels": labels}


def train():
    Path(CFG["model_dir"]).mkdir(parents=True, exist_ok=True)
    device = torch.device(CFG["device"])
    print(f"Device: {device}")

    data = load_graph().to(device)

    model = FraudPredictor(
        metadata=data.metadata(),
        hidden=CFG["hidden_dim"],
    ).to(device)

    # Lazy init — one dummy forward
    _ = model(data.x_dict, data.edge_index_dict)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", patience=5, factor=0.5
    )

    best_auc    = 0.0
    patience_ct = 0
    history     = []

    print("\nStarting training...")
    for epoch in range(1, CFG["epochs"] + 1):
        t0   = time.time()
        loss = train_epoch(model, data, optimiser, CFG["pos_weight"], device)
        metrics = evaluate(model, data, data["txn"].val_mask, device, CFG["threshold"])
        scheduler.step(metrics["auc"])

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "loss" : round(loss, 4),
            "auc"  : round(metrics["auc"], 4),
            "ap"   : round(metrics["ap"],  4),
        })

        flag = ""
        if metrics["auc"] > best_auc:
            best_auc    = metrics["auc"]
            patience_ct = 0
            model.save(f"{CFG['model_dir']}/best_model.pt")
            flag = " ✓ best"
        else:
            patience_ct += 1

        print(
            f"Epoch {epoch:03d} | loss {loss:.4f} | "
            f"AUC {metrics['auc']:.4f} | AP {metrics['ap']:.4f} | "
            f"{elapsed:.1f}s{flag}"
        )

        if patience_ct >= CFG["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (patience={CFG['patience']})")
            break

    with open(f"{CFG['model_dir']}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation
    print("\n── Final evaluation ──")
    model.load(f"{CFG['model_dir']}/best_model.pt", device=str(device))
    test_metrics = evaluate(model, data, data["txn"].test_mask, device, CFG["threshold"])
    print(f"Test AUC : {test_metrics['auc']:.4f}")
    print(f"Test AP  : {test_metrics['ap']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        test_metrics["labels"],
        test_metrics["preds"],
        target_names=["Normal", "Fraud"],
    ))
    print(f"\nBest model checkpoint → {CFG['model_dir']}/best_model.pt")


if __name__ == "__main__":
    train()