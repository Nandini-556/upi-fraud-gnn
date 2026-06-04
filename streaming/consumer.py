"""
Kafka Consumer — consumes UPI transactions and runs GNN inference in real-time.

Flow:
  Kafka topic → consume message → update incremental graph
             → run GNN on subgraph → emit fraud score
             → trigger ring detector every N txns

Usage:
  python streaming/consumer.py
"""

import json
import time
import pickle
from collections import deque
from pathlib import Path
from threading import Thread, Lock

import torch
from confluent_kafka import Consumer, KafkaError

from model.gnn import FraudPredictor
from detection.ring_detector import IncrementalGraph


KAFKA_BROKER = "localhost:9092"
TOPIC_NAME   = "upi-transactions"
MODEL_PATH   = "model/checkpoints/best_model.pt"
GRAPH_PATH   = "data/processed/graph.pt"
USER_IDX_PATH = "data/processed/user_idx.pkl"

FRAUD_THRESHOLD   = 0.70
RING_DETECT_EVERY = 200   # re-run ring detection every N transactions


# ── shared state (thread-safe) ────────────────────────────────────────────────

class FraudState:
    def __init__(self):
        self._lock         = Lock()
        self.total_txns    = 0
        self.fraud_count   = 0
        self.recent_alerts = deque(maxlen=100)
        self.current_rings = []
        self.tps_window    = deque(maxlen=60)   # last 60 timestamps

    def add_alert(self, alert: dict):
        with self._lock:
            self.recent_alerts.appendleft(alert)
            self.fraud_count += 1

    def tick(self):
        with self._lock:
            self.total_txns += 1
            self.tps_window.append(time.time())

    def get_tps(self) -> float:
        with self._lock:
            if len(self.tps_window) < 2:
                return 0.0
            span = self.tps_window[-1] - self.tps_window[0]
            return len(self.tps_window) / max(span, 1)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "total_txns" : self.total_txns,
                "fraud_count": self.fraud_count,
                "tps"        : round(self.get_tps(), 1),
                "rings"      : self.current_rings,
                "alerts"     : list(self.recent_alerts)[:20],
            }


# ── GNN inference helper ──────────────────────────────────────────────────────

class LiveInference:
    """
    Runs GNN inference on the global graph.
    In production you'd do subgraph sampling; here we run full-graph
    inference on the pre-loaded graph and look up the txn node by index.
    """

    def __init__(self, device: str = "cpu"):
        self.device   = torch.device(device)
        self.model    = None
        self.data     = None
        self.user_idx = None
        self._ready   = False

    def load(self):
        if not Path(MODEL_PATH).exists():
            print(f"Model not found at {MODEL_PATH}. "
                  "Run training first, or inference will be skipped.")
            return

        print("Loading graph and model...")
        self.data = torch.load(GRAPH_PATH, map_location=self.device)
        with open(USER_IDX_PATH, "rb") as f:
            self.user_idx = pickle.load(f)

        self.model = FraudPredictor(
            metadata=self.data.metadata(), hidden=64
        ).to(self.device)
        self.model.load(MODEL_PATH, device=str(self.device))
        self._ready = True
        print("Model ready for inference.")

    def score(self, txn_row_idx: int) -> float:
        """Return fraud probability for the given transaction index."""
        if not self._ready:
            return -1.0

        with torch.no_grad():
            probs = self.model.predict(
                self.data.x_dict,
                self.data.edge_index_dict,
            )
        if txn_row_idx < len(probs):
            return float(probs[txn_row_idx].item())
        return 0.0


# ── consumer loop ─────────────────────────────────────────────────────────────

STATE    = FraudState()
INC_GRAPH = IncrementalGraph(detect_interval=RING_DETECT_EVERY)
INFERENCE = LiveInference()


def format_alert(msg: dict, score: float) -> dict:
    return {
        "txn_id"  : msg.get("txn_id"),
        "card1"   : msg.get("card1"),
        "amount"  : msg.get("amount"),
        "score"   : round(score, 4),
        "ts"      : time.strftime("%H:%M:%S"),
        "type"    : (
            "CRITICAL"  if score > 0.90 else
            "HIGH"      if score > 0.80 else
            "MEDIUM"
        ),
    }


def consume():
    consumer = Consumer({
        "bootstrap.servers" : KAFKA_BROKER,
        "group.id"          : "fraud-detector-v1",
        "auto.offset.reset" : "latest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC_NAME])
    print(f"Subscribed to '{TOPIC_NAME}'. Waiting for messages...\n")

    txn_buffer : list[dict] = []   # used to map txn_id → row index offline

    try:
        while True:
            raw = consumer.poll(timeout=1.0)

            if raw is None:
                continue
            if raw.error():
                if raw.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"Kafka error: {raw.error()}")
                continue

            try:
                msg = json.loads(raw.value().decode("utf-8"))
            except Exception:
                continue

            STATE.tick()
            txn_id = msg.get("txn_id", -1)
            card1  = msg.get("card1",  -1)
            amount = msg.get("amount", 0.0)

            # Update incremental graph
            INC_GRAPH.add_transaction(src=card1, dst=txn_id, amount=amount)

            # GNN inference (using pre-computed full-graph probabilities)
            # In a real system you'd do per-subgraph inference here
            score = INFERENCE.score(STATE.total_txns % 5000)   # demo approximation

            if score >= FRAUD_THRESHOLD:
                alert = format_alert(msg, score)
                STATE.add_alert(alert)
                print(
                    f"🚨 FRAUD  txn:{txn_id:>10}  card:{card1:>6}"
                    f"  ₹{amount:>10,.0f}  score={score:.3f}"
                )
            elif STATE.total_txns % 200 == 0:
                print(
                    f"   OK    txn:{txn_id:>10}  card:{card1:>6}"
                    f"  ₹{amount:>10,.0f}  score={score:.3f}"
                    f"  total={STATE.total_txns:,}"
                    f"  tps={STATE.get_tps():.0f}"
                )

            # Update ring state
            STATE.current_rings = INC_GRAPH.get_rings()

    except KeyboardInterrupt:
        print("\nConsumer stopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    INFERENCE.load()
    consume()
