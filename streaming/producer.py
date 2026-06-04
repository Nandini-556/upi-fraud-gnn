"""
Kafka Producer — simulates live UPI transaction stream from the dataset.

Usage:
  python streaming/producer.py

Requires:
  - Kafka running on localhost:9092 (docker-compose up -d kafka)
  - Processed dataset at data/processed/transactions.parquet
"""

import json
import time
import random
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic


KAFKA_BROKER = "localhost:9092"
TOPIC_NAME   = "upi-transactions"
PARQUET_PATH = Path("data/processed/transactions.parquet")


def ensure_topic():
    admin = AdminClient({"bootstrap.servers": KAFKA_BROKER})
    existing = admin.list_topics(timeout=5).topics

    if TOPIC_NAME not in existing:
        print(f"Creating Kafka topic: {TOPIC_NAME}")
        admin.create_topics([
            NewTopic(TOPIC_NAME, num_partitions=3, replication_factor=1)
        ])
        time.sleep(1)
    else:
        print(f"Topic '{TOPIC_NAME}' already exists.")


def delivery_report(err, msg):
    if err:
        print(f"Message delivery failed: {err}")


def stream(speed: float = 1.0, max_msgs: int = None):
    """
    Stream transactions to Kafka.

    speed   : 1.0 = real-time simulation, 10.0 = 10x faster
    max_msgs: stop after N messages (None = stream forever)
    """
    if not PARQUET_PATH.exists():
        print(f"Dataset not found at {PARQUET_PATH}")
        print("Run: python data/download_data.py && python data/graph_builder.py")
        return

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} transactions.")

    # Sort by transaction time so stream is chronological
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT").reset_index(drop=True)

    producer = Producer({
        "bootstrap.servers": KAFKA_BROKER,
        "linger.ms"        : 5,
        "batch.num.messages": 100,
    })

    sent  = 0
    start = time.time()

    print(f"Streaming to topic '{TOPIC_NAME}' (speed={speed}x)...")
    print("Ctrl+C to stop.\n")

    for _, row in df.iterrows():
        if max_msgs and sent >= max_msgs:
            break

        msg = {
            "txn_id"   : int(row["TransactionID"]),
            "amount"   : float(row["TransactionAmt"]),
            "card1"    : int(row["card1"]),
            "is_fraud" : int(row["isFraud"]),
            "ts"       : time.time(),
        }
        if "ProductCD" in row:
            msg["product"] = str(row["ProductCD"])

        producer.produce(
            TOPIC_NAME,
            key   = str(msg["card1"]),
            value = json.dumps(msg).encode("utf-8"),
            callback = delivery_report,
        )
        producer.poll(0)
        sent += 1

        # Simulate inter-transaction delay
        delay = random.uniform(0.005, 0.05) / speed
        time.sleep(delay)

        if sent % 500 == 0:
            elapsed = time.time() - start
            tps = sent / elapsed
            print(f"  Sent {sent:,} msgs | {tps:.0f} msgs/s | "
                  f"fraud in stream: {msg['is_fraud']}")

    producer.flush()
    print(f"\nDone. Sent {sent:,} messages.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed",    type=float, default=5.0,
                        help="Playback speed multiplier (default 5x)")
    parser.add_argument("--max-msgs", type=int,   default=None,
                        help="Stop after N messages")
    args = parser.parse_args()

    try:
        ensure_topic()
        stream(speed=args.speed, max_msgs=args.max_msgs)
    except KeyboardInterrupt:
        print("\nProducer stopped.")
