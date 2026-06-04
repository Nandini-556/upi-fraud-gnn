#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  UPI Fraud Sentinel — Start everything
#  Usage: ./scripts/run_all.sh
# ─────────────────────────────────────────────────────────────
set -e

GREEN='\033[0;32m'; BLUE='\033[0;34m'; AMBER='\033[0;33m'; NC='\033[0m'; BOLD='\033[1m'

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${AMBER}[WARN]${NC}  $1"; }

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
    ok "venv activated"
else
    warn "venv not found — run scripts/setup.sh first"
fi

# Check processed graph
if [ ! -f "data/processed/graph.pt" ]; then
    info "Graph not built. Running data pipeline..."
    python data/download_data.py
    python data/graph_builder.py
fi

# Check model checkpoint
if [ ! -f "model/checkpoints/best_model.pt" ]; then
    info "Model not trained. Running training (this may take a few minutes)..."
    python model/train.py
fi

# Start Kafka via Docker (background)
if command -v docker-compose &>/dev/null; then
    info "Starting Kafka..."
    docker-compose up -d zookeeper kafka
    info "Waiting for Kafka to be ready..."
    sleep 8
    ok "Kafka running on localhost:9092"

    # Start producer in background
    info "Starting Kafka producer (streaming simulation)..."
    python streaming/producer.py --speed 5 &
    PRODUCER_PID=$!
    echo "  Producer PID: $PRODUCER_PID"

    # Start consumer in background
    info "Starting Kafka consumer (live inference)..."
    python streaming/consumer.py &
    CONSUMER_PID=$!
    echo "  Consumer PID: $CONSUMER_PID"
else
    warn "Docker not found — skipping Kafka. API will run in demo mode."
fi

echo ""
echo -e "${GREEN}${BOLD}Starting FastAPI server...${NC}"
echo ""
echo "  Dashboard → http://localhost:8000/dashboard"
echo "  API docs  → http://localhost:8000/docs"
echo "  Press Ctrl+C to stop"
echo ""

# Start API (foreground)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Cleanup on exit
if [ ! -z "$PRODUCER_PID" ]; then kill $PRODUCER_PID 2>/dev/null || true; fi
if [ ! -z "$CONSUMER_PID" ]; then kill $CONSUMER_PID 2>/dev/null || true; fi
