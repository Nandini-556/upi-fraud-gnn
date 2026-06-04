#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  UPI Fraud Sentinel — One-command setup
#  Usage: chmod +x scripts/setup.sh && ./scripts/setup.sh
# ─────────────────────────────────────────────────────────────
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; AMBER='\033[0;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${AMBER}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo -e "${BOLD}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║    UPI Fraud Sentinel — Setup         ║"
echo "  ║    GraphSAGE + GAT + Kafka            ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${NC}"

# ── Python check ──────────────────────────────────────────────
info "Checking Python version..."
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
PY_MINOR=$(echo $PY_VER | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ]; then
    error "Python 3.10+ required. Found: $PY_VER"
fi
ok "Python $PY_VER"

# ── Virtual environment ───────────────────────────────────────
info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "Created venv/"
else
    ok "venv/ already exists"
fi

source venv/bin/activate

# ── pip upgrade ───────────────────────────────────────────────
info "Upgrading pip..."
pip install --quiet --upgrade pip

# ── Detect GPU ────────────────────────────────────────────────
CUDA_AVAILABLE=false
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
    info "CUDA $CUDA_VER detected — installing GPU PyTorch"
    CUDA_AVAILABLE=true
    PYG_CUDA="cu118"
    TORCH_IDX="https://download.pytorch.org/whl/cu118"
else
    info "No CUDA found — installing CPU PyTorch"
    PYG_CUDA="cpu"
    TORCH_IDX="https://download.pytorch.org/whl/cpu"
fi

# ── PyTorch ───────────────────────────────────────────────────
info "Installing PyTorch 2.2.0..."
pip install --quiet torch==2.2.0 --index-url $TORCH_IDX
ok "PyTorch installed"

# ── PyTorch Geometric ─────────────────────────────────────────
info "Installing PyTorch Geometric..."
pip install --quiet torch_geometric
pip install --quiet torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+${PYG_CUDA}.html
ok "PyG installed"

# ── Other requirements ────────────────────────────────────────
info "Installing project dependencies..."
pip install --quiet -r requirements.txt
ok "All dependencies installed"

# ── Package init files ────────────────────────────────────────
touch data/__init__.py model/__init__.py \
      streaming/__init__.py detection/__init__.py \
      api/__init__.py scripts/__init__.py 2>/dev/null || true

# ── Directories ───────────────────────────────────────────────
mkdir -p data/raw data/processed model/checkpoints
ok "Directories created"

# ── Docker check ──────────────────────────────────────────────
if command -v docker &>/dev/null; then
    ok "Docker found — you can use docker-compose for Kafka"
else
    warn "Docker not found — you'll need to install Kafka manually for streaming"
fi

echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Download dataset:     python data/download_data.py"
echo "  2. Build graph:          python data/graph_builder.py"
echo "  3. Train model:          python model/train.py"
echo "  4. Start API:            uvicorn api.main:app --reload --port 8000"
echo "  5. Open dashboard:       http://localhost:8000/dashboard"
echo ""
echo "  Or run everything:       ./scripts/run_all.sh"
echo ""
