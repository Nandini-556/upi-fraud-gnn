FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only PyTorch for Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch_geometric
RUN pip install --no-cache-dir \
    torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Make __init__.py files so imports work as packages
RUN touch data/__init__.py model/__init__.py \
         streaming/__init__.py detection/__init__.py api/__init__.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
