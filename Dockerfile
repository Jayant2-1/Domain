# ── Stage 1: Build React frontend ───────────────────────────────────────────
FROM node:20-alpine AS node-builder

WORKDIR /build
COPY frontend-react/package.json ./
COPY frontend-react/package-lock.json* ./
RUN npm install
COPY frontend-react/ ./
RUN npm run build

# ── Stage 2: Python application ────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m -s /bin/bash appuser

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY rag/ rag/
COPY finetune/ finetune/
COPY frontend/ frontend/
COPY sql/ sql/

# Copy React build from Node stage
COPY --from=node-builder /build/dist /app/frontend-react/dist

# Pre-create directories the app writes to
RUN mkdir -p data faiss_index adapters finetune/data && \
    chown -R appuser:appuser /app

# Switch to non-root
USER appuser

# Expose API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
