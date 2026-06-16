# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: train a demo checkpoint + export ONNX (build-time, needs torch) ──
# This bakes a ready-to-serve ONNX model into the image so the deployed demo
# works with no runtime training step. Skip by mounting your own checkpoint.
FROM python:3.11-slim AS modelbuilder
WORKDIR /build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[ml]"
RUN wifipose-checkpoint --epochs 6 --out /build/checkpoints/wifipose.pth \
 && wifipose-export --checkpoint /build/checkpoints/wifipose.pth --out /build/checkpoints/wifipose.onnx

# ── Stage 3: lean runtime (onnxruntime only, NO torch) ───────────────────────
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY --from=modelbuilder /build/checkpoints/wifipose.onnx /app/checkpoints/wifipose.onnx
COPY --from=frontend /app/frontend/dist /app/frontend/dist

EXPOSE 8000
# PORT is provided by most PaaS (Railway/Render); default to 8000 locally.
CMD ["sh", "-c", "uvicorn wifipose.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
