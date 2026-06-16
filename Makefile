# WiFi Pose Intelligence Platform — dev shortcuts
# Windows users: `make` works under Git Bash; equivalents are in README.

.PHONY: help install install-ml lint test checkpoint export benchmark \
        api worker producer up down seed fmt frontend-dev frontend-build

help:
	@echo "Targets:"
	@echo "  install        Install runtime + dev deps (editable)"
	@echo "  install-ml     Also install torch/h5py for training & export"
	@echo "  checkpoint     Train a small demo model on simulated CSI -> .pth"
	@echo "  export         Export the .pth checkpoint to ONNX"
	@echo "  benchmark      Compare PyTorch vs ONNX Runtime latency"
	@echo "  api            Run the FastAPI server (reload)"
	@echo "  worker         Run the Redis-stream inference consumer"
	@echo "  producer       Run the synthetic CSI producer"
	@echo "  up / down      Start / stop the full docker-compose stack"
	@echo "  seed           Insert demo events into the database"
	@echo "  test / lint    Run pytest / ruff"

install:
	pip install -e ".[dev]"

install-ml:
	pip install -e ".[dev,ml]"

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest -q

checkpoint:
	wifipose-checkpoint --epochs 8 --out checkpoints/wifipose.pth

export:
	wifipose-export --checkpoint checkpoints/wifipose.pth --out checkpoints/wifipose.onnx

benchmark:
	wifipose-benchmark --checkpoint checkpoints/wifipose.pth --onnx checkpoints/wifipose.onnx

api:
	uvicorn wifipose.api.main:app --reload --host 0.0.0.0 --port 8000

worker:
	wifipose-consume

producer:
	wifipose-produce

up:
	docker compose up --build -d

down:
	docker compose down -v

seed:
	python scripts/seed_demo.py

frontend-dev:
	cd frontend && npm install && npm run dev

frontend-build:
	cd frontend && npm install && npm run build
