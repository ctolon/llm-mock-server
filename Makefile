.PHONY: help install run dev test docker-build docker-up docker-down k8s-apply k8s-delete clean

PORT ?= 8000
HOST ?= 0.0.0.0
IMAGE_NAME ?= openai-vllm-mock:latest

help: ## Show available Make targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies via uv
	uv sync

run: ## Run the production FastAPI server
	uv run uvicorn main:app --host $(HOST) --port $(PORT)

dev: ## Run the FastAPI server in hot-reload development mode
	uv run uvicorn main:app --host $(HOST) --port $(PORT) --reload

test: ## Run the automated pytest test suite
	uv run pytest -v

docker-build: ## Build the Docker image
	docker build -t $(IMAGE_NAME) .

docker-up: ## Start the server via Docker Compose
	docker compose up --build -d

docker-down: ## Stop the Docker Compose stack
	docker compose down

k8s-apply: ## Apply the Kubernetes manifests
	kubectl apply -f kubernetes.yaml

k8s-delete: ## Remove the Kubernetes resources
	kubectl delete -f kubernetes.yaml

clean: ## Remove virtual environments and cache directories
	rm -rf .pytest_cache __pycache__ .ruff_cache .mypy_cache
