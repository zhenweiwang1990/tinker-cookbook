.PHONY: help build start stop logs restart clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker build -t cua-rl-training-monitor:latest .

start: ## Start the container
	./scripts/docker-start.sh

stop: ## Stop the container
	./scripts/docker-stop.sh

logs: ## View container logs
	./scripts/docker-logs.sh

restart: stop start ## Restart the container

clean: ## Remove container and image
	docker stop training-monitor 2>/dev/null || true
	docker rm training-monitor 2>/dev/null || true
	docker rmi cua-rl-training-monitor:latest 2>/dev/null || true

dev: ## Run in development mode (local, not Docker)
	npm run dev

install: ## Install dependencies
	npm install

