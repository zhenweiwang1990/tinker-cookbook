#!/bin/bash

# View logs from the training monitor and PostgreSQL containers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Use docker compose (newer) or docker-compose (older)
if command -v docker &> /dev/null && docker compose version &> /dev/null 2>&1; then
  DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
  DOCKER_COMPOSE="docker-compose"
else
  echo "Error: docker compose is not available"
  exit 1
fi

# Check if services are running
if ! $DOCKER_COMPOSE ps | grep -q "Up"; then
  echo "Services are not running"
  echo "Start them with: ./scripts/docker-start.sh"
  exit 1
fi

# Show logs
if [ "$1" = "postgres" ]; then
  $DOCKER_COMPOSE logs -f postgres
elif [ "$1" = "monitor" ]; then
  $DOCKER_COMPOSE logs -f training-monitor
else
  $DOCKER_COMPOSE logs -f
fi
