#!/bin/bash

# Stop and remove the training monitor and PostgreSQL containers

set -e

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

echo "Stopping services..."
$DOCKER_COMPOSE down

echo "✓ Services stopped and removed"
