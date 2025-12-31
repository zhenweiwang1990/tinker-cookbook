#!/bin/bash

# Docker start script for training monitor
# This script uses docker-compose to start PostgreSQL and the training monitor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Default values
BUILD=true
PORT=3000

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --no-build)
      BUILD=false
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --port PORT       Port to expose (default: 3000)"
      echo "  --no-build        Skip building the image"
      echo "  --help            Show this help message"
      echo ""
      echo "This script starts PostgreSQL and the training monitor using docker-compose."
      echo "The database connection is configured via environment variables in docker-compose.yml"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
  echo "Error: docker-compose or docker is not installed"
  exit 1
fi

# Use docker compose (newer) or docker-compose (older)
if command -v docker &> /dev/null && docker compose version &> /dev/null 2>&1; then
  DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
  DOCKER_COMPOSE="docker-compose"
else
  echo "Error: docker compose is not available"
  exit 1
fi

echo "Starting training monitor with PostgreSQL..."
echo ""

# Build if requested
if [ "$BUILD" = true ]; then
  echo "Building Docker images..."
  $DOCKER_COMPOSE build
  echo ""
fi

# Start services
echo "Starting services..."
$DOCKER_COMPOSE up -d

echo ""
echo "Services started!"
echo ""
echo "Training monitor: http://localhost:${PORT}"
echo "PostgreSQL: localhost:5432"
echo ""
echo "To view logs:"
echo "  $DOCKER_COMPOSE logs -f"
echo ""
echo "To stop services:"
echo "  $DOCKER_COMPOSE down"
echo ""
