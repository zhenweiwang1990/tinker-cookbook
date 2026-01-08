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
echo "Waiting for services to be healthy..."

# Wait for services to be healthy
MAX_WAIT=60
WAIT_TIME=0
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    # Check if both services are healthy
    POSTGRES_HEALTHY=$($DOCKER_COMPOSE ps postgres | grep -q "healthy" && echo "yes" || echo "no")
    MONITOR_HEALTHY=$($DOCKER_COMPOSE ps training-monitor | grep -q "Up" && echo "yes" || echo "no")
    
    if [ "$POSTGRES_HEALTHY" = "yes" ] && [ "$MONITOR_HEALTHY" = "yes" ]; then
        echo "✓ Services are healthy"
        break
    fi
    
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "⚠️  Services may not be fully ready yet"
fi

echo ""
echo "Initializing database schema..."
cd "$PROJECT_DIR/.."  # Go to tinker-cookbook root

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "⚠️  Warning: 'uv' not found. Skipping automatic database initialization."
    echo "   Please run manually: ./training-monitor/scripts/init-database.sh"
else
    # Run database initialization
    cd tinker_cookbook/recipes/cua_rl
    if uv run python migrate_database.py 2>&1 | grep -q "SUCCESS\|up to date"; then
        echo "✓ Database schema initialized"
    else
        echo "⚠️  Database initialization may have failed"
        echo "   Please run manually: ./training-monitor/scripts/init-database.sh"
    fi
fi

cd "$PROJECT_DIR"

echo ""
echo "============================================"
echo "Training monitor: http://localhost:${PORT}"
echo "PostgreSQL: localhost:5433"
echo "============================================"
echo ""
echo "To view logs:"
echo "  $DOCKER_COMPOSE logs -f"
echo ""
echo "To stop services:"
echo "  $DOCKER_COMPOSE down"
echo ""
