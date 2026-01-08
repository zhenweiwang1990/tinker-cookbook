#!/bin/bash

# Initialize database schema for CUA RL training
# This script runs the necessary migrations to create/update database tables.
#
# Usage:
#   ./scripts/init-database.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================"
echo "Database Initialization"
echo "============================================"
echo ""

# Check if PostgreSQL is running
echo "Checking PostgreSQL connection..."
if ! PGPASSWORD=training_password psql -h 127.0.0.1 -p 5433 -U training_user -d training_db -c "SELECT 1" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to PostgreSQL at 127.0.0.1:5433"
    echo ""
    echo "Please start PostgreSQL first:"
    echo "  cd training-monitor && make start"
    echo ""
    exit 1
fi

echo "✓ PostgreSQL is running"
echo ""

# Run migrations using Python script
echo "Running database migrations..."
cd "$PROJECT_ROOT/tinker_cookbook/recipes/cua_rl"

if [ ! -f "alembic.ini" ]; then
    echo "ERROR: alembic.ini not found"
    echo "Are you in the right directory?"
    exit 1
fi

# Run migrations
uv run python migrate_database.py

echo ""
echo "============================================"
echo "✅ Database initialized successfully!"
echo "============================================"
echo ""
echo "You can now:"
echo "  - View the training monitor: http://localhost:3001"
echo "  - Run training: ./train.sh"
echo "  - Run benchmarks: ./benchmark.sh"
echo ""

