#!/bin/bash

# Start script for training monitor

# Set default database path if not set
if [ -z "$TRAINING_DB_PATH" ]; then
    # Try to find database in common locations
    if [ -f "../logs/training.db" ]; then
        export TRAINING_DB_PATH="../logs/training.db"
    elif [ -f "logs/training.db" ]; then
        export TRAINING_DB_PATH="logs/training.db"
    else
        echo "Warning: Training database not found. Please set TRAINING_DB_PATH environment variable."
        echo "Example: export TRAINING_DB_PATH=/path/to/training.db"
    fi
fi

# Start Next.js dev server
npm run dev

