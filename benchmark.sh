#!/bin/bash
# Benchmark evaluation script for CUA agents
#
# This script evaluates a model on a specific dataset and saves results to the database.
# Unlike training baseline evaluation, this is a standalone benchmark that can:
# - Use different model providers (Tinker, OpenAI, Anthropic, etc.)
# - Evaluate on custom datasets
# - Run without a training session
# - Compare different models on the same tasks
#
# Usage examples:
#   # Benchmark with Tinker provider (default)
#   ./benchmark.sh
#
#   # Benchmark with specific model
#   ./benchmark.sh --model Qwen/Qwen3-VL-30B-A3B-Instruct
#
#   # Benchmark with checkpoint
#   ./benchmark.sh --model-path /path/to/checkpoint
#
#   # Benchmark with OpenAI (future support)
#   ./benchmark.sh --provider openai --model gpt-4
#
#   # Custom dataset
#   ./benchmark.sh --eval-source task_adapter --eval-split eval
#
#   # Custom benchmark name
#   ./benchmark.sh --name my_benchmark_v1

set -e  # Exit on error

# ============================================================================
# Default Configuration
# ============================================================================

# Model settings
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_PATH=""                              # Optional: checkpoint path for fine-tuned models

# Evaluation dataset configuration
EVAL_SOURCE_TYPE="task_adapter"
EVAL_SPLIT_TYPE="eval"
TRAIN_RATIO=0.99
SEED=42

# Benchmark settings
BENCHMARK_NAME=""                          # Optional: custom name for this benchmark run
MAX_TURNS=3
TEMPERATURE=1.0
MAX_TOKENS=2048

# Box configuration
BOX_TYPE="android"

# Coordinate generation mode
COORDINATE_MODE="gbox"                    # gbox or direct
COORDINATE_SCALE=""                       # auto (empty), true, or false

# Concurrency and timeout
MAX_CONCURRENT_ROLLOUTS=8
MAX_TASK_TIME=1800                         # 30 minutes
MAX_TURN_TIME=300                          # 5 minutes

# Logging
LOG_PATH="./benchmark_logs"

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-path|--checkpoint)
            MODEL_PATH="$2"
            shift 2
            ;;
        --eval-source)
            EVAL_SOURCE_TYPE="$2"
            shift 2
            ;;
        --eval-split)
            EVAL_SPLIT_TYPE="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --name|--benchmark-name)
            BENCHMARK_NAME="$2"
            shift 2
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --box-type)
            BOX_TYPE="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT_ROLLOUTS="$2"
            shift 2
            ;;
        --max-task-time)
            MAX_TASK_TIME="$2"
            shift 2
            ;;
        --max-turn-time)
            MAX_TURN_TIME="$2"
            shift 2
            ;;
        --log-path)
            LOG_PATH="$2"
            shift 2
            ;;
        --coordinate-mode)
            COORDINATE_MODE="$2"
            shift 2
            ;;
        --coordinate-scale)
            COORDINATE_SCALE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Benchmark evaluation script for CUA agents."
            echo ""
            echo "Model Options:"
            echo "  --model MODEL_NAME              Model to evaluate (default: Qwen/Qwen2.5-1.5B-Instruct)"
            echo "  --model-path PATH               Checkpoint path for fine-tuned models"
            echo ""
            echo "Dataset Options:"
            echo "  --eval-source SOURCE            Eval data source (default: task_adapter)"
            echo "  --eval-split SPLIT              Eval split: train/eval (default: eval)"
            echo "  --train-ratio RATIO             Train/eval split ratio (default: 0.99)"
            echo "  --seed SEED                     Random seed (default: 42)"
            echo ""
            echo "Benchmark Options:"
            echo "  --name, --benchmark-name NAME   Custom name for this benchmark run"
            echo "  --max-turns TURNS               Max turns per task (default: 20)"
            echo "  --temperature TEMP              Temperature (default: 1.0)"
            echo "  --max-tokens TOKENS             Max tokens (default: 2048)"
            echo ""
            echo "Execution Options:"
            echo "  --box-type TYPE                 Box type: android/pc (default: android)"
            echo "  --max-concurrent N              Max concurrent rollouts (default: 8)"
            echo "  --max-task-time SECONDS         Max time per task (default: 1800)"
            echo "  --max-turn-time SECONDS         Max time per turn (default: 300)"
            echo "  --log-path PATH                 Log directory path (default: ./benchmark_logs)"
            echo "  --coordinate-mode MODE          Coordinate mode: gbox/direct (default: gbox)"
            echo ""
            echo "Examples:"
            echo "  # Basic benchmark with Tinker"
            echo "  $0"
            echo ""
            echo "  # Benchmark specific model"
            echo "  $0 --model Qwen/Qwen2.5-3B-Instruct"
            echo ""
            echo "  # Benchmark fine-tuned checkpoint"
            echo "  $0 --model-path /path/to/checkpoint"
            echo ""
            echo "  # Custom dataset and settings"
            echo "  $0 --eval-source demo_eval --seed 123 --name my_benchmark"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Environment Check
# ============================================================================

echo "============================================"
echo "CUA Benchmark Evaluation Script"
echo "============================================"
echo ""

# Check required environment variables
if [ -z "$GBOX_API_KEY" ]; then
    echo "ERROR: GBOX_API_KEY environment variable is not set"
    echo "Please set it with: export GBOX_API_KEY=your_api_key"
    exit 1
fi

if [ -z "$TINKER_API_KEY" ]; then
    echo "Note: TINKER_API_KEY not set, will use GBOX_API_KEY as fallback"
fi

# Check database connection
if [ -z "$DATABASE_URL" ]; then
    echo "Note: DATABASE_URL not set, using default PostgreSQL connection"
    echo "      (host=127.0.0.1, port=5433, db=training_db)"
fi

echo ""

# ============================================================================
# Display Configuration
# ============================================================================

echo "Configuration:"
echo "----------------------------------------"
echo "Model:                $MODEL_NAME"
if [ -n "$MODEL_PATH" ]; then
    echo "Checkpoint:           $MODEL_PATH"
fi
echo ""
echo "Evaluation Dataset:"
echo "  Source:             $EVAL_SOURCE_TYPE"
echo "  Split:              $EVAL_SPLIT_TYPE"
echo "  Train Ratio:        $TRAIN_RATIO"
echo "  Seed:               $SEED"
echo ""
echo "Benchmark Settings:"
if [ -n "$BENCHMARK_NAME" ]; then
    echo "  Name:               $BENCHMARK_NAME"
fi
echo "  Max Turns:          $MAX_TURNS"
echo "  Temperature:        $TEMPERATURE"
echo "  Max Tokens:         $MAX_TOKENS"
echo ""
echo "Execution:"
echo "  Box Type:           $BOX_TYPE"
echo "  Concurrent Rollouts: $MAX_CONCURRENT_ROLLOUTS"
echo "  Max Task Time:      ${MAX_TASK_TIME}s"
echo "  Max Turn Time:      ${MAX_TURN_TIME}s"
echo "  Log Path:           $LOG_PATH"
echo "  Coordinate Mode:    $COORDINATE_MODE"
echo "----------------------------------------"
echo ""

# ============================================================================
# Build Command
# ============================================================================

# Build evaluation task configuration JSON
EVAL_TASKS_JSON="{\"source_type\": \"$EVAL_SOURCE_TYPE\", \"split_type\": \"$EVAL_SPLIT_TYPE\", \"train_ratio\": $TRAIN_RATIO, \"seed\": $SEED}"

# Build the command
CMD="uv run python -m tinker_cookbook.recipes.cua_rl.benchmark \
    model_name=\"$MODEL_NAME\""

# Add optional model_path
if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD \
    model_path=\"$MODEL_PATH\""
fi

# Add optional benchmark_name
if [ -n "$BENCHMARK_NAME" ]; then
    CMD="$CMD \
    benchmark_name=\"$BENCHMARK_NAME\""
fi

# Add remaining parameters
CMD="$CMD \
    eval_tasks='$EVAL_TASKS_JSON' \
    seed=$SEED \
    max_turns=$MAX_TURNS \
    temperature=$TEMPERATURE \
    max_tokens=$MAX_TOKENS \
    box_type=$BOX_TYPE \
    max_concurrent_rollouts=$MAX_CONCURRENT_ROLLOUTS \
    max_task_time_seconds=$MAX_TASK_TIME \
    max_turn_time_seconds=$MAX_TURN_TIME \
    log_path=\"$LOG_PATH\" \
    coordinate_mode=$COORDINATE_MODE"

# Add coordinate_scale if provided
if [ -n "$COORDINATE_SCALE" ]; then
    CMD="$CMD \
    coordinate_scale=$COORDINATE_SCALE"
fi

# ============================================================================
# Execute Benchmark
# ============================================================================

echo "Starting benchmark evaluation..."
echo ""
echo "Command:"
echo "$CMD"
echo ""
echo "============================================"
echo ""

# Execute the command
eval $CMD

# ============================================================================
# Completion
# ============================================================================

EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Benchmark evaluation completed successfully!"
    echo ""
    echo "Results have been saved to the database."
    echo "View results in the training monitor or query the database."
else
    echo "Benchmark evaluation failed with exit code: $EXIT_CODE"
fi
echo "============================================"

exit $EXIT_CODE

