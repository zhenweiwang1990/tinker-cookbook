#!/bin/bash
# Training script for CUA RL
# 
# This script trains a CUA model using Tinker's RL framework with configurable parameters.
# All training data and logs are saved to the database for monitoring.
#
# Usage examples:
#   # Basic training with default settings
#   ./train.sh
#
#   # Custom log path
#   ./train.sh --log-path ./my_logs
#
#   # Custom model and hyperparameters
#   ./train.sh --model Qwen/Qwen2.5-1.5B-Instruct --lr 2e-5 --group-size 8
#
#   # Skip baseline evaluation
#   ./train.sh --skip-baseline
#
#   # Custom training data (using task_adapter)
#   ./train.sh --train-split train --eval-split eval --train-ratio 0.99
#
#   # Resume from existing log directory
#   ./train.sh --log-path ./logs --resume

set -e  # Exit on error

# ============================================================================
# Default Configuration
# ============================================================================

# Model and LoRA settings
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct"
LORA_RANK=32

# Training data configuration
# Source types: demo_training, demo_eval, demo_all, task_adapter
TRAIN_SOURCE_TYPE="task_adapter"
TRAIN_SPLIT_TYPE="train"     # train or eval (for task_adapter)
EVAL_SOURCE_TYPE="task_adapter"
EVAL_SPLIT_TYPE="eval"        # train or eval (for task_adapter)
TRAIN_RATIO=0.8              # Train/eval split ratio (only used with task_adapter)
SEED=42
CATEGORY="demo"                   # Optional: filter tasks by category (demo, airbnb, instagram)

# Training hyperparameters
GROUP_SIZE=4
GROUPS_PER_BATCH=2
LEARNING_RATE=1e-5
MAX_TOKENS=2048
TEMPERATURE=1.0
KL_PENALTY_COEF=0.0
NUM_SUBSTEPS=1
MAX_TURNS=20

# Logging and evaluation
LOG_PATH="./logs"
EVAL_EVERY=10
SAVE_EVERY=2
NUM_GROUPS_TO_LOG=1
MAX_CONCURRENT_ROLLOUTS=8

# Execution settings
SKIP_BASELINE=true
BEHAVIOR_IF_EXISTS="resume"   # resume, error, or overwrite

# Timeout settings (in seconds)
MAX_TASK_TIME=1800            # 30 minutes
MAX_TURN_TIME=300             # 5 minutes

# Box configuration
BOX_TYPE="android"            # android or pc

# Coordinate generation mode
COORDINATE_MODE="direct"        # gbox or direct
COORDINATE_SCALE="true"           # auto (empty), true, or false (default: auto-detect based on mode)

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --train-source)
            TRAIN_SOURCE_TYPE="$2"
            shift 2
            ;;
        --train-split)
            TRAIN_SPLIT_TYPE="$2"
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
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --groups-per-batch|--batch-size)
            GROUPS_PER_BATCH="$2"
            shift 2
            ;;
        --lr|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --kl-penalty)
            KL_PENALTY_COEF="$2"
            shift 2
            ;;
        --substeps)
            NUM_SUBSTEPS="$2"
            shift 2
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --log-path)
            LOG_PATH="$2"
            shift 2
            ;;
        --eval-every)
            EVAL_EVERY="$2"
            shift 2
            ;;
        --save-every)
            SAVE_EVERY="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT_ROLLOUTS="$2"
            shift 2
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --run-baseline)
            SKIP_BASELINE=false
            shift
            ;;
        --resume)
            BEHAVIOR_IF_EXISTS="resume"
            shift
            ;;
        --overwrite)
            BEHAVIOR_IF_EXISTS="overwrite"
            shift
            ;;
        --box-type)
            BOX_TYPE="$2"
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
            echo "Training script for CUA RL with configurable parameters."
            echo ""
            echo "Model Options:"
            echo "  --model MODEL_NAME              Model to train (default: Qwen/Qwen3-VL-30B-A3B-Instruct)"
            echo "  --lora-rank RANK                LoRA rank (default: 32)"
            echo ""
            echo "Data Options:"
            echo "  --train-source SOURCE           Training data source (default: task_adapter)"
            echo "  --train-split SPLIT             Training split: train/eval (default: train)"
            echo "  --eval-source SOURCE            Eval data source (default: task_adapter)"
            echo "  --eval-split SPLIT              Eval split: train/eval (default: eval)"
            echo "  --train-ratio RATIO             Train/eval split ratio (default: 0.99)"
            echo "  --seed SEED                     Random seed (default: 42)"
            echo "  --category CATEGORY             Filter by category: demo/airbnb/instagram"
            echo ""
            echo "Training Hyperparameters:"
            echo "  --group-size SIZE               Group size (default: 4)"
            echo "  --groups-per-batch SIZE         Groups per batch (default: 2)"
            echo "  --lr, --learning-rate RATE      Learning rate (default: 1e-5)"
            echo "  --max-tokens TOKENS             Max tokens (default: 2048)"
            echo "  --temperature TEMP              Temperature (default: 1.0)"
            echo "  --kl-penalty COEF               KL penalty coefficient (default: 0.0)"
            echo "  --substeps NUM                  Number of substeps (default: 1)"
            echo "  --max-turns TURNS               Max turns per task (default: 3)"
            echo ""
            echo "Logging and Evaluation:"
            echo "  --log-path PATH                 Log directory path (default: ./logs)"
            echo "  --eval-every N                  Evaluate every N batches (default: 10)"
            echo "  --save-every N                  Save checkpoint every N batches (default: 10)"
            echo "  --max-concurrent N              Max concurrent rollouts (default: 8)"
            echo ""
            echo "Execution Options:"
            echo "  --skip-baseline                 Skip baseline evaluation (default)"
            echo "  --run-baseline                  Run baseline evaluation"
            echo "  --resume                        Resume from existing checkpoint (default)"
            echo "  --overwrite                     Overwrite existing logs"
            echo "  --box-type TYPE                 Box type: android/pc (default: android)"
            echo "  --max-task-time SECONDS         Max time per task (default: 1800)"
            echo "  --max-turn-time SECONDS         Max time per turn (default: 300)"
            echo "  --coordinate-mode MODE          Coordinate mode: gbox/direct (default: gbox)"
            echo ""
            echo "Examples:"
            echo "  # Basic training"
            echo "  $0"
            echo ""
            echo "  # Filter tasks by category"
            echo "  $0 --category demo"
            echo ""
            echo "  # Custom model and hyperparameters"
            echo "  $0 --model Qwen/Qwen2.5-3B-Instruct --lr 2e-5 --group-size 8"
            echo ""
            echo "  # Skip baseline and use custom log path"
            echo "  $0 --skip-baseline --log-path ./my_training_run"
            echo ""
            echo "  # Resume from checkpoint"
            echo "  $0 --log-path ./logs --resume"
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
echo "CUA RL Training Script"
echo "============================================"
echo ""

# Check required environment variables
if [ -z "$GBOX_API_KEY" ]; then
    echo "ERROR: GBOX_API_KEY environment variable is not set"
    echo "Please set it with: export GBOX_API_KEY=your_api_key"
    exit 1
fi

# Check if TINKER_API_KEY is set (optional, will use GBOX_API_KEY as fallback)
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
echo "LoRA Rank:            $LORA_RANK"
echo ""
echo "Training Data:"
echo "  Source:             $TRAIN_SOURCE_TYPE"
echo "  Split:              $TRAIN_SPLIT_TYPE"
echo "Evaluation Data:"
echo "  Source:             $EVAL_SOURCE_TYPE"
echo "  Split:              $EVAL_SPLIT_TYPE"
echo "  Train Ratio:        $TRAIN_RATIO"
echo "  Seed:               $SEED"
if [ -n "$CATEGORY" ]; then
    echo "  Category:           $CATEGORY"
fi
echo ""
echo "Hyperparameters:"
echo "  Group Size:         $GROUP_SIZE"
echo "  Groups per Batch:   $GROUPS_PER_BATCH"
echo "  Learning Rate:      $LEARNING_RATE"
echo "  Max Tokens:         $MAX_TOKENS"
echo "  Temperature:        $TEMPERATURE"
echo "  KL Penalty:         $KL_PENALTY_COEF"
echo "  Substeps:           $NUM_SUBSTEPS"
echo "  Max Turns:          $MAX_TURNS"
echo ""
echo "Logging:"
echo "  Log Path:           $LOG_PATH"
echo "  Eval Every:         $EVAL_EVERY"
echo "  Save Every:         $SAVE_EVERY"
echo "  Concurrent Rollouts: $MAX_CONCURRENT_ROLLOUTS"
echo ""
echo "Execution:"
echo "  Skip Baseline:      $SKIP_BASELINE"
echo "  Behavior if Exists: $BEHAVIOR_IF_EXISTS"
echo "  Box Type:           $BOX_TYPE"
echo "  Max Task Time:      ${MAX_TASK_TIME}s"
echo "  Max Turn Time:      ${MAX_TURN_TIME}s"
echo "  Coordinate Mode:    $COORDINATE_MODE"
echo "----------------------------------------"
echo ""

# ============================================================================
# Build Command
# ============================================================================

# Build task configuration JSON strings
TASKS_JSON="{\"source_type\": \"$TRAIN_SOURCE_TYPE\", \"split_type\": \"$TRAIN_SPLIT_TYPE\", \"train_ratio\": $TRAIN_RATIO, \"seed\": $SEED}"
EVAL_TASKS_JSON="{\"source_type\": \"$EVAL_SOURCE_TYPE\", \"split_type\": \"$EVAL_SPLIT_TYPE\", \"train_ratio\": $TRAIN_RATIO, \"seed\": $SEED}"

# Add optional category filter (applies to both train and eval task sets)
if [ -n "$CATEGORY" ]; then
    TASKS_JSON="${TASKS_JSON%?}, \"category\": \"$CATEGORY\"}"
    EVAL_TASKS_JSON="${EVAL_TASKS_JSON%?}, \"category\": \"$CATEGORY\"}"
fi

# Build the command
CMD="uv run python -m tinker_cookbook.recipes.cua_rl.train \
    model_name=\"$MODEL_NAME\" \
    lora_rank=$LORA_RANK \
    tasks='$TASKS_JSON' \
    eval_tasks='$EVAL_TASKS_JSON' \
    seed=$SEED \
    group_size=$GROUP_SIZE \
    groups_per_batch=$GROUPS_PER_BATCH \
    learning_rate=$LEARNING_RATE \
    max_tokens=$MAX_TOKENS \
    temperature=$TEMPERATURE \
    kl_penalty_coef=$KL_PENALTY_COEF \
    num_substeps=$NUM_SUBSTEPS \
    max_turns=$MAX_TURNS \
    log_path=\"$LOG_PATH\" \
    eval_every=$EVAL_EVERY \
    save_every=$SAVE_EVERY \
    max_concurrent_rollouts=$MAX_CONCURRENT_ROLLOUTS \
    skip_baseline=$SKIP_BASELINE \
    behavior_if_log_dir_exists=$BEHAVIOR_IF_EXISTS \
    box_type=$BOX_TYPE \
    max_task_time_seconds=$MAX_TASK_TIME \
    max_turn_time_seconds=$MAX_TURN_TIME \
    coordinate_mode=$COORDINATE_MODE"

# Add coordinate_scale if provided
if [ -n "$COORDINATE_SCALE" ]; then
    CMD="$CMD \
    coordinate_scale=$COORDINATE_SCALE"
fi

# ============================================================================
# Execute Training
# ============================================================================

echo "Starting training..."
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
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "============================================"

exit $EXIT_CODE

