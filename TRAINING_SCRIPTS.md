# CUA RL Training & Benchmark Scripts

Quick access scripts for training and benchmarking Computer Use Agent (CUA) models.

## Quick Start

### Prerequisites

1. **Start the database** (training-monitor includes PostgreSQL):

```bash
cd training-monitor
make start
# or
./scripts/docker-start.sh
```

This starts PostgreSQL on port 5433 and the web UI on port 3001.

2. **Set environment variables**:

```bash
export GBOX_API_KEY=your_api_key
export TINKER_API_KEY=your_tinker_api_key
```

### Training

```bash
# Run training with defaults
./train.sh

# Database migrations run automatically! No manual setup needed.

# View all options
./train.sh --help
```

### Benchmark

```bash
# Run benchmark with defaults
./benchmark.sh

# Benchmark specific model
./benchmark.sh --model Qwen/Qwen2.5-3B-Instruct

# Benchmark checkpoint
./benchmark.sh --model-path ./logs/checkpoint_000100.pt

# View all options
./benchmark.sh --help
```

## Common Options

### Training (`train.sh`)

```bash
# Custom model and hyperparameters
./train.sh --model Qwen/Qwen2.5-3B-Instruct --lr 2e-5 --group-size 8

# Skip baseline evaluation
./train.sh --skip-baseline

# Resume from checkpoint
./train.sh --log-path ./logs --resume

# Production training
./train.sh \
  --model Qwen/Qwen2.5-3B-Instruct \
  --lr 2e-5 \
  --group-size 8 \
  --groups-per-batch 4 \
  --max-concurrent 16
```

### Benchmark (`benchmark.sh`)

```bash
# Named benchmark
./benchmark.sh --name my_benchmark_v1

# Custom dataset
./benchmark.sh --eval-source demo_eval --seed 123

# Compare models
./benchmark.sh --model Qwen/Qwen2.5-1.5B-Instruct --name qwen_1.5b
./benchmark.sh --model Qwen/Qwen2.5-3B-Instruct --name qwen_3b
```

## Environment Variables

**Required:**
- `GBOX_API_KEY` - API key for GBox (Android/PC simulator)

**Optional:**
- `TINKER_API_KEY` - Tinker API key (defaults to GBOX_API_KEY)
- `DATABASE_URL` - PostgreSQL connection URL
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB` - Database connection details

## Monitoring

All training and benchmark results are automatically saved to PostgreSQL database.

View results in the training monitor:

```bash
cd training-monitor
npm run dev
# Visit http://localhost:3000
```

## Documentation

For detailed documentation, see:
- [Scripts README](tinker_cookbook/recipes/cua_rl/SCRIPTS_README.md) - Complete guide
- [Quick Reference](tinker_cookbook/recipes/cua_rl/SCRIPTS_QUICKREF.md) - Quick lookup
- [Training Guide](tinker_cookbook/recipes/cua_rl/tasks/TRAINING_GUIDE.md) - Best practices
- [Database Usage](tinker_cookbook/recipes/cua_rl/database/DATABASE_USAGE.md) - Database info

## Examples

### Quick Test

```bash
# Small model, minimal settings
./train.sh \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --group-size 2 \
  --groups-per-batch 1 \
  --skip-baseline
```

### Production Training

```bash
./train.sh \
  --model Qwen/Qwen2.5-3B-Instruct \
  --lr 2e-5 \
  --group-size 8 \
  --groups-per-batch 4 \
  --eval-every 10 \
  --save-every 10 \
  --max-concurrent 16 \
  --log-path ./production_logs
```

### Model Comparison

```bash
# Baseline
./benchmark.sh --model Qwen/Qwen2.5-1.5B-Instruct --name baseline

# After training
./benchmark.sh --model-path ./logs/checkpoint_000100.pt --name trained

# Compare in training monitor
```

## Troubleshooting

**API key not set:**
```bash
export GBOX_API_KEY=your_api_key
export TINKER_API_KEY=your_tinker_api_key
```

**Database connection failed:**
```bash
cd training-monitor
make start
# or check if containers are running
docker-compose ps
```

**Database schema error ("column does not exist"):**

This should never happen as migrations run automatically, but if it does:

```bash
cd tinker_cookbook/recipes/cua_rl
uv run python migrate_database.py
```

See `tinker_cookbook/recipes/cua_rl/DATABASE_SETUP.md` for details.

**Log directory exists:**
```bash
./train.sh --log-path ./logs --resume  # Resume
# or
./train.sh --log-path ./logs --overwrite  # Start fresh
```

---

For more information, run `./train.sh --help` or `./benchmark.sh --help`

