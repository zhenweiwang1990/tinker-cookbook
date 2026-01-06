# CUA RL Database Implementation Summary

## Implementation Status

✅ **All core features have been implemented**

### 1. Database Infrastructure
- ✅ SQLAlchemy ORM model definitions (`database_models.py`)
- ✅ Database initialization and session management (`database.py`)
- ✅ Complete DAO layer (`database_dao.py`)
- ✅ Global database context (`database_context.py`)
- ✅ Alembic database migrations (`alembic/`)
- ✅ Database rebuild script (`rebuild_database.py`)

### 2. Data Recording Modules
- ✅ Task and validator recording (`database_task_loader.py`)
- ✅ Rollout recording (`database_rollout.py`)
- ✅ Step recording (`database_step.py`)
- ✅ Eval and Baseline recording (`database_eval.py`)
- ✅ Training loop hooks (`database_training_hooks.py`)

### 3. Integration Points
- ✅ Database initialization on training start (`train.py`)
- ✅ Automatic saving on task loading (`task_loader.py`, `cua_env.py`)
- ✅ Recording during rollout (`rollout.py`)
- ✅ Training step recording (via hooks)
- ✅ Evaluation and baseline recording (via wrappers)

## Database Location

The database is managed by a PostgreSQL server, and its location depends on the PostgreSQL configuration.

## Automatically Recorded Data

### Training Session (Training)
- Training configuration and hyperparameters
- Training status and progress
- Start/end times

### Tasks and Validators (Task & Validator)
- All loaded tasks are automatically saved
- Validator information is saved
- Support for loading tasks from database

### Training Steps (Step)
- Detailed information for each step
- Rollout and training progress
- Metrics and loss values

### Rollout Groups (Group)
- Each step/eval/baseline contains multiple groups
- Group status and progress
- Group-level statistics (number of rollouts, success rate, etc.)

### Rollouts (Rollout)
- Complete rollout information
- Task execution results
- Validation results
- Rewards and metrics
- Supports three source types: step/eval/baseline
- Belongs to a group (associated via group_id)

### Turns, Actions, Observations (Turn, Action, Obs)
- Detailed information for each turn
- All actions and observations

### Evaluations (Eval & Baseline)
- Evaluation results
- Baseline evaluation results

## Usage

Database integration is fully automatic and requires no additional configuration. Simply start training normally:

```bash
python -m tinker_cookbook.recipes.cua_rl.train \
    --model_name "Qwen/Qwen3-VL-30B-A3B-Instruct" \
    --log_path "/path/to/logs"
```

The database will be automatically created and updated in the PostgreSQL server.

## Query Examples

```python
from tinker_cookbook.recipes.cua_rl.database import init_database, get_session
from tinker_cookbook.recipes.cua_rl.database_dao import (
    get_training_by_run_name,
    list_rollouts_by_step,
    get_status_history,
)

# Initialize database connection
init_database("postgresql://training_user:training_password@127.0.0.1:5432/training_db")

# Query training records
with get_session() as session:
    training = get_training_by_run_name(session, "cua_rl-...")
    print(f"Training ID: {training.id}")
    print(f"Status: {training.status}")
    print(f"Progress: {training.progress_percent}%")
    
    # Query all rollouts for a step
    steps = training.steps
    if steps:
        step = steps[0]
        rollouts = list_rollouts_by_step(session, step.id)
        print(f"Step {step.step} has {len(rollouts)} rollouts")
    
    # Query status history
    history = get_status_history(session, "training", training.id)
    for h in history:
        print(f"{h.changed_at}: {h.old_status} -> {h.new_status}")
```

## File Structure

```
tinker_cookbook/recipes/cua_rl/
├── database_models.py          # SQLAlchemy model definitions
├── database.py                 # Database initialization and session management
├── database_dao.py             # DAO layer (CRUD for all tables)
├── database_context.py         # Global database context
├── database_task_loader.py     # Task and validator database operations
├── database_rollout.py          # Rollout database operations
├── database_step.py             # Step database operations
├── database_eval.py             # Eval and Baseline database operations
├── database_training_hooks.py   # Training loop hooks
├── database_schema.md           # Database table structure documentation
├── DATABASE_USAGE.md           # Usage guide
├── DATABASE_IMPLEMENTATION.md  # Implementation summary (this file)
├── rebuild_database.py         # Database rebuild script
├── alembic.ini                 # Alembic configuration file
└── alembic/                    # Alembic migrations directory
    ├── env.py                  # Alembic environment configuration
    └── versions/               # Migration scripts directory
```

## Technology Stack

- **ORM**: SQLAlchemy 2.0+
- **Database**: PostgreSQL
- **Integration Method**: 
  - Automatic integration (task loading, training start)
  - Hooks/Wrappers (rollout, step, eval)

## Database Configuration

Configure via environment variables:

```bash
# Method 1: Use DATABASE_URL
export DATABASE_URL=postgresql://user:password@host:port/dbname

# Method 2: Use individual POSTGRES_* variables
export POSTGRES_HOST=127.0.0.1
export POSTGRES_PORT=5432
export POSTGRES_DB=training_db
export POSTGRES_USER=training_user
export POSTGRES_PASSWORD=training_password
```

## Notes

1. **Database Session Management**: Use the provided context manager and global context
2. **Error Handling**: Database operation failures will not interrupt training, only log warnings
3. **Performance**: Database operations are asynchronous and have minimal impact on training performance
4. **Concurrency**: PostgreSQL fully supports concurrent access and is suitable for production environments

## Database Migrations

The project uses Alembic for database migration management. Migrations are automatically run during database initialization.

### Rebuild Database

If you need to rebuild the database (drop all tables and recreate):

```bash
uv run python -m tinker_cookbook.recipes.cua_rl.rebuild_database
```

### Create New Migration

When model structure changes:

```bash
cd tinker_cookbook/recipes/cua_rl
uv run alembic revision --autogenerate -m "describe changes"
uv run alembic upgrade head
```

## Dependencies

Requires SQLAlchemy, PostgreSQL driver, and Alembic:

```bash
pip install sqlalchemy>=2.0.0 psycopg2-binary alembic>=1.13.0
```

Or install complete CUA dependencies:

```bash
pip install -e ".[cua]"
```

## Next Steps

All core features are complete. You can:
1. Test database functionality
2. Add more query functions as needed
3. Create data analysis scripts
