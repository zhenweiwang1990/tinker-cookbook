### genv_local quickstart (Umetrip tasks)

This integrates `genv-umetrip/tasks/**` into `cua_rl` without GBox by running genv's `LocalEnv` (docker backend + Android emulator).

#### 1) Prerequisites
- **Docker**: running and able to pull `ghcr.io/babelcloud/genv-umetrip:latest` (or your custom image via `GENV_DOCKER_IMAGE`).
- **Android emulator tooling**: `sdkmanager`, `avdmanager`, `emulator`, `adb` available in PATH (genv will manage/reuse AVD).
- **genv python package** installed in the same interpreter you use to run cua_rl:

```bash
pip install -e "/Users/zhenwei/workspace/genv-umetrip/rl"
```

#### 2) Task source config
Use `source_type="genv_umetrip"` and point `tasks_dir` to your dataset:

```json
{
  "source_type": "genv_umetrip",
  "tasks_dir": "/Users/zhenwei/workspace/genv-umetrip/tasks",
  "split_type": "train",
  "train_ratio": 0.8,
  "seed": 0,
  "limit": 5
}
```

Notes:
- The database `task.task_id` uses genv `meta.id` (e.g. `task-001`) so GraphQL checks and DB seeding match genv conventions.
- `group_size > 1` is supported, but **rollouts are executed serially** within a group and **globally serialized** to avoid multiple emulators/backends fighting each other.

#### 3) Run training
Example (minimal):

```bash
export DATABASE_URL="postgresql://training_user:training_password@127.0.0.1:5433/training_db"
export TINKER_API_KEY="..."  # required for on-policy RL sampling

python3 -m tinker_cookbook.recipes.cua_rl.core.train \
  --env-mode genv_local \
  --tasks '{"source_type":"genv_umetrip","tasks_dir":"/Users/zhenwei/workspace/genv-umetrip/tasks","split_type":"train","seed":0,"limit":5}' \
  --group-size 2 \
  --groups-per-batch 1 \
  --max-turns 20
```

#### 4) Acceptance checks (DB + monitor)
- **DB**: you should see `task` rows for `task-001` etc, and new `rollout/turn/action/obs/validation` rows.
- **Screenshots**: are written under `training-monitor/public/screenshots/rollout_<uuid>/turn_<turn_id>/obs_<id>.png`.
- **Monitor**: the rollout page should show:
  - screenshot_before / screenshot_after images
  - `checks` JSON observation
  - `validation` row with `details_json.final_result`

If you see `reward=0` always, check that the task has non-empty `evaluation.checks` and the backend is reachable by the emulator (we rewrite app endpoint to `http://10.0.2.2:<port>` automatically).

