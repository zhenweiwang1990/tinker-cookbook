# Benchmark Task Filtering - Quick Reference

## 快速使用

```bash
# 只跑 demo 类别
./benchmark.sh --category demo

# 只跑特定任务
./benchmark.sh --task-names '12_enable_battery_saver,06_min_brightness'

# 组合使用
./benchmark.sh --category airbnb --eval-split train

# 查看帮助
./benchmark.sh --help
```

## 可用类别

- `demo` - Android 系统任务（32个）
- `airbnb` - Airbnb 应用任务（32个）
- `instagram` - Instagram 应用任务

## 查看任务列表

```bash
# 查看 demo 任务
ls tinker_cookbook/recipes/cua_rl/tasks/demo/

# 查看 airbnb 任务
ls tinker_cookbook/recipes/cua_rl/tasks/airbnb/

# 查看 instagram 任务
ls tinker_cookbook/recipes/cua_rl/tasks/instagram/
```

## 任务名称示例

Demo 任务：
- `01_open_settings`
- `06_min_brightness`
- `12_enable_battery_saver`
- `15_download_gbox_logo`

Airbnb 任务：
- `01_i_plan_to_go_to_united`
- `02_help_me_ask_the_host_of`
- `03_i_want_to_experience_unique_rooms`

## 完整使用示例

```bash
# 测试特定 demo 任务
./benchmark.sh \
  --category demo \
  --task-names '12_enable_battery_saver,06_min_brightness' \
  --max-turns 10

# 在 vLLM 上跑 airbnb 训练任务
./benchmark.sh \
  --provider vllm \
  --provider-base-url http://localhost:8000/v1 \
  --category airbnb \
  --eval-split train

# 快速验证几个任务
./benchmark.sh \
  --task-names '01_open_settings,06_min_brightness,12_enable_battery_saver' \
  --max-concurrent 3 \
  --max-turns 5
```

详细文档请查看 `BENCHMARK_TASK_FILTERING.md`

