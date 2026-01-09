# Benchmark Task Filtering

`benchmark.sh` 现在支持通过类别或任务名称来过滤要运行的任务。

## 新增参数

### `--category CATEGORY`
按任务类别过滤，支持的类别包括：
- `demo` - Android 系统设置任务（32个任务）
- `airbnb` - Airbnb 应用任务（32个任务）
- `instagram` - Instagram 应用任务

### `--task-names NAMES` 或 `--tasks NAMES`
按任务名称过滤，可以指定一个或多个任务（逗号分隔）。

任务名称格式为目录名，例如：
- `12_enable_battery_saver`
- `06_min_brightness`
- `01_i_plan_to_go_to_united`

## 使用示例

### 1. 只运行 demo 类别的任务
```bash
./benchmark.sh --category demo
```

### 2. 只运行 airbnb 类别的任务
```bash
./benchmark.sh --category airbnb
```

### 3. 运行特定的一个任务
```bash
./benchmark.sh --task-names '12_enable_battery_saver'
```

### 4. 运行特定的多个任务
```bash
./benchmark.sh --task-names '12_enable_battery_saver,06_min_brightness,15_download_gbox_logo'
```

### 5. 组合使用：只运行 demo 类别中的训练集任务
```bash
./benchmark.sh --category demo --eval-split train
```

### 6. 组合使用：在特定类别中运行特定任务
```bash
./benchmark.sh --category demo --task-names '12_enable_battery_saver,06_min_brightness'
```

## 任务发现

### 查看所有 demo 任务
```bash
ls tinker_cookbook/recipes/cua_rl/tasks/demo/
```

### 查看所有 airbnb 任务
```bash
ls tinker_cookbook/recipes/cua_rl/tasks/airbnb/
```

## 完整示例

### 在 vLLM 上测试特定的 demo 任务
```bash
./benchmark.sh \
  --provider vllm \
  --provider-base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --category demo \
  --task-names '12_enable_battery_saver,06_min_brightness' \
  --max-turns 10 \
  --name "demo_battery_brightness_test"
```

### 快速验证几个任务
```bash
./benchmark.sh \
  --task-names '12_enable_battery_saver,06_min_brightness,15_download_gbox_logo' \
  --max-turns 5 \
  --max-concurrent 3
```

## 注意事项

1. **任务名称必须精确匹配**：任务名称区分大小写，必须与目录名完全一致。

2. **类别与任务名称可以组合**：先按类别过滤，再按任务名称过滤。

3. **train/eval 分割**：
   - 默认使用 `eval` 分割（约 20% 的任务）
   - 可以通过 `--eval-split train` 使用训练集任务（约 80%）
   - 使用 `--train-ratio 0` 表示使用所有任务（不分割）

4. **查看帮助**：运行 `./benchmark.sh --help` 查看所有可用参数。

## 技术实现

过滤功能通过以下方式实现：

1. **Category 过滤**：基于任务所在的目录路径（demo/airbnb/instagram）识别任务类别
2. **Task names 过滤**：基于任务的 `name` 属性匹配任务名称
3. **配置传递**：通过 `TaskSourceConfig` 的 `category` 和 `task_names` 参数传递过滤条件
4. **Python 实现**：核心过滤逻辑在 `tinker_cookbook/recipes/cua_rl/task_loader.py` 中

## 相关文件

- `benchmark.sh` - Shell 脚本入口，解析命令行参数
- `tinker_cookbook/recipes/cua_rl/benchmark.py` - Python benchmark 包装器
- `tinker_cookbook/recipes/cua_rl/task_loader.py` - 任务加载和过滤核心逻辑
- `tinker_cookbook/recipes/cua_rl/tasks/` - 任务定义目录

