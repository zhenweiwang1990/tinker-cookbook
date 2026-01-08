# 坐标模式快速参考卡

## 🎯 选择模式

| 需求 | 推荐配置 | 命令示例 |
|------|---------|---------|
| **Android + 外部坐标模型** | GBox 模式 | `--coordinate-mode gbox --box-type android` |
| **Android + VLM 直出坐标** | Direct 模式 | `--coordinate-mode direct --box-type android` |
| **Android + Qwen3-VL** | Direct + 缩放 | `--coordinate-mode direct --coordinate-scale true` |
| **PC/Linux + 外部坐标模型** | GBox 模式 | `--coordinate-mode gbox --box-type linux` |
| **PC/Linux + VLM 直出坐标** | Direct 模式 | `--coordinate-mode direct --box-type linux` |
| **PC/Linux + Qwen3-VL** | Direct + 缩放 | `--coordinate-mode direct --coordinate-scale true --box-type linux` |

## 📋 参数速查

### 必选参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--coordinate-mode` | `gbox` \| `direct` | 坐标生成模式 |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--box-type` | `android` | GBox 环境类型：`android`, `linux`, `windows` |
| `--coordinate-scale` | auto | 是否启用坐标缩放（Android Direct: false, PC Direct: true） |
| `--x-scale-ratio` | `auto` | X 轴缩放比例（默认 `screen_width/1000`） |
| `--y-scale-ratio` | `auto` | Y 轴缩放比例（默认 `screen_height/1000`） |

## 🚀 常用命令

### Android GBox 模式
```bash
./benchmark.sh \
  --coordinate-mode gbox \
  --model-path tinker://path/to/weights
```

### Android Direct 模式（GPT-4V/Claude）
```bash
./benchmark.sh \
  --coordinate-mode direct \
  --coordinate-scale false \
  --model-path tinker://path/to/weights
```

### Android Direct 模式（Qwen3-VL）
```bash
./benchmark.sh \
  --coordinate-mode direct \
  --coordinate-scale true \
  --model-path tinker://path/to/weights
```

### PC/Linux GBox 模式
```bash
./benchmark.sh \
  --coordinate-mode gbox \
  --box-type linux \
  --model-path tinker://path/to/weights
```

### PC/Linux Direct 模式（Qwen3-VL）
```bash
./benchmark.sh \
  --coordinate-mode direct \
  --coordinate-scale true \
  --box-type linux \
  --model-path tinker://path/to/weights
```

## 📊 日志格式对比

### GBox 模式
```
Action: tap | target=login button
  ↳ Coords: (540, 1200) | coord_time=0.123s | exec_time=1.0s | total=1.123s
```

### Direct 模式（无缩放）
```
Action: tap | target=login button
  ↳ Coords: (540, 1200) | coord_time=0.000s | exec_time=1.0s | total=1.0s
```

### Direct 模式（启用缩放）
```
Action: tap | target=login button
  ↳ Coords: (500, 790) → (540, 1200) | coord_time=0.000s | exec_time=1.0s | total=1.0s
```
- 🔵 青色 `(500, 790)`: VLM 输出（基于 1000×1000）
- 🟡 黄色 `(540, 1200)`: 缩放后执行坐标

## 🎨 操作对比

### Android 操作
- `tap` - 触摸点击
- `swipe` - 滑动
- `button_press` - 设备按键（back, home, menu）
- `long_press` - 长按

### PC 操作
- `click` - 鼠标点击（left/right/double）
- `key_press` - 键盘按键（Control+C, Enter）
- `drag` - 鼠标拖拽
- `scroll` - 滚动

## 🔧 坐标格式

### GBox 模式（VLM 输出）
```json
{
  "name": "action",
  "args": {
    "action_type": "tap",
    "target": {
      "element": "login button",
      "label": "Sign In",
      "location": "center"
    }
  }
}
```

### Direct 模式（VLM 输出）
```json
{
  "name": "action",
  "args": {
    "action_type": "tap",
    "target": {
      "element": "login button",
      "coordinates": [540, 1200]
    }
  }
}
```

## ⚙️ 提示语文件映射

| box_type | coordinate_mode | 提示语文件 |
|----------|-----------------|-----------|
| `android` | `gbox` | `android-system-prompt-gbox.txt` |
| `android` | `direct` | `android-system-prompt-direct.txt` |
| `linux` | `gbox` | `pc-system-prompt-gbox.txt` |
| `linux` | `direct` | `pc-system-prompt-direct.txt` |
| `windows` | `gbox` | `pc-system-prompt-gbox.txt` |
| `windows` | `direct` | `pc-system-prompt-direct.txt` |

## 🧮 缩放计算示例

### 场景：720×1520 Android 屏幕 + Qwen3-VL

**配置**:
```bash
--coordinate-scale true
```

**自动计算**:
```
x_scale_ratio = 720 / 1000 = 0.720
y_scale_ratio = 1520 / 1000 = 1.520
```

**坐标转换**:
```
VLM 输出: [809, 742] (基于 1000×1000)
缩放后:   [582, 1128] (实际执行)

计算:
  x = 809 × 0.720 = 582
  y = 742 × 1.520 = 1128
```

### 场景：1920×1080 PC 屏幕 + Qwen3-VL

**配置**:
```bash
--coordinate-scale true --box-type linux
```

**自动计算**:
```
x_scale_ratio = 1920 / 1000 = 1.920
y_scale_ratio = 1080 / 1000 = 1.080
```

**坐标转换**:
```
VLM 输出: [78, 28] (基于 1000×1000)
缩放后:   [150, 30] (实际执行)

计算:
  x = 78 × 1.920 = 150
  y = 28 × 1.080 = 30
```

## 📖 文档索引

| 文档 | 内容 |
|------|------|
| `COORDINATE_MODE_USAGE.md` | Android 坐标模式详细指南 |
| `PC_COORDINATE_MODE_SUPPORT.md` | PC 坐标模式详细指南 |
| `PC_COORDINATE_IMPLEMENTATION_SUMMARY.md` | PC 实现技术总结 |
| `COORDINATE_MODE_QUICK_REFERENCE.md` | 本文档 - 快速参考 |

## 🐛 故障排除

### 问题：坐标超出屏幕范围

**日志**:
```
[WARNING] Scaled coordinates out of screen bounds (screen: 720x1520): 
original=(1050, 580) → scaled=(756, 882)
```

**原因**: VLM 输出坐标超过 1000（可能没有正确理解归一化范围）

**解决**: 
1. 检查提示语是否正确加载
2. 调整 VLM 训练数据或 temperature

### 问题：`coordinate_scale=False` 但日志显示无缩放

**日志**:
```
[INFO] [DirectCoordinateGenerator] Initialized without coordinate scaling: 
screen=1080x2400, center=(540, 1200)
```

**原因**: CLI 参数传递问题或自动检测逻辑错误

**解决**:
1. 检查启动命令是否包含 `--coordinate-scale true`
2. 对于 Direct 模式，确认传递了正确的参数

### 问题：PC 提示语未加载

**症状**: PC 环境下使用了 Android 操作（tap 而不是 click）

**解决**: 确保设置了 `--box-type linux` 或 `windows`

## 💡 最佳实践

1. **明确指定 box_type**: 避免依赖默认值
2. **根据 VLM 选择缩放策略**: Qwen3-VL 用缩放，GPT-4V 不用
3. **查看日志验证**: 检查坐标转换是否符合预期
4. **逐步调试**: 先用 GBox 验证环境，再切换到 Direct
5. **保存日志**: 便于问题复现和调试

## ⚡️ 性能对比

| 模式 | 坐标生成时间 | 准确性 | 灵活性 |
|------|------------|--------|--------|
| **GBox** | ~100-200ms | 高（外部模型） | 低（依赖外部 API） |
| **Direct** | ~0ms | 中（依赖 VLM） | 高（VLM 端到端） |
| **Direct + Scale** | ~0ms | 中 | 高 |

## 🎓 学习路径

1. **入门**: 使用 GBox 模式熟悉系统
2. **进阶**: 切换到 Direct 模式（无缩放）
3. **高级**: 使用 Direct 模式 + 坐标缩放
4. **专家**: 根据不同 VLM 调优缩放参数

---

**最后更新**: 2026-01-08  
**版本**: 1.0

