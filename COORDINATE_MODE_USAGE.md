# Coordinate Mode Usage Guide

## Overview

The agent now supports two coordinate generation modes:

1. **GBox Mode** (default): Uses an external GBox model (gbox-handy-1) to convert natural language descriptions to coordinates
2. **Direct Mode** (new): VLM directly generates coordinates in its output

In Direct mode, there are two coordinate output methods:
- **No Scaling** (coordinate_scale=False): Model outputs actual screen pixel coordinates, system prompt includes screen dimensions
- **Scaling Enabled** (coordinate_scale=True, **default for Direct mode**): Model outputs normalized coordinates (e.g., 0-1000), system automatically scales to actual screen pixels

## Usage

### Training Scripts

When using `train.sh`, add the `--coordinate-mode` and `--coordinate-scale` parameters:

```bash
# GBox mode (default, no scaling)
./train.sh --coordinate-mode gbox

# Direct mode - scaling enabled (default, suitable for models like Qwen3-VL)
./train.sh --coordinate-mode direct

# Direct mode - explicitly enable scaling
./train.sh --coordinate-mode direct --coordinate-scale true

# Direct mode - no scaling (model outputs actual pixel coordinates)
./train.sh --coordinate-mode direct --coordinate-scale false
```

Complete example:

```bash
# Using Qwen3-VL with coordinate scaling (recommended)
./train.sh \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --coordinate-mode direct \
  --coordinate-scale true \
  --group-size 4 \
  --groups-per-batch 2 \
  --lr 1e-5 \
  --max-turns 20
```

### Evaluation Scripts

When using `benchmark.sh`, add the `--coordinate-mode` and `--coordinate-scale` parameters:

```bash
# GBox mode (default)
./benchmark.sh --coordinate-mode gbox

# Direct mode - scaling enabled (default)
./benchmark.sh --coordinate-mode direct

# Direct mode - no scaling
./benchmark.sh --coordinate-mode direct --coordinate-scale false
```

Complete example:

```bash
./benchmark.sh \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --model-path tinker://experiment-id:train:0/sampler_weights/000080 \
  --coordinate-mode direct \
  --coordinate-scale true \
  --max-turns 20
```

## Mode Comparison

| Feature | GBox Mode | Direct Mode (No Scaling) | Direct Mode (Scaling) |
|---------|-----------|-------------------------|----------------------|
| **Coordinate Generation** | External GBox model | VLM outputs pixel coordinates directly | VLM outputs normalized coordinates |
| **Latency** | ~1-2s/action | ~0s (no extra calls) | ~0s (no extra calls) |
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Cost** | GBox + Tinker API | Tinker API only | Tinker API only |
| **Screen Size Dependency** | None | Strong (included in prompt) | Weak (only for scaling) |
| **Cross-Resolution** | ✅ Good | ❌ Poor (fixed size) | ✅ Good (auto-scaling) |
| **Use Cases** | General tasks | Fixed resolution devices | Multi-resolution devices / Models like Qwen3-VL |

## How It Works

### GBox Mode

```
User: "Click the login button"
  ↓
VLM: {"name": "action", "args": {"action_type": "tap", "target": {"element": "login button"}}}
  ↓
GBox API: Call gbox-handy-1 model
  ↓
Coordinates: {"x": 540, "y": 1800}
  ↓
Execute: box.action.tap(540, 1800)
```

### Direct Mode (No Scaling)

```
User: "Click the login button"
  ↓
Task start:
  1. Create initial system prompt (screen size placeholder: [TO_BE_DETERMINED])
  ↓
First Turn:
  1. Take screenshot
  2. Extract screen dimensions (e.g., 1080x2400)
  3. Regenerate system prompt (replace placeholder with actual value: 1080x2400)
  4. Update messages[0] (system message)
  5. Update DirectCoordinateGenerator screen size
  ↓
VLM sees: "Screen: 1080x2400 pixels"
VLM: {"name": "action", "args": {"action_type": "tap", "target": {"element": "login button", "coordinates": [540, 1800]}}}
  ↓
Pydantic parsing: TargetElement(element="login button", coordinates=[540, 1800])
  ↓  validator auto-extracts: x=540, y=1800
Extract coordinates: {"x": 540, "y": 1800}
  ↓
Execute: box.action.tap(540, 1800)
```

**Key Features**:
- ✅ Two-stage prompt generation (initial placeholder → first turn updates with actual value)
- ✅ VLM knows actual screen dimensions, outputs pixel coordinates directly
- ✅ Coordinate format: `coordinates: [x, y]` (more intuitive)
- ✅ Pydantic validator automatically extracts x, y from array
- ✅ Backward compatible: still supports separate `x`, `y` fields

### Direct Mode (Scaling Enabled) 🆕

```
User: "Click the login button"
  ↓
Task start:
  1. Create initial system prompt (no screen dimensions, because scaling is enabled)
  ↓
First Turn:
  1. Take screenshot
  2. Extract screen dimensions (e.g., 1080x2400)
  3. Calculate scaling ratios: x_scale = 1080/1000 = 1.08, y_scale = 2400/1000 = 2.40
  4. Update DirectCoordinateGenerator screen size and scaling ratios
  5. System prompt unchanged (no screen dimensions)
  ↓
VLM sees: No screen size information
VLM (e.g., Qwen3-VL): Internal image processing to 1000x1000
VLM: {"name": "action", "args": {"action_type": "tap", "target": {"element": "login button", "coordinates": [500, 750]}}}
  ↓
Pydantic parsing: TargetElement(element="login button", coordinates=[500, 750])
  ↓  validator auto-extracts: x=500, y=750
Extract coordinates: {"x": 500, "y": 750}
  ↓
DirectCoordinateGenerator applies scaling:
  x_scaled = 500 * 1.08 = 540
  y_scaled = 750 * 2.40 = 1800
  ↓
Execute: box.action.tap(540, 1800)
```

**Key Features**:
- ✅ Model outputs normalized coordinates (e.g., 0-1000), no need to know actual screen dimensions
- ✅ Automatic scaling to actual screen pixels
- ✅ Suitable for models with fixed internal image resolution like Qwen3-VL
- ✅ Supports arbitrary resolution devices (phones, tablets, desktops)
- ✅ Scaling ratios can be customized (default: screen_width/1000, screen_height/1000)
- ✅ Dynamically updates prompt on first turn
- ⚠️ **If screen dimensions cannot be extracted, will immediately error and exit** (ensures data accuracy)

## Prompt Differences

### GBox Mode Prompt

```
When specifying a target, provide an object with the following fields:
- **element** (required): What the element is (e.g., "login button")
- **label** (optional): Text shown on the element
- **location** (optional): Where on screen (e.g., "top right")

Example:
<tool_call>
{"name": "action", "args": {"action_type": "tap", "target": {"element": "login button"}}}
</tool_call>
```

### Direct Mode Prompt

```
Screen Dimensions: (Dynamically extracted from first screenshot)
- Width: 1080 pixels (or actual screen width)
- Height: 2400 pixels (or actual screen height)

When specifying a target, you MUST provide:
- **element** (required): What the element is (e.g., "login button")
- **coordinates** (required): Array [x, y] in pixels

Example:
<tool_call>
{"name": "action", "args": {"action_type": "tap", "target": {"element": "login button", "coordinates": [540, 1800]}}}
</tool_call>
```

**Note**: 
- Coordinate format is `coordinates: [x, y]` (array), more intuitive
- System also supports separate `x` and `y` fields (backward compatible)
- Screen dimensions are automatically extracted from the screenshot on the first turn, no manual configuration needed.

## Performance Recommendations

### When to Use GBox Mode

- ✅ Initial evaluation/baseline testing
- ✅ Complex UI or small targets
- ✅ Tasks requiring extremely high accuracy
- ✅ Scenarios where latency is not a concern

### When to Use Direct Mode

- ✅ Fine-tuned models
- ✅ Need to reduce API costs
- ✅ Latency-sensitive applications
- ✅ Simple and clear UI layouts
- ✅ Large target elements (e.g., large buttons, cards)

## Troubleshooting

### Common Issues in Direct Mode

1. **Model doesn't output coordinates**
   - Check if prompt is correctly loaded (`android-system-prompt-direct.txt`)
   - Verify model understands coordinate requirements
   - Check logs for warnings

2. **Inaccurate coordinates**
   - Check if model outputs reasonable coordinates (within screen bounds)
   - Review first turn logs to confirm screen dimensions were correctly detected
   - Consider using GBox mode as reference
   - Check if more training data is needed

3. **Screen dimension extraction failure**
   - **Symptom**: Rollout errors out on first turn
   - **Cause**: Cannot extract screen dimensions from screenshot
   - **Solution**: 
     - Check if screenshot format is correct (should be base64 data URI)
     - Check if GBox API returns screenshot normally
     - Switch to GBox mode if issue persists

4. **Parsing errors**
   - Check `DirectCoordinateGenerator` logs
   - Verify model output JSON format
   - Ensure target field contains x, y

### Log Examples

**GBox Mode**:
```
[Agent Init] Using GBox coordinate mode (external model)
[GBox Coordinate] Generating coordinates: action_type=tap, target=login button
[GBox Coordinate] Coordinate generation successful
```

**Direct Mode**:
```
[Agent Init] Using Direct coordinate mode (VLM outputs coordinates)
[Agent Init] System prompt created with placeholder screen dimensions
[Turn 1] Taking screenshot...
[Turn 1] Detected screen dimensions: 1080x2400
[Turn 1] Updated system prompt with screen dimensions: 1080x2400
[Direct Coordinate] Extracted tap coordinates: (540, 1800)
```

**Screen dimension extraction error**:
```
[Agent Init] System prompt created with placeholder screen dimensions
[Turn 1] Taking screenshot...
ERROR: Failed to extract screen dimensions from screenshot: Invalid image format
RuntimeError: In Direct coordinate mode, screen dimensions are required.
Rollout terminated.
```

## Code Integration

### Python API

If using the Python API directly:

```python
from tinker_cookbook.recipes.cua_rl.agent.tinker_cua_agent import TinkerCuaAgent

# GBox mode
agent = TinkerCuaAgent(
    gbox_api_key=gbox_api_key,
    tinker_api_key=tinker_api_key,
    tinker_model_path=model_path,
    coordinate_mode="gbox"  # default
)

# Direct mode
agent = TinkerCuaAgent(
    gbox_api_key=gbox_api_key,
    tinker_api_key=tinker_api_key,
    tinker_model_path=model_path,
    coordinate_mode="direct"  # new
)
```

## Next Steps

1. **Experimental Comparison**: Run both modes on the same tasks to compare performance
2. **Hybrid Strategy**: Dynamically switch modes based on task complexity
3. **Fine-tuning**: Train Direct mode using successful trajectories from GBox mode
4. **Coordinate Correction**: Add lightweight post-processing to improve Direct mode accuracy

## More Information

For detailed implementation, refer to:
- Refactoring plan: `COORDINATE_MODE_REFACTOR_PLAN.md`
- Code implementation: `tinker_cookbook/recipes/cua_rl/gbox/`
- Prompt templates: `tinker_cookbook/recipes/cua_rl/prompts/`

