# Multi-Provider Support for CUA Benchmark

This guide explains how to use different LLM providers (Tinker, vLLM, OpenRouter, OpenAI) with the CUA benchmark system.

## Overview

The CUA benchmark system now supports multiple inference providers while preserving all the advanced features of `tinker-cookbook`:

- ✅ **Prompt-based tool calling** (tool examples in system prompt)
- ✅ **Advanced coordinate handling** (`gbox` and `direct` modes)
- ✅ **Detailed database recording** (via `RolloutRecorder`)
- ✅ **Rich rollout logging** (via `RolloutLogger`)
- ✅ **Consistent evaluation** (same prompts and logic across all providers)

## Supported Providers

| Provider | Use Case | Multimodal | Logprobs |
|----------|----------|------------|----------|
| `tinker` | Training and evaluation with Tinker checkpoints | ✅ | ✅ |
| `vllm` | Local deployment for fast evaluation | ✅ | ❌ |
| `openrouter` | Cloud API access to many models | ✅ | ❌ |
| `openai` | GPT models (gpt-4-vision, etc.) | ✅ | ❌ |

## Quick Start

### 1. Tinker Provider (Default)

```bash
./benchmark.sh \
  --provider tinker \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --model-path "tinker://experiment-id:train:0/sampler_weights/000080"
```

Environment variables:
- `TINKER_API_KEY`: Required
- `GBOX_API_KEY`: Required

### 2. vLLM Provider (Local)

First, start vLLM server:

```bash
# Install vLLM
pip install vllm

# Start vLLM server with vision model
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --port 8000 \
  --max-model-len 4096
```

Then run benchmark:

```bash
./benchmark.sh \
  --provider vllm \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --provider-base-url http://localhost:8000/v1
```

Environment variables:
- `GBOX_API_KEY`: Required

### 3. OpenRouter Provider

```bash
./benchmark.sh \
  --provider openrouter \
  --model qwen/qwen3-vl-30b-a3b-instruct \
  --provider-api-key "$OPENROUTER_API_KEY"
```

Environment variables:
- `GBOX_API_KEY`: Required
- `OPENROUTER_API_KEY`: Required (or pass via `--provider-api-key`)

Model names for OpenRouter:
- `qwen/qwen3-vl-30b-a3b-instruct`
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4-vision-preview`

See [OpenRouter Models](https://openrouter.ai/models) for full list.

###4. OpenAI Provider

```bash
./benchmark.sh \
  --provider openai \
  --model gpt-4-vision-preview \
  --provider-api-key "$OPENAI_API_KEY"
```

Environment variables:
- `GBOX_API_KEY`: Required
- `OPENAI_API_KEY`: Required (or pass via `--provider-api-key`)

## Command-Line Options

### Provider Options

```bash
--provider PROVIDER              # tinker, vllm, openrouter, openai (default: tinker)
--provider-base-url URL          # API base URL (for vLLM, default uses provider default)
--provider-api-key KEY           # API key (for OpenRouter, OpenAI, auto-detects from env)
```

### Model Options

```bash
--model MODEL_NAME               # Model name or identifier
--model-path PATH                # For Tinker: checkpoint path (tinker://...)
```

### Dataset Options

```bash
--eval-source SOURCE             # task_adapter, demo_eval, etc. (default: task_adapter)
--eval-split SPLIT               # train, eval (default: eval)
--seed SEED                      # Random seed (default: 42)
```

### Execution Options

```bash
--max-turns TURNS                # Max turns per task (default: 20)
--temperature TEMP               # Temperature (default: 1.0)
--max-concurrent N               # Max concurrent rollouts (default: 8)
--coordinate-mode MODE           # gbox or direct (default: direct)
```

## Architecture

### Key Components

1. **BaseInferenceClient** (`base_inference_client.py`)
   - Abstract interface: `generate_text(messages, max_tokens, temperature) -> str`
   - All providers implement this simple interface

2. **TinkerInferenceClient** (`tinker_inference_client.py`)
   - Wraps Tinker's `SamplingClient`
   - Returns raw tokens from `sample_async()`
   - Decodes tokens to text

3. **HTTPInferenceClient** (`http_inference_client.py`)
   - Generic OpenAI-compatible HTTP client
   - Used for vLLM, OpenRouter, OpenAI
   - **Does NOT pass `tools` parameter** (preserves tinker-cookbook's prompt-based approach)

4. **TinkerCuaAgent** (`tinker_cua_agent.py`)
   - Accepts `inference_client` during initialization
   - Calls `inference_client.generate_text()` to get raw text
   - Tokenizes text and uses `renderer.parse_response()` to extract tool calls
   - **All existing logic preserved**: prompts, tool parsing, coordinate handling, logging

### Data Flow

```
User Request
    ↓
benchmark.sh
    ↓
benchmark.py (BenchmarkConfig)
    ↓
train.py (CLIConfig)
    ↓
CUAEnv.__init__(provider, provider_base_url, provider_api_key)
    ↓
CUAEnv.run_rollout_with_tinker_model()
    ↓
TinkerCuaAgent.__init__()
    ├→ inference_client_factory.create_inference_client()
    │   ├→ TinkerInferenceClient (for Tinker)
    │   └→ HTTPInferenceClient (for vLLM/OpenRouter/OpenAI)
    ↓
TinkerCuaAgent.run_task()
    ↓
TinkerCuaAgent._sample_with_model()
    ├→ inference_client.generate_text() → raw text
    ├→ tokenizer.encode(text) → tokens
    └→ renderer.parse_response(tokens) → tool calls
    ↓
Tool execution (perform_action, etc.)
    ↓
Database recording (RolloutRecorder)
```

## Key Design Decisions

### 1. Prompt-Based Tool Calling

**All providers use the same approach:**
- Tools are defined in the system prompt with examples
- Model generates text like `<tool_call>{"name": "tap", "arguments": {...}}</tool_call>`
- `renderer.parse_response()` extracts tool calls from text

**We do NOT use native function calling:**
- No `tools` parameter passed to API
- This ensures consistent behavior across all providers
- Preserves tinker-cookbook's existing tool parsing logic

### 2. Minimal Interface

**`BaseInferenceClient` has only one method:**
```python
async def generate_text(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    pass
```

**Why so simple?**
- All providers generate text from messages
- `TinkerCuaAgent` handles everything else (tokenization, parsing, tool extraction)
- Easy to add new providers

### 3. Backward Compatibility

**Legacy Tinker-only code still works:**
```python
# Old way (still works)
TinkerCuaAgent(
    gbox_api_key=key,
    tinker_api_key=api_key,
    tinker_model_path=checkpoint,
)

# New way (recommended)
TinkerCuaAgent(
    gbox_api_key=key,
    provider="tinker",
    provider_model_name=model,
    provider_api_key=api_key,
)
```

## Troubleshooting

### vLLM Connection Error

```
RuntimeError: Cannot connect to vllm at http://localhost:8000/v1
```

**Solution:** Ensure vLLM server is running:
```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8000
```

### OpenRouter Model Not Found

```
RuntimeError: openrouter API error: 404: No endpoints found for model
```

**Solution:** Check model name at https://openrouter.ai/models. Model names are lowercase with `/` separator.

### Tinker Checkpoint Loading Failed

```
ValueError: Tinker provider requires model_path parameter
```

**Solution:** For Tinker provider, you must specify `--model-path` with a `tinker://...` checkpoint path.

### Tool Parsing Errors

If tool calls are not being detected:
1. Check system prompt includes tool examples
2. Verify model supports vision and long context
3. Try lower temperature (e.g., 0.7)
4. Check coordinate mode matches your model's training

## Performance Comparison

| Provider | Speed | Cost | Training Support |
|----------|-------|------|------------------|
| Tinker | Fast | Low | ✅ Full (logprobs) |
| vLLM | Very Fast | Free (local) | ❌ Eval only |
| OpenRouter | Medium | Variable | ❌ Eval only |
| OpenAI | Medium | High | ❌ Eval only |

**Training requires Tinker provider** (only Tinker provides logprobs needed for RL training).

## Next Steps

1. **Run your first benchmark:**
   ```bash
   ./benchmark.sh --help
   ```

2. **View results:**
   - Database: Query PostgreSQL `training_db`
   - Logs: Check `benchmark_logs/` directory
   - Training Monitor: Start web UI

3. **Add more providers:**
   - Edit `inference_client_factory.py` to add provider configs
   - Implement `BaseInferenceClient` for custom backends

## Reference

- API Types: `docs/api-reference/types.md`
- Renderers: `docs/rendering.mdx`
- Coordinate Modes: `COORDINATE_MODE_USAGE.md`
- Training: `docs/rl/rl-basic.mdx`

