#!/bin/bash
set -e

# Load .env from project root if present (so MODEL_NAME and others work without manual export).
# Important: do NOT override environment variables already provided by the caller.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
load_env_defaults() {
    local env_file="$1"
    local line key val
    while IFS= read -r line || [ -n "$line" ]; do
        # Trim leading whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        # Skip comments / empty lines
        [[ -z "$line" || "$line" == \#* ]] && continue
        # Allow optional "export "
        line="${line#export }"
        # Parse KEY=VALUE (very common .env format)
        if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            val="${BASH_REMATCH[2]}"
            # Only set if not already provided by the environment
            if [[ -z "${!key+x}" ]]; then
                # Strip surrounding single/double quotes if present
                if [[ "$val" =~ ^\"(.*)\"$ ]]; then
                    val="${BASH_REMATCH[1]}"
                elif [[ "$val" =~ ^\'(.*)\'$ ]]; then
                    val="${BASH_REMATCH[1]}"
                fi
                export "$key=$val"
            fi
        fi
    done < "$env_file"
}
if [ -f "$PROJECT_ROOT/.env" ]; then
    load_env_defaults "$PROJECT_ROOT/.env"
    echo "✓ Loaded .env defaults from $PROJECT_ROOT"
fi

echo "=========================================="
echo "vLLM Inference Server - Base VLM Model"
echo "=========================================="
echo ""

# Default values - Using Qwen3-VL for CUA
# NOTE: This script now only respects MODEL_NAME for selecting the base model.
#       The previous BASE_MODEL env var is ignored for simplicity.
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3-VL-30B-A3B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_DEVICES="${GPU_DEVICES:-all}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION_WAS_SET=true
if [[ -z "${GPU_MEMORY_UTILIZATION+x}" || -z "${GPU_MEMORY_UTILIZATION}" ]]; then
    GPU_MEMORY_UTILIZATION_WAS_SET=false
    GPU_MEMORY_UTILIZATION="0.85"
fi
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-cua-server}"
DTYPE="${DTYPE:-auto}"
SERVED_MODEL_NAME_WAS_SET=true
if [[ -z "${SERVED_MODEL_NAME+x}" || -z "${SERVED_MODEL_NAME}" ]]; then
    SERVED_MODEL_NAME_WAS_SET=false
    SERVED_MODEL_NAME=""
fi
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# Function to detect GPU type and select appropriate vLLM image
detect_gpu_type() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "unknown"
        return 1
    fi
    
    # Use timeout to prevent hanging (5 seconds timeout)
    local gpu_name
    if command -v timeout &> /dev/null; then
        gpu_name=$(timeout 5 nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr '[:upper:]' '[:lower:]' || echo "")
    else
        # Fallback if timeout command is not available
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr '[:upper:]' '[:lower:]' || echo "")
    fi
    
    if [[ -z "${gpu_name}" ]]; then
        echo "unknown"
        return 1
    fi
    
    # Remove extra whitespace
    gpu_name=$(echo "${gpu_name}" | xargs)
    
    if echo "${gpu_name}" | grep -qi "h100"; then
        echo "h100"
        return 0
    elif echo "${gpu_name}" | grep -qi "gh200\|grace hopper"; then
        echo "gh200"
        return 0
    elif echo "${gpu_name}" | grep -qi "h200"; then
        echo "h200"
        return 0
    elif echo "${gpu_name}" | grep -qi "gb10"; then
        echo "gb10"
        return 0
    else
        echo "unknown"
        return 1
    fi
}

select_vllm_image() {
    local override_image="${1:-}"
    
    if [[ -n "${override_image}" ]]; then
        echo "${override_image}"
        return 0
    fi
    
    local gpu_type
    gpu_type=$(detect_gpu_type)
    
    case "${gpu_type}" in
        h100|h200)
            echo "vllm/vllm-openai:latest"
            ;;
        gh200)
            echo "rajesh550/gh200-vllm:0.11.1rc2"
            ;;
        gb10)
            # GB10 requires a newer CUDA/toolchain stack (sm_121a support).
            # Use NVIDIA NGC vLLM image.
            echo "nvcr.io/nvidia/vllm:25.12.post1-py3"
            ;;
        *)
            # Default to vllm/vllm-openai:latest for unknown GPUs
            echo "vllm/vllm-openai:latest"
            ;;
    esac
}

# Multi-GPU support: Parse GPU_DEVICES to determine actual GPU count
if command -v nvidia-smi &> /dev/null; then
    TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
    
    # Parse GPU_DEVICES to get actual number of GPUs to use
    if [ "$GPU_DEVICES" = "all" ]; then
        NUM_GPUS_TO_USE=$TOTAL_GPUS
        GPU_LIST=$(seq -s, 0 $((TOTAL_GPUS - 1)))
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+$ ]]; then
        # Single GPU index
        NUM_GPUS_TO_USE=1
        GPU_LIST="$GPU_DEVICES"
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
        # Comma-separated GPU indices (e.g., "0,1,2,3")
        GPU_LIST="$GPU_DEVICES"
        NUM_GPUS_TO_USE=$(echo "$GPU_DEVICES" | tr ',' '\n' | wc -l)
    elif [[ "$GPU_DEVICES" =~ ^[0-9]+-[0-9]+$ ]]; then
        # Range format (e.g., "0-7")
        START_GPU=$(echo "$GPU_DEVICES" | cut -d'-' -f1)
        END_GPU=$(echo "$GPU_DEVICES" | cut -d'-' -f2)
        NUM_GPUS_TO_USE=$((END_GPU - START_GPU + 1))
        GPU_LIST=$(seq -s, $START_GPU $END_GPU)
    else
        echo "Warning: Invalid GPU_DEVICES format: $GPU_DEVICES. Using all GPUs."
        NUM_GPUS_TO_USE=$TOTAL_GPUS
        GPU_LIST=$(seq -s, 0 $((TOTAL_GPUS - 1)))
    fi
    
    # Validate GPU indices
    for gpu in $(echo "$GPU_LIST" | tr ',' ' '); do
        if [ "$gpu" -ge "$TOTAL_GPUS" ] || [ "$gpu" -lt 0 ]; then
            echo "Error: GPU index $gpu is out of range (0-$((TOTAL_GPUS - 1)))"
            exit 1
        fi
    done
else
    TOTAL_GPUS=0
    NUM_GPUS_TO_USE=1
    GPU_LIST="0"
fi

# Auto-detect tensor parallel size if not set
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
    TENSOR_PARALLEL_SIZE=$NUM_GPUS_TO_USE
    echo "Auto-detected $NUM_GPUS_TO_USE GPU(s), setting TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
fi

# Validate tensor parallel size matches GPU count
if [ "$TENSOR_PARALLEL_SIZE" -gt "$NUM_GPUS_TO_USE" ]; then
    echo "Warning: TENSOR_PARALLEL_SIZE ($TENSOR_PARALLEL_SIZE) > available GPUs ($NUM_GPUS_TO_USE)."
    echo "Setting TENSOR_PARALLEL_SIZE to $NUM_GPUS_TO_USE"
    TENSOR_PARALLEL_SIZE=$NUM_GPUS_TO_USE
fi

# Set CUDA_VISIBLE_DEVICES for container
# This ensures vLLM sees the correct GPU indices inside the container
CUDA_VISIBLE_DEVICES="$GPU_LIST"

# Auto-detect vLLM image based on GPU type (can be overridden by VLLM_IMAGE env var)
if [[ -z "${VLLM_IMAGE:-}" ]]; then
    echo "Detecting GPU type to select appropriate vLLM image..."
    gpu_type=$(detect_gpu_type 2>/dev/null || echo "unknown")
    if [[ "${gpu_type}" != "unknown" ]]; then
        echo "Detected GPU type: ${gpu_type}"
    else
        echo "Could not detect GPU type, using default image"
    fi
    VLLM_IMAGE=$(select_vllm_image)
    echo "Selected vLLM image: ${VLLM_IMAGE}"
else
    echo "Using manually specified VLLM_IMAGE: ${VLLM_IMAGE}"
fi

# Heuristic: GB10 + bf16 conv3d can fail on some stacks; prefer fp16 unless user overrides.
if [ "$DTYPE" = "auto" ]; then
    detected_gpu_type=$(detect_gpu_type 2>/dev/null || echo "unknown")
    if [ "$detected_gpu_type" = "gb10" ]; then
        DTYPE="float16"
        echo "Detected GB10 GPU → setting DTYPE=float16 (override with DTYPE=...)"
    fi
fi

# Extra runtime env flags for specific GPU stacks.
EXTRA_ENV_VARS=""
if [ "${detected_gpu_type:-unknown}" = "gb10" ]; then
    # Work around cuDNN frontend engine selection failures on some GB10 stacks.
    EXTRA_ENV_VARS="$EXTRA_ENV_VARS -e TORCH_CUDNN_V8_API_DISABLED=1"
fi

# Some GB10 software stacks cannot run the multimodal encoder profiling kernels
# (e.g. conv3d) for maximum feature sizes. Skipping profiling can allow the
# server to start; real requests may still be constrained by kernel support.
MM_PROFILING_FLAG=""
if [ "${detected_gpu_type:-unknown}" = "gb10" ]; then
    MM_PROFILING_FLAG="--skip-mm-profiling"
fi

# Display usage information
SHOW_HELP=false
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    SHOW_HELP=true
fi

# Light argument parsing for convenience.
# - Supports: --served-model-name (requested), --gpu-memory-utilization
# - Supports: "--" to pass remaining args to vLLM
while [[ "$SHOW_HELP" = false && $# -gt 0 ]]; do
    case "$1" in
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            SERVED_MODEL_NAME_WAS_SET=true
            shift 2
            ;;
        --served-model-name=*)
            SERVED_MODEL_NAME="${1#*=}"
            SERVED_MODEL_NAME_WAS_SET=true
            shift
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            GPU_MEMORY_UTILIZATION_WAS_SET=true
            shift 2
            ;;
        --gpu-memory-utilization=*)
            GPU_MEMORY_UTILIZATION="${1#*=}"
            GPU_MEMORY_UTILIZATION_WAS_SET=true
            shift
            ;;
        --)
            shift
            # Everything after "--" is passed to vLLM as-is
            if [[ $# -gt 0 ]]; then
                if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
                    VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS $*"
                else
                    VLLM_EXTRA_ARGS="$*"
                fi
            fi
            break
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        *)
            # Unknown args: treat as additional vLLM args (best-effort).
            if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
                VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS $1"
            else
                VLLM_EXTRA_ARGS="$1"
            fi
            shift
            ;;
    esac
done

if [[ "$SHOW_HELP" = true ]]; then
    echo "Usage: ./scripts/run_vllm_base.sh"
    echo ""
    echo "Environment variables:"
    echo "  MODEL_NAME          - Base model name or path (default: unsloth/Qwen3-VL-30B-A3B-Instruct)"
    echo "  PORT                - API server port (default: 8000)"
    echo "  HOST                - API server host (default: 0.0.0.0)"
    echo "  GPU_DEVICES         - GPU devices to use: 'all', '0-7', '0,1,2,3', or single index (default: all)"
    echo "  TENSOR_PARALLEL_SIZE - Tensor parallelism size (default: auto-detect from GPU_DEVICES)"
    echo "  MAX_MODEL_LEN       - Maximum model length (default: 32800)"
    echo "  GPU_MEMORY_UTILIZATION - GPU memory utilization ratio (default: 0.8, range: 0.0-1.0)"
    echo "  TRUST_REMOTE_CODE   - Trust remote code (default: true)"
    echo "  CONTAINER_NAME      - Docker container name (default: vllm-cua-server)"
    echo "  VLLM_IMAGE          - Docker image to use (default: auto-detect based on GPU type)"
    echo "  MODEL_HUB           - Model hub to use: 'huggingface' or 'modelscope' (default: modelscope)"
    echo "  HF_ENDPOINT         - Hugging Face endpoint (default: https://hf-mirror.com for China)"
    echo "  MODELSCOPE_CACHE    - ModelScope cache directory (default: ~/.cache/modelscope)"
    echo "  SERVED_MODEL_NAME   - Served model name shown in /v1/models (default: derived from model)"
    echo "  VLLM_EXTRA_ARGS     - Extra args appended to vLLM command (optional)"
    echo ""
    echo "Script arguments:"
    echo "  --served-model-name <name>         - Set served model name (same as SERVED_MODEL_NAME)"
    echo "  --gpu-memory-utilization <ratio>   - Set GPU memory utilization (same as GPU_MEMORY_UTILIZATION)"
    echo "  -- <args...>                       - Pass remaining args directly to vLLM"
    echo ""
    echo "Examples (MODEL_NAME only, BASE_MODEL is no longer used):"
    echo "  # Run with default settings (auto-detect all GPUs)"
    echo "  ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with 8 GPUs (GPUs 0-7)"
    echo "  GPU_DEVICES=0-7 TENSOR_PARALLEL_SIZE=8 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with specific 4 GPUs"
    echo "  GPU_DEVICES=0,1,2,3 TENSOR_PARALLEL_SIZE=4 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with custom port and 8 GPUs"
    echo "  PORT=8080 GPU_DEVICES=0-7 TENSOR_PARALLEL_SIZE=8 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Run with different VLM model on 8 GPUs"
    echo "  MODEL_NAME=unsloth/Qwen3-VL-30B-A3B-Instruct GPU_DEVICES=0-7 TENSOR_PARALLEL_SIZE=8 ./scripts/run_vllm_base.sh"
    echo ""
    echo "  # Use ModelScope (for China users) with 8 GPUs"
    echo "  MODEL_HUB=modelscope MODEL_NAME=unsloth/Qwen3-VL-30B-A3B-Instruct GPU_DEVICES=0-7 TENSOR_PARALLEL_SIZE=8 ./scripts/run_vllm_base.sh"
    echo ""
    exit 0
fi

# Check for GPU and prepare GPU flags
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null && [ "$TOTAL_GPUS" -gt 0 ]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""

    # If nvidia-smi cannot report memory (e.g. shows [N/A] on some GB10 stacks),
    # vLLM will still enforce a strict check at startup. Use a safer default on GB10
    # unless the user explicitly provided GPU_MEMORY_UTILIZATION.
    if [[ "${detected_gpu_type:-unknown}" = "gb10" && "$GPU_MEMORY_UTILIZATION_WAS_SET" = false ]]; then
        echo "GPU Memory Check:"
        echo "  - Note: GB10 + some drivers may show memory as [N/A] in nvidia-smi."
        echo "  - vLLM enforces a startup check using actual free memory."
        echo "  → Using a safer default GPU_MEMORY_UTILIZATION=0.75 for GB10 (override with GPU_MEMORY_UTILIZATION=... or --gpu-memory-utilization ...)"
        GPU_MEMORY_UTILIZATION="0.75"
        echo ""
    fi
    
    # Auto-adjust GPU memory utilization based on available memory
    # Get the first GPU's memory info (for single GPU) or calculate average
    if [ "$NUM_GPUS_TO_USE" -eq 1 ]; then
        # Single GPU: get the first GPU in GPU_LIST
        FIRST_GPU=$(echo "$GPU_LIST" | cut -d',' -f1)
        GPU_MEM_INFO=$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | awk -F', ' -v gpu="$FIRST_GPU" '$1==gpu {print $2, $3}')
    else
        # Multi-GPU: use first GPU as reference
        FIRST_GPU=$(echo "$GPU_LIST" | cut -d',' -f1)
        GPU_MEM_INFO=$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits | awk -F', ' -v gpu="$FIRST_GPU" '$1==gpu {print $2, $3}')
    fi
    
    if [ -n "$GPU_MEM_INFO" ]; then
        TOTAL_MEM=$(echo "$GPU_MEM_INFO" | awk '{print $1}')
        FREE_MEM=$(echo "$GPU_MEM_INFO" | awk '{print $2}')
        
        # Guard against non-numeric values (some drivers report "[N/A]")
        if [[ "$TOTAL_MEM" =~ ^[0-9]+$ ]] && [[ "$FREE_MEM" =~ ^[0-9]+$ ]] && [ "$TOTAL_MEM" -gt 0 ]; then
            # Calculate available memory ratio
            AVAILABLE_RATIO=$(awk "BEGIN {printf \"%.3f\", $FREE_MEM / $TOTAL_MEM}")
            REQUESTED_MEM=$(awk "BEGIN {printf \"%.2f\", $GPU_MEMORY_UTILIZATION * $TOTAL_MEM}")
            
            echo "GPU Memory Check:"
            echo "  - Total memory: ${TOTAL_MEM} MiB"
            echo "  - Free memory: ${FREE_MEM} MiB"
            echo "  - Available ratio: $(awk "BEGIN {printf \"%.1f\", $AVAILABLE_RATIO * 100}")%"
            echo "  - Requested utilization: ${GPU_MEMORY_UTILIZATION} (${REQUESTED_MEM} MiB)"
            
            # If requested memory exceeds available memory, adjust utilization
            if (( $(awk "BEGIN {print ($REQUESTED_MEM > $FREE_MEM)}") )); then
                # Set utilization to 90% of available memory to leave some headroom
                ADJUSTED_UTIL=$(awk "BEGIN {printf \"%.2f\", ($FREE_MEM * 0.90) / $TOTAL_MEM}")
                # Cap at 0.95 maximum
                if (( $(awk "BEGIN {print ($ADJUSTED_UTIL > 0.95)}") )); then
                    ADJUSTED_UTIL=0.95
                fi
                # Floor at 0.1 minimum
                if (( $(awk "BEGIN {print ($ADJUSTED_UTIL < 0.1)}") )); then
                    ADJUSTED_UTIL=0.1
                fi
                
                echo "  ⚠ Warning: Requested memory (${REQUESTED_MEM} MiB) exceeds available (${FREE_MEM} MiB)"
                echo "  → Auto-adjusting GPU_MEMORY_UTILIZATION from ${GPU_MEMORY_UTILIZATION} to ${ADJUSTED_UTIL}"
                GPU_MEMORY_UTILIZATION=$ADJUSTED_UTIL
            else
                echo "  ✓ Available memory is sufficient"
            fi
            echo ""
        fi
    fi
    
    echo "Multi-GPU Configuration:"
    echo "  - Total GPUs available: $TOTAL_GPUS"
    echo "  - GPUs to use: $GPU_LIST ($NUM_GPUS_TO_USE GPU(s))"
    echo "  - Tensor parallel size: $TENSOR_PARALLEL_SIZE"
    echo ""
    
    # Prepare GPU flags for Docker
    # Use --runtime nvidia instead of --gpus (matching gelato deployment)
    GPU_FLAGS="--runtime nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
    
    # For multi-GPU, add NCCL environment variables for better communication
    if [ "$NUM_GPUS_TO_USE" -gt 1 ]; then
        echo "  - Multi-GPU mode enabled (NCCL optimizations will be applied)"
    fi
else
    echo "Warning: No GPU detected. vLLM requires GPU for optimal performance."
    echo "Continuing anyway (may fail if CUDA is not available)..."
    GPU_FLAGS="--ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
fi

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Model Hub Selection
# MODEL_HUB: "huggingface" (default) or "modelscope"
MODEL_HUB="${MODEL_HUB:-huggingface}"

# ModelScope cache directory
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"

echo "Configuration:"
echo "  - Base model: $MODEL_NAME"
echo "  - Docker image: $VLLM_IMAGE"
echo "  - Model hub: $MODEL_HUB"
echo "  - Container name: $CONTAINER_NAME"
echo "  - Port: $PORT"
echo "  - Host: $HOST"
echo "  - GPU devices: $GPU_LIST ($NUM_GPUS_TO_USE GPU(s))"
echo "  - Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  - Max model length: $MAX_MODEL_LEN"
echo "  - GPU memory utilization: $GPU_MEMORY_UTILIZATION"
if [ "$MODEL_HUB" = "modelscope" ]; then
    echo "  - ModelScope cache: $MODELSCOPE_CACHE"
else
    echo "  - Hugging Face endpoint: $HF_ENDPOINT"
fi
echo "  - Auto restart: enabled"
echo ""

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo ""
fi

# Prepare Docker command with auto-restart and multi-GPU support
# Set NCCL environment variables for multi-GPU communication
NCCL_ENV_VARS=""
if [ "$NUM_GPUS_TO_USE" -gt 1 ]; then
    NCCL_ENV_VARS="-e NCCL_DEBUG=INFO \
        -e NCCL_IB_DISABLE=0 \
        -e NCCL_IB_GID_INDEX=3 \
        -e NCCL_SOCKET_IFNAME=^docker0,lo \
        -e NCCL_P2P_DISABLE=0 \
        -e NCCL_SHM_DISABLE=0"
fi

# Determine if we need to set entrypoint for images that don't default to vLLM API server
VLLM_ENTRYPOINT_FLAG=""
IMAGE_NEEDS_PY_ENTRYPOINT=false
if [[ "$VLLM_IMAGE" == *"gh200"* ]] || [[ "$VLLM_IMAGE" == *"rajesh550"* ]] || [[ "$VLLM_IMAGE" == nvcr.io/nvidia/vllm:* ]]; then
    IMAGE_NEEDS_PY_ENTRYPOINT=true
fi

# In ModelScope mode we need to run a pre-command (pip install), so force a bash entrypoint.
# Otherwise, on images like vllm/vllm-openai (entrypoint api_server.py), "bash -c ..." would be
# treated as api_server args and fail with "unrecognized arguments: -c ...".
if [ "$MODEL_HUB" = "modelscope" ]; then
    VLLM_ENTRYPOINT_FLAG="--entrypoint bash"
elif [ "$IMAGE_NEEDS_PY_ENTRYPOINT" = true ]; then
    # Some custom images don't set a vLLM entrypoint; use python3 and pass "-m ...".
    VLLM_ENTRYPOINT_FLAG="--entrypoint python3"
fi

if [ "$MODEL_HUB" = "modelscope" ]; then
    # ModelScope configuration
    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        $GPU_FLAGS \
        $VLLM_ENTRYPOINT_FLAG \
        -p $PORT:$PORT \
        -v $MODELSCOPE_CACHE:/root/.cache/modelscope \
        -v $HF_CACHE_DIR:/root/.cache/huggingface \
        -e MODELSCOPE_CACHE=/root/.cache/modelscope \
        -e HF_HOME=/root/.cache/huggingface \
        -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -e VLLM_MM_VIDEO_MAX_NUM=0 \
        -e VLLM_MM_IMAGE_MAX_NUM=16 \
        $EXTRA_ENV_VARS \
        $NCCL_ENV_VARS \
        $VLLM_IMAGE"
else
    # Hugging Face configuration
    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --restart=always \
        $GPU_FLAGS \
        $VLLM_ENTRYPOINT_FLAG \
        -p $PORT:$PORT \
        -v $HF_CACHE_DIR:/root/.cache/huggingface \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_HUB_ENABLE_HF_TRANSFER=0 \
        -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -e VLLM_MM_VIDEO_MAX_NUM=0 \
        -e VLLM_MM_IMAGE_MAX_NUM=16 \
        $EXTRA_ENV_VARS \
        $NCCL_ENV_VARS \
        $VLLM_IMAGE"
fi

# Prepare vLLM command arguments
# Note: Following gelato deployment pattern - simpler configuration without enforce-eager
# Video is disabled via environment variables (VLLM_MM_VIDEO_MAX_NUM=0)
# For vllm/vllm-openai image, the entrypoint is already set, so we just pass arguments
# For GH200 image, we may need to set entrypoint explicitly
if [ "$MODEL_HUB" = "modelscope" ]; then
    # ModelScope can be provided as:
    # - "org/name" (e.g. "Tongyi-MAI/MAI-UI-8B") -> map to ModelScope cache layout
    #   (ModelScope snapshot_download typically materializes to /root/.cache/modelscope/<org>/<name>)
    # - an absolute path inside container (e.g. "/root/.cache/modelscope/...") -> use as-is
    MODELSCOPE_REPO_ID=""
    if [[ "$MODEL_NAME" == /* ]]; then
        MODEL_PATH="$MODEL_NAME"
    elif [[ "$MODEL_NAME" == */* ]]; then
        MODELSCOPE_REPO_ID="$MODEL_NAME"
        MODEL_ORG="${MODEL_NAME%%/*}"
        MODEL_SHORT_NAME="${MODEL_NAME#*/}"
        MODEL_PATH="/root/.cache/modelscope/$MODEL_ORG/$MODEL_SHORT_NAME"
    else
        # Fallback: treat it as a repo id and map to cache root
        MODELSCOPE_REPO_ID="$MODEL_NAME"
        MODEL_PATH="/root/.cache/modelscope/$MODEL_NAME"
    fi

    # Default served model name (what appears in /v1/models) should be the repo id
    # rather than the local filesystem path.
    if [ -z "$SERVED_MODEL_NAME" ]; then
        if [ -n "$MODELSCOPE_REPO_ID" ]; then
            SERVED_MODEL_NAME="$MODELSCOPE_REPO_ID"
        else
            # Try to derive "org/name" from common cache layout: /root/.cache/modelscope/<org>/<name>
            if [[ "$MODEL_PATH" == /root/.cache/modelscope/*/* ]]; then
                tmp="${MODEL_PATH#/root/.cache/modelscope/}"  # org/name/...
                org="${tmp%%/*}"
                rest="${tmp#*/}"
                name="${rest%%/*}"
                SERVED_MODEL_NAME="$org/$name"
            else
                SERVED_MODEL_NAME="$MODEL_NAME"
            fi
        fi
    fi

    echo "Using ModelScope local path: $MODEL_PATH"
    VLLM_BASE_CMD="--model $MODEL_PATH --served-model-name $SERVED_MODEL_NAME --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization=$GPU_MEMORY_UTILIZATION --dtype $DTYPE $MM_PROFILING_FLAG"
else
    if [ -z "$SERVED_MODEL_NAME" ]; then
        SERVED_MODEL_NAME="$MODEL_NAME"
    fi
    VLLM_BASE_CMD="--model $MODEL_NAME --served-model-name $SERVED_MODEL_NAME --host $HOST --port $PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization=$GPU_MEMORY_UTILIZATION --dtype $DTYPE $MM_PROFILING_FLAG"
fi

# Append any additional args requested by the user.
if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    VLLM_BASE_CMD="$VLLM_BASE_CMD $VLLM_EXTRA_ARGS"
fi

# Add trust-remote-code flag if enabled
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    VLLM_BASE_CMD="$VLLM_BASE_CMD --trust-remote-code"
fi

# Enable auto tool choice for function calling support
# This is required when using tool_choice="auto" in API requests
VLLM_BASE_CMD="$VLLM_BASE_CMD --enable-auto-tool-choice"
# Use qwen3_coder parser for Qwen3 models (supports text-based extraction)
# Alternative: qwen3_xml (if model outputs XML format tool calls)
VLLM_BASE_CMD="$VLLM_BASE_CMD --tool-call-parser qwen3_coder"

# Build the command string
# Check if model requires newer transformers (Qwen3-VL)
if [[ "$MODEL_NAME" == *"Qwen3-VL"* ]] || [[ "$MODEL_NAME" == *"Qwen2.5-VL"* ]]; then
    NEEDS_NEW_TRANSFORMERS=true
else
    NEEDS_NEW_TRANSFORMERS=false
fi

# Build vLLM command based on image type and model hub
if [ "$MODEL_HUB" = "modelscope" ]; then
    # Entry point is bash; run install + launch explicitly.
    # If the model isn't present in the mounted cache yet, download it via ModelScope.
    # Also hard-disable video inputs at the CLI layer to avoid video-path profiling.
    VLLM_CMD="-lc \"pip install modelscope -q && export MODELSCOPE_CACHE=/root/.cache/modelscope && if [ -n '$MODELSCOPE_REPO_ID' ] && [ ! -d '$MODEL_PATH' ]; then python3 -c \\\"from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('$MODELSCOPE_REPO_ID', cache_dir='/root/.cache/modelscope')\\\"; fi && python3 -m vllm.entrypoints.openai.api_server $VLLM_BASE_CMD --limit-mm-per-prompt '{\\\"image\\\":16,\\\"video\\\":0}'\""
elif [ "$IMAGE_NEEDS_PY_ENTRYPOINT" = true ]; then
    # Entry point is python3; pass module invocation and args (do NOT prefix with "python3").
    VLLM_CMD="-m vllm.entrypoints.openai.api_server $VLLM_BASE_CMD"
else
    # vllm/vllm-openai image: entrypoint already runs api_server.py; just pass args.
    VLLM_CMD="$VLLM_BASE_CMD"
fi

echo "Starting vLLM inference server with auto-restart..."

if [ "$MODEL_HUB" = "modelscope" ]; then
    echo "Note: Installing ModelScope SDK..."
fi
echo ""

# Run the container in detached mode
if [ "$MODEL_HUB" = "modelscope" ]; then
    # Use eval for bash -c command
    CONTAINER_ID=$(eval "$DOCKER_CMD $VLLM_CMD")
else
    # Direct execution
    CONTAINER_ID=$($DOCKER_CMD $VLLM_CMD)
fi

echo "✓ Container started successfully!"
echo "  Container ID: ${CONTAINER_ID:0:12}"
echo "  Container name: $CONTAINER_NAME"
echo ""
echo "API will be available at: http://$HOST:$PORT/v1"
echo "OpenAI-compatible endpoints:"
echo "  - Chat completions: http://$HOST:$PORT/v1/chat/completions"
echo "  - Completions: http://$HOST:$PORT/v1/completions"
echo ""
echo "Model (served): $SERVED_MODEL_NAME"
echo "Multi-GPU: $NUM_GPUS_TO_USE GPU(s) (Tensor Parallel: $TENSOR_PARALLEL_SIZE)"
echo ""
echo "Useful commands:"
echo "  - View logs: docker logs -f $CONTAINER_NAME"
echo "  - Stop server: docker stop $CONTAINER_NAME"
echo "  - Remove container: docker rm $CONTAINER_NAME"
echo "  - Check GPU usage: nvidia-smi"
echo ""
echo "Example curl command (with image):"
echo '  curl http://localhost:'$PORT'/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model": "'$SERVED_MODEL_NAME'", "messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}], "max_tokens": 1000}'"'"
echo ""

# Follow logs
echo "Following container logs (Ctrl+C to detach, container will keep running)..."
echo "=========================================="
docker logs -f $CONTAINER_NAME

