"""
This module associates model names with metadata, which helps  training code choose good defaults.
"""

from dataclasses import dataclass
from functools import cache
import re


@dataclass
class ModelAttributes:
    organization: str  # meta-llama, Qwen, etc.
    version_str: str  # just the version number e.g. "3.1", "2.5"
    size_str: str  # size of the model e.g. "8B", "72B", "1.5B"
    is_chat: bool  # is chat/instruct model
    is_vl: bool = False  # is vision-language model


@cache
def get_llama_info() -> dict[str, ModelAttributes]:
    org = "meta-llama"
    return {
        "Llama-3.2-1B-Instruct": ModelAttributes(org, "3.2", "1B", True),
        "Llama-3.2-3B-Instruct": ModelAttributes(org, "3.2", "3B", True),
        "Llama-3.1-8B-Instruct": ModelAttributes(org, "3.1", "8B", True),
        "Llama-3.2-1B": ModelAttributes(org, "3.2", "1B", False),
        "Llama-3.2-3B": ModelAttributes(org, "3.2", "3B", False),
        "Llama-3.1-8B": ModelAttributes(org, "3.1", "8B", False),
        "Llama-3.1-70B": ModelAttributes(org, "3.1", "70B", False),
        "Llama-3.3-70B-Instruct": ModelAttributes(org, "3.3", "70B", True),
    }


def get_qwen_info() -> dict[str, ModelAttributes]:
    org = "Qwen"
    return {
        "Qwen2.5-VL-7B-Instruct": ModelAttributes(org, "2.5", "7B", True, is_vl=True),
        "Qwen3-VL-30B-A3B-Instruct": ModelAttributes(org, "3", "30B-A3B", True, is_vl=True),
        "Qwen3-VL-235B-A22B-Instruct": ModelAttributes(org, "3", "235B-A22B", True, is_vl=True),
        "Qwen3-4B-Base": ModelAttributes(org, "3", "4B", False),
        "Qwen3-8B-Base": ModelAttributes(org, "3", "8B", False),
        "Qwen3-14B-Base": ModelAttributes(org, "3", "14B", False),
        "Qwen3-30B-A3B-Base": ModelAttributes(org, "3", "30B-A3B", False),
        "Qwen3-0.6B": ModelAttributes(org, "3", "0.6B", True),
        "Qwen3-1.7B": ModelAttributes(org, "3", "1.7B", True),
        "Qwen3-4B": ModelAttributes(org, "3", "4B", True),
        "Qwen3-8B": ModelAttributes(org, "3", "8B", True),
        "Qwen3-14B": ModelAttributes(org, "3", "14B", True),
        "Qwen3-32B": ModelAttributes(org, "3", "32B", True),
        "Qwen3-30B-A3B": ModelAttributes(org, "3", "30B-A3B", True),
        "Qwen3-4B-Instruct-2507": ModelAttributes(org, "3", "4B", True),
        "Qwen3-30B-A3B-Instruct-2507": ModelAttributes(org, "3", "30B-A3B", True),
        "Qwen3-235B-A22B-Instruct-2507": ModelAttributes(org, "3", "235B-A22B", True),
    }


def get_deepseek_info() -> dict[str, ModelAttributes]:
    org = "deepseek-ai"
    return {
        "DeepSeek-V3.1": ModelAttributes(org, "3", "671B-A37B", True),
        "DeepSeek-V3.1-Base": ModelAttributes(org, "3", "671B-A37B", False),
    }


def get_gpt_oss_info() -> dict[str, ModelAttributes]:
    org = "openai"
    return {
        "gpt-oss-20b": ModelAttributes(org, "1", "21B-A3.6B", True),
        "gpt-oss-120b": ModelAttributes(org, "1", "117B-A5.1B", True),
    }


def get_moonshot_info() -> dict[str, ModelAttributes]:
    org = "moonshotai"
    return {
        "Kimi-K2-Thinking": ModelAttributes(org, "K2", "1T-A32B", True),
    }


@cache
def get_tongyi_mai_info() -> dict[str, ModelAttributes]:
    """
    Tongyi-MAI models.

    Notes:
    - Tongyi-MAI/MAI-UI-8B is a post-trained variant of Qwen3-VL-8B.
    - We treat it as a Qwen-family model for default renderer selection.
    """
    family = "Qwen"
    return {
        "MAI-UI-8B": ModelAttributes(family, "3", "8B", True, is_vl=True),
    }


def _split_org_model(model_name: str) -> tuple[str, str]:
    """
    Split a model identifier into (org, model_name).

    Supports both canonical hub IDs like "Qwen/Qwen3-8B" and local filesystem paths like
    "/root/.cache/modelscope/Tongyi-MAI/MAI-UI-8B" by taking the last two path components.
    """
    parts = [p for p in re.split(r"[\\/]+", model_name) if p]
    if len(parts) < 2:
        raise ValueError(f"Unknown model: {model_name}")
    return parts[-2], parts[-1]


def _get_or_raise(
    info: dict[str, ModelAttributes], model_name: str, model_key: str
) -> ModelAttributes:
    try:
        return info[model_key]
    except KeyError as e:
        raise ValueError(f"Unknown model: {model_name}") from e


def get_model_attributes(model_name: str) -> ModelAttributes:
    org, model_version_full = _split_org_model(model_name)
    if org == "meta-llama":
        return _get_or_raise(get_llama_info(), model_name, model_version_full)
    elif org == "Qwen":
        return _get_or_raise(get_qwen_info(), model_name, model_version_full)
    elif org == "deepseek-ai":
        return _get_or_raise(get_deepseek_info(), model_name, model_version_full)
    elif org == "openai":
        return _get_or_raise(get_gpt_oss_info(), model_name, model_version_full)
    elif org == "moonshotai":
        return _get_or_raise(get_moonshot_info(), model_name, model_version_full)
    elif org == "Tongyi-MAI":
        return _get_or_raise(get_tongyi_mai_info(), model_name, model_version_full)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_names(model_name: str) -> list[str]:
    """
    Return a list of renderers that are designed for the model.
    Used so we can emit a warning if you use a non-recommended renderer.
    The first result is the most recommended renderer for the model.
    """
    org, model_version_full = _split_org_model(model_name)
    if (org, model_version_full) == ("Tongyi-MAI", "MAI-UI-8B"):
        # This model does not carry the "-Instruct" suffix in its hub ID, but it is a
        # post-trained Qwen-VL variant. Try Qwen2.5-VL style first, then fall back to Qwen3-VL.
        return ["qwen25_vl_instruct", "qwen3_vl_instruct"]

    attributes = get_model_attributes(model_name)
    if not attributes.is_chat:
        return ["role_colon"]
    elif attributes.organization == "meta-llama":
        return ["llama3"]
    elif attributes.organization == "Qwen":
        if attributes.version_str == "2.5":
            # Qwen2.5-VL and Qwen2.5 LLMs share the <|im_start|> chat-template family.
            if attributes.is_vl:
                return ["qwen25_vl_instruct"] if "-Instruct" in model_name else ["qwen25_vl_instruct"]
            return ["role_colon"]
        elif attributes.version_str == "3":
            if attributes.is_vl:
                if "-Instruct" in model_name:
                    return ["qwen3_vl_instruct"]
                else:
                    return ["qwen3_vl"]
            elif "-Instruct" in model_name:
                return ["qwen3_instruct"]
            else:
                return ["qwen3", "qwen3_disable_thinking"]
        else:
            raise ValueError(f"Unknown model: {model_name}")
    elif attributes.organization == "deepseek-ai":
        return ["deepseekv3_disable_thinking", "deepseekv3"]
    elif attributes.organization == "openai":
        return ["gpt_oss_no_sysprompt", "gpt_oss_medium_reasoning"]
    elif attributes.organization == "moonshotai":
        return ["kimi_k2"]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_name(model_name: str) -> str:
    """
    Return the most recommended renderer for the model.
    """
    return get_recommended_renderer_names(model_name)[0]


def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model identifier to "org/model" form.

    This is useful when users pass filesystem paths (e.g. vLLM served with a local path),
    but our metadata/heuristics expect canonical "org/model" naming.
    """
    org, model_version_full = _split_org_model(model_name)
    return f"{org}/{model_version_full}"


def get_tokenizer_model_name(model_name: str) -> str:
    """
    Return a model identifier that can be used with HuggingFace/Transformers to load
    tokenizer + processor.

    For provider-specific identifiers (like local paths, or models not hosted on HF),
    this returns a compatible base model that shares the same tokenizer/chat template.
    """
    normalized = normalize_model_name(model_name)
    if normalized == "Tongyi-MAI/MAI-UI-8B":
        # Avoid depending on HF network access at runtime.
        # Qwen3-VL tokenizer/template is compatible enough for our parsing + prompt formatting,
        # and is commonly already cached in environments that run CUA.
        return "Qwen/Qwen3-VL-30B-A3B-Instruct"
    return normalized
