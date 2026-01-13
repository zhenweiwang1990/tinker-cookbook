from __future__ import annotations

import json
import re
from typing import Any


def repair_json_text(s: str) -> str:
    """
    Best-effort JSON repair for common LLM formatting mistakes.

    This is intentionally conservative: we only apply transforms that are very likely
    to preserve semantics (no deep inference).
    """
    s = s.strip()

    # Strip common code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Common glitch: extra "{" before sibling keys, e.g.
    #   {"a": {...},{"b": {...}}}  ->  {"a": {...}, "b": {...}}
    s = re.sub(r"\}\s*,\s*\{\s*(\"[a-zA-Z0-9_]+\"\s*:)", r"}, \1", s)

    # Missing comma between a closed object and next key:
    #   {"a": {...} "b": 1} -> {"a": {...}, "b": 1}
    s = re.sub(r"\}\s+(\"[a-zA-Z0-9_]+\"\s*:)", r"}, \1", s)

    # If user pasted <tool_call> blocks, remove the tags.
    s = re.sub(r"^\s*<tool_call>\s*", "", s)
    s = re.sub(r"\s*</tool_call>\s*$", "", s)

    return s.strip()


def extract_first_json(s: str, start_chars: tuple[str, ...] = ("{", "[")) -> Any | None:
    """
    Find and parse the first valid JSON object/array inside a larger string.

    Returns the parsed object, or None if extraction fails.
    """
    s = repair_json_text(s)
    decoder = json.JSONDecoder()

    # Try from the first occurrence of each start char and subsequent occurrences.
    starts: list[int] = []
    for ch in start_chars:
        starts.extend([m.start() for m in re.finditer(re.escape(ch), s)])
    starts = sorted(set(starts))[:100]

    for idx in starts:
        try:
            obj, _end = decoder.raw_decode(s, idx=idx)
            return obj
        except Exception:
            continue
    return None


def loads_dict(s: str) -> dict[str, Any] | None:
    obj = extract_first_json(s, start_chars=("{",))
    return obj if isinstance(obj, dict) else None

