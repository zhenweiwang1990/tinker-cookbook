"""Doubao Private (Volcengine Ark) inference client.

This client mirrors the TS sample implementation:
- POST {base_url}/chat/completions
- Authorization: Bearer {api_key}
- Supports `thinking: { type: "disabled" | "enabled" }`

We keep the interface aligned with BaseInferenceClient: messages in, text out.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from .base_inference_client import BaseInferenceClient

logger = logging.getLogger(__name__)


class DoubaoPrivateInferenceClient(BaseInferenceClient):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        thinking: str = "disabled",
    ):
        self.provider = "doubao_private"
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.thinking = thinking

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        logger.info(
            f"DoubaoPrivateInferenceClient initialized: model={model_name}, base_url={self.base_url}, thinking={thinking}"
        )

    def get_provider_name(self) -> str:
        return self.provider

    def supports_logprobs(self) -> bool:
        return False

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tinker-cookbook message format to OpenAI-like format (image_url compatible)."""
        openai_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
                continue

            # Multimodal: list of {type:"text"|"image", ...}
            if isinstance(content, list):
                openai_content: list[dict[str, Any]] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type == "text":
                        openai_content.append({"type": "text", "text": part.get("text", "")})
                    elif part_type == "image":
                        image_data = part.get("image", "")
                        # We support:
                        # - data URI / URL (str)
                        # - base64 string (str, no data: prefix)
                        # - PIL.Image.Image
                        # - bytes/bytearray
                        if isinstance(image_data, str):
                            if image_data.startswith("data:") or image_data.startswith("http"):
                                url = image_data
                            else:
                                url = f"data:image/png;base64,{image_data}"
                            openai_content.append({"type": "image_url", "image_url": {"url": url}})
                        else:
                            try:
                                import base64
                                import io
                                from PIL import Image

                                if isinstance(image_data, Image.Image):
                                    buf = io.BytesIO()
                                    image_data.save(buf, format="PNG")
                                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                                elif isinstance(image_data, (bytes, bytearray)):
                                    b64 = base64.b64encode(bytes(image_data)).decode("ascii")
                                else:
                                    raise TypeError(f"Unsupported image type: {type(image_data)}")
                                url = f"data:image/png;base64,{b64}"
                                openai_content.append({"type": "image_url", "image_url": {"url": url}})
                            except Exception:
                                # Best-effort: if we can't encode, skip the image part.
                                continue
                openai_messages.append({"role": role, "content": openai_content})
                continue

            # Tool messages (rare in our CUA usage) - pass through as string.
            if role == "tool":
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": str(content),
                    }
                )
                continue

            openai_messages.append({"role": role, "content": str(content)})

        return openai_messages

    async def generate_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._prepare_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 1.0),
            "stream": False,
            "thinking": {"type": self.thinking},
        }

        logger.debug(f"[Doubao] POST {self.base_url}/chat/completions")
        logger.debug(f"[Doubao] Body: {json.dumps(payload, ensure_ascii=False)[:2000]}")

        resp = await self.client.post("/chat/completions", json=payload)
        if resp.status_code >= 300:
            # Try to read error body safely (some deployments may stream errors).
            detail: str
            try:
                detail = json.dumps(resp.json(), ensure_ascii=False)
            except Exception:
                detail = resp.text
            logger.error(f"[Doubao] Error {resp.status_code}: {detail}")
            raise RuntimeError(f"Doubao API error: {resp.status_code} - {detail}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            raise RuntimeError(f"Unexpected Doubao response shape: {data}") from e

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

