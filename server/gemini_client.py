"""Async Gemini API client wrapper verified against google-genai >= 3.6.1.

Covers:
- Decisions API (`MODEL_DECISIONS`, v1beta) via `client.aio.decisions.create`
- Interactions API (`MODEL_HYDROGEN`, `MODEL_NEON`, v1beta) via `client.aio.interactions.create`

Enforces concurrency limit (semaphore=8), per-call timeout (8 s), 1 retry with backoff,
latency/token logging, and graceful failure (never crashes the tick loop).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

from google import genai
from pydantic import BaseModel

from server.config import (
    API_CONCURRENCY,
    API_TIMEOUT_SECONDS,
    GEMINI_API_KEY,
    MODEL_DECISIONS,
    MODEL_HYDROGEN,
)

logger = logging.getLogger("agent_crowd.gemini")

T = TypeVar("T")


@dataclass
class ApiResult(Generic[T]):
    ok: bool
    value: Optional[T] = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    rate_limited: bool = False


def _is_rate_limit_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return "429" in s or "rate" in s or "resource_exhausted" in s or "quota" in s


def _extract_usage(usage: Any) -> tuple[int, int, int, int]:
    if usage is None:
        return 0, 0, 0, 0
    in_tok = int(getattr(usage, "total_input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "total_output_tokens", 0) or 0)
    thought_tok = int(getattr(usage, "total_thought_tokens", 0) or 0)
    tot_tok = int(getattr(usage, "total_tokens", 0) or 0)
    return in_tok, out_tok, thought_tok, tot_tok


class GeminiClient:
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        concurrency: int = API_CONCURRENCY,
        default_timeout: float = API_TIMEOUT_SECONDS,
    ) -> None:
        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )
        self._decide_sem = asyncio.Semaphore(concurrency)
        self._interact_sem = asyncio.Semaphore(concurrency)
        self._default_timeout = default_timeout

    async def decide(
        self,
        *,
        input_blocks: list[dict[str, Any]],
        questions: Mapping[str, dict[str, Any]],
        model: str = MODEL_DECISIONS,
        timeout: Optional[float] = None,
        label: str = "decisions",
    ) -> ApiResult[Any]:
        """Run a v1beta Decisions call with semaphore, timeout, and 1 retry."""
        t_limit = self._default_timeout if timeout is None else timeout
        last_err: Optional[str] = None
        was_rate_limited = False
        t_start = time.perf_counter()

        for attempt in range(2):
            try:
                async with self._decide_sem:
                    call_start = time.perf_counter()
                    res = await asyncio.wait_for(
                        self._client.aio.decisions.create(
                            model=model,
                            input=input_blocks,
                            questions=questions,
                        ),
                        timeout=t_limit,
                    )
                    latency_ms = (time.perf_counter() - call_start) * 1000.0
                in_tok, out_tok, thought_tok, tot_tok = _extract_usage(getattr(res, "usage", None))
                logger.info(
                    "[%s] model=%s q=%d latency=%.1fms tokens=%d (in=%d out=%d)",
                    label,
                    model,
                    len(questions),
                    latency_ms,
                    tot_tok,
                    in_tok,
                    out_tok,
                )
                return ApiResult(
                    ok=True,
                    value=res,
                    latency_ms=latency_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    thought_tokens=thought_tok,
                    total_tokens=tot_tok,
                    rate_limited=was_rate_limited,
                )
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if _is_rate_limit_error(exc):
                    was_rate_limited = True
                logger.warning("[%s] attempt %d failed: %s", label, attempt + 1, last_err)
                if attempt == 0:
                    await asyncio.sleep(0.8 if was_rate_limited else 0.4)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        logger.error("[%s] exhausted retries (%.1fms): %s", label, total_ms, last_err)
        return ApiResult(ok=False, latency_ms=total_ms, error=last_err, rate_limited=was_rate_limited)

    async def interact_structured(
        self,
        *,
        input_blocks: list[dict[str, Any]],
        schema_model: Type[T],
        model: str = MODEL_HYDROGEN,
        system_instruction: Optional[str] = None,
        thinking_level: str = "low",
        store: bool = False,
        timeout: Optional[float] = None,
        label: str = "hydrogen",
    ) -> ApiResult[T]:
        """Run a v1beta Interactions call with JSON schema output and Pydantic validation."""
        t_limit = self._default_timeout if timeout is None else timeout
        last_err: Optional[str] = None
        was_rate_limited = False
        t_start = time.perf_counter()

        kwargs: dict[str, Any] = {
            "model": model,
            "input": input_blocks,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema_model.model_json_schema(),  # type: ignore[attr-defined]
            },
            "generation_config": {"thinking_level": thinking_level},
            "store": store,
        }
        if system_instruction is not None:
            kwargs["system_instruction"] = system_instruction

        for attempt in range(2):
            try:
                async with self._interact_sem:
                    call_start = time.perf_counter()
                    interaction = await asyncio.wait_for(
                        self._client.aio.interactions.create(**kwargs),
                        timeout=t_limit,
                    )
                    latency_ms = (time.perf_counter() - call_start) * 1000.0

                raw_text = interaction.output_text
                if not raw_text:
                    raise ValueError(f"Empty output_text (status={interaction.status})")
                parsed = schema_model.model_validate_json(raw_text)  # type: ignore[attr-defined]
                in_tok, out_tok, thought_tok, tot_tok = _extract_usage(
                    getattr(interaction, "usage", None)
                )
                logger.info(
                    "[%s] model=%s think=%s latency=%.1fms tokens=%d (in=%d thought=%d out=%d)",
                    label,
                    model,
                    thinking_level,
                    latency_ms,
                    tot_tok,
                    in_tok,
                    thought_tok,
                    out_tok,
                )
                return ApiResult(
                    ok=True,
                    value=parsed,
                    latency_ms=latency_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    thought_tokens=thought_tok,
                    total_tokens=tot_tok,
                    rate_limited=was_rate_limited,
                )
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if _is_rate_limit_error(exc):
                    was_rate_limited = True
                logger.warning("[%s] attempt %d failed: %s", label, attempt + 1, last_err)
                if attempt == 0:
                    await asyncio.sleep(0.8 if was_rate_limited else 0.4)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        logger.error("[%s] exhausted retries (%.1fms): %s", label, total_ms, last_err)
        return ApiResult(ok=False, latency_ms=total_ms, error=last_err, rate_limited=was_rate_limited)

    async def aclose(self) -> None:
        if hasattr(self._client.aio, "aclose"):
            await self._client.aio.aclose()
