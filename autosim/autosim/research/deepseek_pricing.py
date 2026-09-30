"""Pinned DeepSeek USD rate card for text-only Anthropic-compatible requests.

This is a budget estimate from provider usage, not an account invoice. The dated card must
be reviewed before it expires; a stale card is not silently treated as current pricing.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any


PRICE_CARD_ID = "deepseek-flash-v4.1-2026-09-10-usd"
PRICE_CARD_EXPIRES = dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc)
MAX_CONTEXT_TOKENS = 1_000_000
MAX_OUTPUT_TOKENS = 384_000
_RATES = {
    "peak": {"input_miss": 0.30, "input_hit": 0.006, "output": 1.20},
    "off_peak": {"input_miss": 0.15, "input_hit": 0.003, "output": 0.60},
}


def _utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        raise ValueError("pricing time must be timezone-aware")
    result = value.astimezone(dt.timezone.utc)
    if result >= PRICE_CARD_EXPIRES:
        raise ValueError("DeepSeek price card expired; verify official rates before running")
    return result


def _flash(model: str) -> None:
    if model != "deepseek-flash":
        raise ValueError("no verified DeepSeek price mapping for this model")


def peak_at(value: dt.datetime) -> bool:
    moment = _utc(value)
    hour = moment.hour
    return moment.weekday() < 5 and (1 <= hour < 4 or 6 <= hour < 10)


def _tokens(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def normalized_usage(value: dict[str, Any]) -> dict[str, int]:
    """Map the two documented response shapes without counting cached input twice."""
    if not isinstance(value, dict):
        raise ValueError("provider usage is missing")
    if "prompt_cache_miss_tokens" in value or "prompt_cache_hit_tokens" in value:
        missing = _tokens(value.get("prompt_cache_miss_tokens"), "cache miss tokens")
        hit = _tokens(value.get("prompt_cache_hit_tokens"), "cache hit tokens")
        output = _tokens(value.get("completion_tokens"), "output tokens")
    else:
        missing = _tokens(value.get("input_tokens"), "input tokens")
        hit = _tokens(value.get("cache_read_input_tokens", 0), "cache read tokens")
        missing += _tokens(value.get("cache_creation_input_tokens", 0),
                           "cache creation tokens")
        output = _tokens(value.get("output_tokens"), "output tokens")
    if missing + hit > MAX_CONTEXT_TOKENS or output > MAX_OUTPUT_TOKENS:
        raise ValueError("provider usage exceeds the supported model context/output bounds")
    return {"input_miss_tokens": missing, "input_hit_tokens": hit,
            "output_tokens": output}


def official_cost_usd(model: str, usage: dict[str, Any], *, at: dt.datetime
                      ) -> tuple[float, dict[str, int], str]:
    """Calculate a single request's cost from its response usage and request start time."""
    _flash(model)
    tier = "peak" if peak_at(at) else "off_peak"
    counts = normalized_usage(usage)
    rates = _RATES[tier]
    amount = sum(counts[f"{name}_tokens"] * rate / 1_000_000
                 for name, rate in rates.items())
    if not math.isfinite(amount):
        raise ValueError("calculated DeepSeek cost is invalid")
    return amount, counts, tier


def request_cost_ceiling_usd(model: str, max_tokens: int, *, at: dt.datetime,
                             input_bytes: int | None = None) -> float:
    """Conservative text-only request reservation using peak prices.

    A UTF-8 request byte bounds the number of text tokens it can encode. Without a measured
    body size, reserve the full context. Every possible input token is treated as a miss.
    """
    _flash(model)
    _utc(at)
    maximum = _tokens(max_tokens, "max_tokens")
    if maximum == 0 or maximum > MAX_OUTPUT_TOKENS:
        raise ValueError("request max_tokens is outside the priced model bound")
    if input_bytes is not None:
        if isinstance(input_bytes, bool) or not isinstance(input_bytes, int) or input_bytes <= 0:
            raise ValueError("request byte count is invalid")
        maximum_input = min(MAX_CONTEXT_TOKENS, input_bytes)
    else:
        maximum_input = MAX_CONTEXT_TOKENS
    peak = _RATES["peak"]
    return ((maximum_input * peak["input_miss"] +
             maximum * peak["output"]) / 1_000_000) * 1.05
