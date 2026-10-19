"""Local, per-turn DeepSeek gate for Claude Code's Anthropic-compatible requests.

The CLI's dollar estimate is not used to authorize spend. A random local token protects the
loopback listener; only this proxy holds the upstream API key. Every admitted request reserves
its maximum priced cost before forwarding. Missing usage remains held, never priced as zero.
"""
from __future__ import annotations

import base64
import binascii
import datetime as dt
import http.server
import json
import secrets
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from .deepseek_pricing import (PRICE_CARD_ID, official_cost_usd,
                               request_cost_ceiling_usd)


class DeepSeekGatewayError(RuntimeError):
    """A request could not be safely admitted or accounted for."""


class TurnCostGate:
    MAX_REQUESTS = 256

    def __init__(self, *, limit_usd: float, model: str, extend_budget: Any = None,
                 checkpoint: Any = None):
        if limit_usd <= 0:
            raise ValueError("turn cost limit must be positive")
        self.limit_usd = float(limit_usd)
        self.model = model
        self.extend_budget = extend_budget
        self.denials: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._holds: dict[str, tuple[float, dt.datetime]] = {}
        self._request_profiles: dict[str, dict] = {}
        self.spent_usd = 0.0
        self.unknown = False
        self.unsafe_bound = False
        self.denied_count = 0
        self.receipts: list[dict[str, Any]] = []
        self.checkpoint = checkpoint
        self._checkpoint_unlocked()

    def _checkpoint_unlocked(self):
        if self.checkpoint is not None:
            try:
                self.checkpoint(self._snapshot_unlocked())
            except Exception as exc:
                self.unsafe_bound = True
                raise DeepSeekGatewayError('durable request accounting unavailable; do not forward') from exc

    def reserve(self, *, model: str, max_tokens: int, at: dt.datetime,
                input_bytes: int | None = None, request_profile: dict | None = None) -> str:
        ceiling = request_cost_ceiling_usd(model, max_tokens, at=at,
                                           input_bytes=input_bytes)
        with self._lock:
            if model != self.model or self.unsafe_bound:
                self.denied_count += 1
                raise DeepSeekGatewayError("unpriced model or unsafe usage bound")
            if len(self.receipts) + len(self._holds) >= self.MAX_REQUESTS:
                self.denied_count += 1
                raise DeepSeekGatewayError("turn DeepSeek request-count bound reached")
            held = sum(row[0] for row in self._holds.values()) + sum(
                row["reserved_usd"] for row in self.receipts if row["status"] == "unknown")
            required = self.spent_usd + held + ceiling
            if required > self.limit_usd + 1e-9 and self.extend_budget is not None:
                try:
                    self.limit_usd = float(self.extend_budget(required))
                except RuntimeError as exc:
                    self.denials.append({"category": "run_model_budget",
                        "reason": str(exc)[:180], "required_usd": required,
                        "input_bytes": input_bytes, "max_tokens": max_tokens})
            if self.spent_usd + held + ceiling > self.limit_usd + 1e-9:
                self.denied_count += 1
                if not self.denials or self.denials[-1].get("required_usd") != required:
                    self.denials.append({"category": "request_reservation_blocked",
                        "required_usd": required, "input_bytes": input_bytes,
                        "max_tokens": max_tokens, "limit_usd": self.limit_usd})
                raise DeepSeekGatewayError("turn DeepSeek budget cannot reserve this request")
            identity = uuid.uuid4().hex
            self._holds[identity] = (ceiling, at)
            if request_profile is not None:
                self._request_profiles[identity] = request_profile
            # Must be durable BEFORE the caller may forward a paid request.
            self._checkpoint_unlocked()
            return identity

    def settle(self, identity: str, usage: dict[str, Any] | None,
               diagnostics: dict[str, Any] | None = None) -> None:
        with self._lock:
            if identity not in self._holds:
                raise DeepSeekGatewayError("request reservation was not found")
            ceiling, at = self._holds.pop(identity)
            profile = self._request_profiles.pop(identity, None)
            profile_metadata = {"request_profile": profile} if profile is not None else {}
            try:
                if usage is None:
                    raise ValueError("provider usage is missing")
                cost, counts, tier = official_cost_usd(self.model, usage, at=at)
                if cost > ceiling + 1e-9:
                    self.unsafe_bound = True
                    raise ValueError("provider usage exceeded reserved cost ceiling")
            except (ValueError, TypeError) as exc:
                self.unknown = True
                self.receipts.append({"status": "unknown", "reserved_usd": ceiling,
                                      "requested_model": self.model,
                                      "price_card": PRICE_CARD_ID,
                                      "reason": str(exc)[:180],
                                      "diagnostics": diagnostics or {},
                                      **profile_metadata,
                                      "at": at.isoformat()})
                self._checkpoint_unlocked()
                return
            self.spent_usd += cost
            self.receipts.append({"status": "priced", "cost_usd": cost,
                                  "requested_model": self.model,
                                  "usage": counts, "tier": tier,
                                  "diagnostics":dict(diagnostics or {}),
                                  **profile_metadata,
                                  "price_card": PRICE_CARD_ID, "at": at.isoformat()})
            self._checkpoint_unlocked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        unknown_hold = sum(row["reserved_usd"] for row in self.receipts
                           if row["status"] == "unknown")
        inflight = sum(row[0] for row in self._holds.values())
        return {"model": self.model,
                "cost_usd": self.spent_usd, "unknown": self.unknown or bool(self._holds),
                "requests": len(self.receipts), "receipts": list(self.receipts),
                "denied_count": self.denied_count,
                "denials": list(self.denials), "limit_usd": self.limit_usd,
                "held_usd": inflight, "unknown_reserved_usd": unknown_hold,
                "accounting_ceiling_usd": self.spent_usd + unknown_hold + inflight,
                "bound_valid": not self.unsafe_bound}


def _has_nontext(value: Any) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("type"), str) and value["type"] in {
                "image", "document", "audio", "video"}:
            return True
        return any(_has_nontext(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_nontext(item) for item in value)
    return False


def _has_unsupported_media(value: Any, *, allow_images: bool) -> bool:
    """Admit opt-in inline images; reserve full input context for image requests.

    Compressed image bytes are not a token bound. Flash vision support and input
    token billing were verified in the official guide on 2026-10-16.
    """
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str) and kind in {"image", "document", "audio", "video"}:
            if kind != "image" or not allow_images:
                return True
            source = value.get("source")
            if (not isinstance(source, dict) or source.get("type") != "base64"
                    or source.get("media_type") not in {
                        "image/png", "image/jpeg", "image/gif", "image/webp"}
                    or not isinstance(source.get("data"), str) or not source["data"]):
                return True
            try:
                if not base64.b64decode(source["data"], validate=True):
                    return True
            except (ValueError, binascii.Error):
                return True
        return any(_has_unsupported_media(item, allow_images=allow_images) for item in value.values())
    if isinstance(value, list):
        return any(_has_unsupported_media(item, allow_images=allow_images) for item in value)
    return False


def _merge_usage(held: dict[str, Any], update: dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens", "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens", "completion_tokens"):
        value = update.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            held[key] = max(int(held.get(key) or 0), value)


def validate_generation_policy(policy: dict | None) -> dict:
    """Explicit controls for fresh, bounded reviews; never infer them from a role."""
    if policy is None:
        return {}
    if not isinstance(policy, dict) or set(policy) - {"max_output_tokens", "thinking"}:
        raise ValueError("unsupported generation policy")
    maximum, thinking = policy.get("max_output_tokens"), policy.get("thinking")
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int)
                                or not 1 <= maximum <= 65536):
        raise ValueError("review max_output_tokens must be an integer in 1..65536")
    if thinking is not None and thinking != "disabled":
        raise ValueError("bounded review only supports an explicit disabled thinking override")
    return {key: value for key, value in policy.items() if value is not None}


def apply_generation_policy(request: dict, policy: dict) -> dict:
    """Keep messages/tools/cache controls intact; limit the actual forwarded request."""
    result = dict(request)
    maximum = result.get("max_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("request max_tokens must be a positive integer")
    if policy.get("max_output_tokens") is not None:
        result["max_tokens"] = min(maximum, policy["max_output_tokens"])
    if policy.get("thinking") == "disabled":
        result["thinking"] = {"type": "disabled"}
        # Effort controls apply to thinking; don't send contradictory CLI defaults.
        if isinstance(result.get("output_config"), dict):
            config = {k: v for k, v in result["output_config"].items() if k != "effort"}
            if config:
                result["output_config"] = config
            else:
                result.pop("output_config")
    return result


class DeepSeekTurnGateway:
    def __init__(self, *, upstream_base_url: str, upstream_key: str,
                 model: str, limit_usd: float, extend_budget: Any = None,
                 generation_policy: dict | None = None, checkpoint: Any = None,
                 allow_images: bool = False):
        base = upstream_base_url.rstrip("/")
        if not base.startswith("https://") or not base.endswith("/anthropic"):
            raise ValueError("DeepSeek upstream must be an HTTPS Anthropic-compatible endpoint")
        if not upstream_key:
            raise ValueError("DeepSeek upstream key is required")
        # Refuse a stale or unpriced card before the CLI starts, not after it loops on 400s.
        request_cost_ceiling_usd(model, 1, at=dt.datetime.now(dt.timezone.utc),
                                 input_bytes=1)
        self.upstream_base_url = base
        self.generation_policy = validate_generation_policy(generation_policy)
        self.allow_images = bool(allow_images)
        self._upstream_key = upstream_key
        self.local_token = secrets.token_urlsafe(32)
        self.gate = TurnCostGate(limit_usd=limit_usd, model=model,
                                 extend_budget=extend_budget, checkpoint=checkpoint)
        self._incoming: list[dict[str, str]] = []
        self._rejections: list[str] = []
        self._incoming_lock = threading.Lock()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("gateway is not running")
        return f"http://127.0.0.1:{self._server.server_port}"

    def snapshot(self) -> dict[str, Any]:
        result = self.gate.snapshot()
        result["generation_policy"] = dict(self.generation_policy)
        result["allow_images"] = self.allow_images
        with self._incoming_lock:
            result["incoming"] = list(self._incoming)
            result["rejections"] = list(self._rejections)
        return result

    def _note_incoming(self, method: str, path: str) -> None:
        with self._incoming_lock:
            if len(self._incoming) < 32:
                self._incoming.append({"method": method, "path": path.split("?", 1)[0][:120]})

    def _note_rejection(self, reason: str) -> None:
        with self._incoming_lock:
            if len(self._rejections) < 32:
                self._rejections.append(reason[:80])

    def __enter__(self) -> "DeepSeekTurnGateway":
        gateway = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                # Never log request headers, body, local token, or upstream key.
                return

            def _error(self, code: int, message: str) -> None:
                body = json.dumps({"error": {"type": "gateway_error",
                                              "message": message}}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
                gateway._note_incoming("POST", self.path)
                parsed = urlsplit(self.path)
                if (parsed.path != "/v1/messages" or parsed.scheme or parsed.netloc or
                        len(parsed.query) > 2048):
                    gateway._note_rejection("unsupported_path")
                    self._error(404, "unsupported API path")
                    return
                supplied = self.headers.get("Authorization", "")
                supplied_key = self.headers.get("x-api-key", "")
                if (not secrets.compare_digest(supplied, "Bearer " + gateway.local_token) and
                        not secrets.compare_digest(supplied_key, gateway.local_token)):
                    gateway._note_rejection("local_auth_failed")
                    self._error(403, "local gateway authentication failed")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 16 * 1024 * 1024:
                        raise ValueError("request body size is unsupported")
                    raw = self.rfile.read(length)
                    request = json.loads(raw)
                    if (not isinstance(request, dict)
                            or _has_unsupported_media(request, allow_images=gateway.allow_images)):
                        raise ValueError("only priced text and explicitly enabled inline images are supported")
                    has_images = _has_nontext(request)
                    requested_maximum=request.get('max_tokens')
                    if gateway.generation_policy:
                        request = apply_generation_policy(request, gateway.generation_policy)
                        raw = json.dumps(request, ensure_ascii=False).encode("utf-8")
                    model = request.get("model")
                    maximum = request.get("max_tokens")
                    at = dt.datetime.now(dt.timezone.utc)
                    from .request_fingerprint import request_fingerprint
                    identity = gateway.gate.reserve(model=model, max_tokens=maximum, at=at,
                        input_bytes=None if has_images else len(raw),
                        request_profile={**request_fingerprint(request, body_bytes=len(raw)),
                                         "input_modality": "image_and_text" if has_images else "text"})
                except (ValueError, TypeError, DeepSeekGatewayError) as exc:
                    gateway._note_rejection(type(exc).__name__ + ":" + str(exc)[:80])
                    # Local deterministic refusal is not a provider rate limit. Do not
                    # let the CLI spend its watchdog window retrying the same request.
                    self._error(400,
                                str(exc)[:180])
                    return

                usage: dict[str, Any] = {}
                diagnostics: dict[str, Any] = {"phase": "upstream_connect",
                    "requested_max_output_tokens":requested_maximum,
                    "effective_max_output_tokens":maximum}
                stream_complete = False
                settled = False
                upstream = urllib.request.Request(
                    gateway.upstream_base_url + "/v1/messages" +
                    ("?" + parsed.query if parsed.query else ""), data=raw,
                    headers={"Content-Type": "application/json",
                             "Accept": self.headers.get("Accept", "application/json"),
                             "anthropic-version": self.headers.get("anthropic-version",
                                                                   "2023-06-01"),
                             **({"anthropic-beta": self.headers["anthropic-beta"]}
                                if "anthropic-beta" in self.headers else {}),
                             "x-api-key": gateway._upstream_key}, method="POST")
                try:
                    with urllib.request.urlopen(upstream, timeout=900) as response:
                        diagnostics.update(http_status=response.status, phase="response_read")
                        content_type = response.headers.get("Content-Type", "")
                        if request.get("stream") is True:
                            self.send_response(response.status)
                            self.send_header("Content-Type", content_type or "text/event-stream")
                            self.send_header("Cache-Control", "no-cache")
                            self.send_header("Transfer-Encoding", "chunked")
                            self.send_header("Connection", "close")
                            self.end_headers()
                            client_open = True
                            while True:
                                line = response.readline(2 * 1024 * 1024)
                                if not line:
                                    break
                                if line.startswith(b"data:"):
                                    try:
                                        event = json.loads(line[5:].lstrip())
                                        reason=(event.get('delta') or {}).get('stop_reason')
                                        if reason is None:
                                            reason=(event.get('message') or {}).get('stop_reason')
                                        if isinstance(reason,str):
                                            diagnostics['provider_stop_reason']=reason[:80]
                                        if event.get("type") == "message_stop":
                                            stream_complete = True
                                        _merge_usage(usage, event.get("usage") or {})
                                        _merge_usage(usage, (event.get("message") or {}).get(
                                            "usage") or {})
                                        if stream_complete and not settled:
                                            # Seal known usage before exposing the terminal
                                            # event. The CLI may exit as soon as it sees it.
                                            gateway.gate.settle(identity, usage or None, diagnostics=diagnostics)
                                            settled = True
                                    except (ValueError, TypeError, AttributeError):
                                        pass
                                if client_open:
                                    try:
                                        self.wfile.write(f"{len(line):x}\r\n".encode() +
                                                         line + b"\r\n")
                                        self.wfile.flush()
                                    except (BrokenPipeError, ConnectionResetError):
                                        client_open = False
                            if client_open:
                                self.wfile.write(b"0\r\n\r\n")
                                self.wfile.flush()
                        else:
                            body = response.read(16 * 1024 * 1024 + 1)
                            if len(body) > 16 * 1024 * 1024:
                                raise ValueError("provider response exceeded gateway bound")
                            result = json.loads(body)
                            if isinstance(result.get('stop_reason'),str):
                                diagnostics['provider_stop_reason']=result['stop_reason'][:80]
                            _merge_usage(usage, result.get("usage") or {})
                            gateway.gate.settle(identity, usage or None, diagnostics=diagnostics)
                            settled = True
                            self.send_response(response.status)
                            self.send_header("Content-Type", content_type or "application/json")
                            self.send_header("Content-Length", str(len(body)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                            self.wfile.write(body)
                except urllib.error.HTTPError as exc:
                    diagnostics.update(http_status=exc.code, error_type="HTTPError")
                    self._error(exc.code, "upstream rejected the request; usage is unknown")
                except (OSError, ValueError, TypeError) as exc:
                    diagnostics.update(error_type=type(exc).__name__, errno=getattr(exc, "errno", None))
                    try:
                        self._error(502, "upstream response failed; usage is unknown")
                    except (OSError, ValueError):
                        pass
                finally:
                    if request.get("stream") is True and not stream_complete:
                        diagnostics.setdefault("error_type", "IncompleteStream")
                        # Initial input usage is not a receipt for all generated output.
                        usage = {}
                    if not settled:
                        gateway.gate.settle(identity, usage or None, diagnostics=diagnostics)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
                gateway._note_incoming("GET", self.path)
                self._error(404, "unsupported API path")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


@contextmanager
def deepseek_turn_gateway(*, upstream_base_url: str, upstream_key: str,
                          model: str, limit_usd: float,
                          extend_budget: Any = None,
                          generation_policy: dict | None = None,
                          checkpoint: Any = None,
                          allow_images: bool = False) -> Iterator[DeepSeekTurnGateway]:
    with DeepSeekTurnGateway(upstream_base_url=upstream_base_url,
                             upstream_key=upstream_key, model=model,
                             limit_usd=limit_usd, extend_budget=extend_budget,
                             generation_policy=generation_policy, checkpoint=checkpoint,
                             allow_images=allow_images) as gateway:
        yield gateway
