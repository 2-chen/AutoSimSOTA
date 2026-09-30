"""Local, per-turn DeepSeek gate for Claude Code's Anthropic-compatible requests.

The CLI's dollar estimate is not used to authorize spend. A random local token protects the
loopback listener; only this proxy holds the upstream API key. Every text request reserves
its maximum priced cost before forwarding. Missing usage remains held, never priced as zero.
"""
from __future__ import annotations

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

    def __init__(self, *, limit_usd: float, model: str, extend_budget: Any = None):
        if limit_usd <= 0:
            raise ValueError("turn cost limit must be positive")
        self.limit_usd = float(limit_usd)
        self.model = model
        self.extend_budget = extend_budget
        self.denials: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._holds: dict[str, tuple[float, dt.datetime]] = {}
        self.spent_usd = 0.0
        self.unknown = False
        self.denied_count = 0
        self.receipts: list[dict[str, Any]] = []

    def reserve(self, *, model: str, max_tokens: int, at: dt.datetime,
                input_bytes: int | None = None) -> str:
        ceiling = request_cost_ceiling_usd(model, max_tokens, at=at,
                                           input_bytes=input_bytes)
        with self._lock:
            if model != self.model or self.unknown:
                self.denied_count += 1
                raise DeepSeekGatewayError("unpriced model or unknown prior request usage")
            if len(self.receipts) + len(self._holds) >= self.MAX_REQUESTS:
                self.denied_count += 1
                raise DeepSeekGatewayError("turn DeepSeek request-count bound reached")
            held = sum(row[0] for row in self._holds.values())
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
            return identity

    def settle(self, identity: str, usage: dict[str, Any] | None) -> None:
        with self._lock:
            if identity not in self._holds:
                raise DeepSeekGatewayError("request reservation was not found")
            ceiling, at = self._holds.pop(identity)
            try:
                if usage is None:
                    raise ValueError("provider usage is missing")
                cost, counts, tier = official_cost_usd(self.model, usage, at=at)
                if cost > ceiling + 1e-9:
                    raise ValueError("provider usage exceeded reserved cost ceiling")
            except (ValueError, TypeError):
                self.unknown = True
                self.receipts.append({"status": "unknown", "reserved_usd": ceiling,
                                      "price_card": PRICE_CARD_ID,
                                      "at": at.isoformat()})
                return
            self.spent_usd += cost
            self.receipts.append({"status": "priced", "cost_usd": cost,
                                  "usage": counts, "tier": tier,
                                  "price_card": PRICE_CARD_ID, "at": at.isoformat()})

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"cost_usd": self.spent_usd, "unknown": self.unknown or bool(self._holds),
                    "requests": len(self.receipts), "receipts": list(self.receipts),
                    "denied_count": self.denied_count,
                    "denials": list(self.denials), "limit_usd": self.limit_usd,
                    "held_usd": sum(row[0] for row in self._holds.values())}


def _has_nontext(value: Any) -> bool:
    if isinstance(value, dict):
        if isinstance(value.get("type"), str) and value["type"] in {
                "image", "document", "audio", "video"}:
            return True
        return any(_has_nontext(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_nontext(item) for item in value)
    return False


def _merge_usage(held: dict[str, Any], update: dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens", "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens", "completion_tokens"):
        value = update.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            held[key] = max(int(held.get(key) or 0), value)


class DeepSeekTurnGateway:
    def __init__(self, *, upstream_base_url: str, upstream_key: str,
                 model: str, limit_usd: float, extend_budget: Any = None):
        base = upstream_base_url.rstrip("/")
        if not base.startswith("https://") or not base.endswith("/anthropic"):
            raise ValueError("DeepSeek upstream must be an HTTPS Anthropic-compatible endpoint")
        if not upstream_key:
            raise ValueError("DeepSeek upstream key is required")
        # Refuse a stale or unpriced card before the CLI starts, not after it loops on 400s.
        request_cost_ceiling_usd(model, 1, at=dt.datetime.now(dt.timezone.utc),
                                 input_bytes=1)
        self.upstream_base_url = base
        self._upstream_key = upstream_key
        self.local_token = secrets.token_urlsafe(32)
        self.gate = TurnCostGate(limit_usd=limit_usd, model=model,
                                 extend_budget=extend_budget)
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
                    if not isinstance(request, dict) or _has_nontext(request):
                        raise ValueError("only priced text requests are supported")
                    model = request.get("model")
                    maximum = request.get("max_tokens")
                    at = dt.datetime.now(dt.timezone.utc)
                    identity = gateway.gate.reserve(model=model, max_tokens=maximum, at=at,
                                                    input_bytes=len(raw))
                except (ValueError, TypeError, DeepSeekGatewayError) as exc:
                    gateway._note_rejection(type(exc).__name__ + ":" + str(exc)[:80])
                    # Local deterministic refusal is not a provider rate limit. Do not
                    # let the CLI spend its watchdog window retrying the same request.
                    self._error(400,
                                str(exc)[:180])
                    return

                usage: dict[str, Any] = {}
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
                                if line.startswith(b"data: "):
                                    try:
                                        event = json.loads(line[6:])
                                        _merge_usage(usage, event.get("usage") or {})
                                        _merge_usage(usage, (event.get("message") or {}).get(
                                            "usage") or {})
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
                            _merge_usage(usage, result.get("usage") or {})
                            self.send_response(response.status)
                            self.send_header("Content-Type", content_type or "application/json")
                            self.send_header("Content-Length", str(len(body)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                            self.wfile.write(body)
                except urllib.error.HTTPError as exc:
                    self._error(exc.code, "upstream rejected the request; usage is unknown")
                except (OSError, ValueError, TypeError) as exc:
                    try:
                        self._error(502, "upstream response failed; usage is unknown")
                    except (OSError, ValueError):
                        pass
                finally:
                    gateway.gate.settle(identity, usage or None)

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
                          extend_budget: Any = None) -> Iterator[DeepSeekTurnGateway]:
    with DeepSeekTurnGateway(upstream_base_url=upstream_base_url,
                             upstream_key=upstream_key, model=model,
                             limit_usd=limit_usd, extend_budget=extend_budget) as gateway:
        yield gateway
