"""OpenAI-compatible LLM client used by AutoSim.

Credentials are read from the process environment and are never included in
returned metadata.  DeepSeek-specific variables intentionally take precedence
over the legacy OpenAI aliases so ``autosim research`` has one unambiguous
configuration surface.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import requests


DEFAULT_LLM_BASE_URL = "http://10.1.21.21:3000/v1"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_LLM_API_KEY = ""  # Explicit environment/credential configuration required.

#: The checkout this system lives in. Moved here when the research entry point stopped being
#: one benchmark's runner: a path to the project is not any benchmark's business.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_credential_file(*, project_root: Path | None = None) -> dict[str, Any]:
    """Load the project's own dotenv, and never return a value from it.

    A dotenv file is not an operating-system environment, so this reads exactly one: the
    project's `.env`, or the file `AUTOSIM_ENV_FILE` names. It does not search parent
    directories, and it refuses a file that other users can read -- a credential in a
    world-readable file is a credential already disclosed.

    Only the three variables this client reads are taken. The return value names what was
    loaded and whether a key is present; it never carries the key.
    """
    root = (project_root or PROJECT_ROOT).absolute()
    explicit = os.environ.get("AUTOSIM_ENV_FILE")
    candidates = [Path(explicit).absolute()] if explicit else [root / ".env"]
    source = next((path.absolute() for path in candidates if path.is_file()), None)
    loaded: list[str] = []
    if source is not None:
        if source.stat().st_mode & 0o077:
            raise PermissionError(
                f"credential file must not be group/world accessible: {source}")
        for line in source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.removeprefix("export ").split("=", 1)
            key, value = key.strip(), value.strip()
            if key not in {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"}:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key not in os.environ and value:
                os.environ[key] = value
                loaded.append(key)
    os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-flash")
    return {"source": str(source) if source else "process_environment_only",
            "loaded_variable_names": sorted(loaded),
            "model": os.environ["DEEPSEEK_MODEL"],
            "base_url": os.environ["DEEPSEEK_BASE_URL"],
            "api_key_available": bool(os.environ.get("DEEPSEEK_API_KEY"))}


class LLMClient:
    """Small OpenAI chat/completions wrapper with env-based defaults."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("AUTOSIM_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or DEFAULT_LLM_API_KEY
        )
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("AUTOSIM_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_LLM_BASE_URL
        )
        self.model = (
            model
            or os.environ.get("DEEPSEEK_MODEL")
            or os.environ.get("AUTOSIM_LLM_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_LLM_MODEL
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 8192,
        timeout: int = 180,
    ) -> str:
        """Call /chat/completions and return the first assistant message."""
        content, _ = self.chat_with_metadata(system, user, max_tokens=max_tokens, timeout=timeout)
        return content

    def chat_with_metadata(
        self,
        system: str,
        user: str,
        max_tokens: int = 8192,
        timeout: int = 180,
        retries: int = 2,
        thinking: Optional[str] = None,
    ) -> tuple[str, dict]:
        """Return content plus non-secret provider metadata.

        Authentication failures are terminal.  Rate limits and transient
        provider/server failures receive at most ``retries`` retries.
        """
        if not self.api_key:
            return "", {"available": False, "model": self.model, "base_url": self.base_url}

        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            url = base_url
        else:
            url = f"{base_url}/chat/completions"

        # Every production request crosses this boundary, even when it originated in a
        # newly added controller path. This is defense in depth for credentials and
        # machine-local home/temp paths; callers still must project structured data (for
        # example demonstration records) before it reaches this text-only API.
        from .research.common import sanitize_model_text

        system = sanitize_model_text(system)
        user = sanitize_model_text(user)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        attempts = 0
        while True:
            attempts += 1
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
                # `json_object` is not a hint, it is a contract: the provider rejects the
                # request unless the prompt contains the word "json". Sending it
                # unconditionally made the requirement invisible -- a caller whose prompt did
                # not happen to say "json" got an opaque HTTP 400 having asked for nothing
                # unusual, and the failure looked like the prompt being wrong rather than
                # like this function promising something the prompt had not agreed to.
                if "json" in (system + " " + user).lower():
                    payload["response_format"] = {"type": "json_object"}
                if thinking is not None:
                    if thinking not in {"enabled", "disabled"}:
                        raise ValueError("thinking must be enabled or disabled")
                    payload["thinking"] = {"type": thinking}
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                if attempts >= retries + 1:
                    raise RuntimeError(f"LLM API transport error: {type(exc).__name__}") from exc
                time.sleep(min(2 ** (attempts - 1), 4))
                continue
            if resp.status_code == 200:
                break
            # Providers may echo credentials or prompt material in error bodies.
            if resp.status_code in {401, 403} or attempts >= retries + 1 or (
                resp.status_code != 429 and resp.status_code < 500
            ):
                raise RuntimeError(f"LLM API HTTP {resp.status_code}")
            time.sleep(min(2 ** (attempts - 1), 4))

        payload = resp.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("LLM API returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM API returned empty content")
        metadata = {
            "available": True,
            "provider_request_id": payload.get("id"),
            "provider_model": payload.get("model"),
            "requested_model": self.model,
            "base_url": self.base_url,
            "attempts": attempts,
            "usage": payload.get("usage"),
            "finish_reason": choices[0].get("finish_reason"),
            "thinking": thinking or "provider_default",
        }
        return content, metadata
