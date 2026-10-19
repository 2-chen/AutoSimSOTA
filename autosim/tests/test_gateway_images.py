import base64
import http.client
import io
import json
from urllib.parse import urlparse

import pytest

from autosim.research import deepseek_gateway as module


def image():
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
            "data": base64.b64encode(b"small-compressed-fixture").decode()}}


def test_inline_images_require_explicit_enablement_and_supported_media():
    assert module._has_unsupported_media(image(), allow_images=False)
    assert not module._has_unsupported_media(image(), allow_images=True)
    for kind in ("document", "audio", "video"):
        assert module._has_unsupported_media({"type": kind}, allow_images=True)
    assert module._has_unsupported_media(
        {"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}},
        allow_images=True)
    broken = image()
    broken["source"]["data"] = "not base64!"
    assert module._has_unsupported_media(broken, allow_images=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_image_gateway_reserves_context_not_compressed_bytes(monkeypatch, enabled):
    forwarded = []
    held = []

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def upstream(request, **kwargs):
        forwarded.append(json.loads(request.data))
        assert held[-1]["held_usd"] > .30
        return Response(json.dumps({"usage": {"input_tokens": 1040, "output_tokens": 10},
                                    "content": [{"type": "text", "text": "ok"}]}).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", upstream)
    with module.DeepSeekTurnGateway(upstream_base_url="https://api.deepseek.com/anthropic",
            upstream_key="fixture", model="deepseek-flash", limit_usd=1,
            allow_images=enabled, checkpoint=lambda value: held.append(value)) as gateway:
        url = urlparse(gateway.base_url)
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        body = {"model": "deepseek-flash", "max_tokens": 100,
                "messages": [{"role": "user", "content": [image()]}]}
        conn.request("POST", "/v1/messages", body=json.dumps(body),
                     headers={"Authorization": "Bearer " + gateway.local_token,
                              "Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == (200 if enabled else 400)
        response.read()
        conn.close()
        snapshot = gateway.snapshot()
        assert len(forwarded) == int(enabled)
        assert snapshot["requests"] == int(enabled)
        if enabled:
            assert forwarded[0]["messages"] == body["messages"]
            assert snapshot["accounting_ceiling_usd"] < .01
            assert snapshot["held_usd"] == 0
