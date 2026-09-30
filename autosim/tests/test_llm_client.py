"""The network boundary applies the final text-only privacy scrub."""

from autosim import llm_client


def test_llm_client_sanitizes_prompts_at_the_network_boundary(monkeypatch):
    sent = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"id": "test", "model": "stub", "choices": [
                {"message": {"content": "ok"}, "finish_reason": "stop"}]}

    def fake_post(url, *, headers, json, timeout):
        sent.update(url=url, headers=headers, payload=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    client = llm_client.LLMClient(api_key="fixture-key", base_url="https://provider.invalid",
                                  model="stub")

    content, _ = client.chat_with_metadata(
        'Read source. DEEPSEEK_API_KEY = "PRIVATE_PROMPT_SECRET"',
        'DATA_ROOT="/home/private-user/demos"; benchmark_path="/data/libero"\n'
        '#!/bin/bash',
    )

    assert content == "ok"
    messages = sent["payload"]["messages"]
    transmitted = "\n".join(message["content"] for message in messages)
    assert "PRIVATE_PROMPT_SECRET" not in transmitted
    assert "/home/private-user" not in transmitted
    assert "/data/libero" in transmitted
    assert "#!/bin/bash" in transmitted
    assert sent["headers"]["Authorization"] == "Bearer fixture-key"
