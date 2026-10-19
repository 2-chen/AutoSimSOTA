"""The network boundary applies the final text-only privacy scrub."""

import pytest

from autosim import llm_client


@pytest.mark.parametrize("variable", [None, "DEEPSEEK_MODEL", "AUTOSIM_LLM_MODEL",
                                      "OPENAI_MODEL"])
def test_direct_client_uses_flash_for_every_configuration_entry(monkeypatch, variable):
    for key in ("DEEPSEEK_MODEL", "AUTOSIM_LLM_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    if variable:
        monkeypatch.setenv(variable, "deepseek-flash")
    assert llm_client.LLMClient(api_key="fixture-key").model == "deepseek-flash"


@pytest.mark.parametrize("model", ["deepseek-v4-pro", "deepseek-pro",
                                  "deepseek-reasoner", "deepseek-v4-flash"])
@pytest.mark.parametrize("variable", ["explicit", "DEEPSEEK_MODEL", "AUTOSIM_LLM_MODEL",
                                      "OPENAI_MODEL"])
def test_direct_client_rejects_other_deepseek_models_before_network(monkeypatch, model,
                                                                 variable):
    for key in ("DEEPSEEK_MODEL", "AUTOSIM_LLM_MODEL", "OPENAI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda *a, **k: pytest.fail("paid request must not start"))
    if variable != "explicit":
        monkeypatch.setenv(variable, model)
    with pytest.raises(ValueError, match="must use deepseek-flash"):
        llm_client.LLMClient(api_key="fixture-key",
                            model=model if variable == "explicit" else None)


def test_direct_request_rechecks_model_after_client_mutation(monkeypatch):
    client = llm_client.LLMClient(api_key="fixture-key", model="deepseek-flash")
    client.model = "deepseek-v4-pro"
    monkeypatch.setattr(llm_client.requests, "post",
                        lambda *a, **k: pytest.fail("paid request must not start"))
    with pytest.raises(ValueError, match="must use deepseek-flash"):
        client.chat("system", "user")


def test_credential_file_rejects_pro_without_rewriting_configuration(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    contents = "DEEPSEEK_MODEL=deepseek-v4-pro\nDEEPSEEK_API_KEY=fixture-secret\n"
    path.write_text(contents)
    path.chmod(0o600)
    monkeypatch.setenv("AUTOSIM_ENV_FILE", str(path))
    for key in ("DEEPSEEK_MODEL", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"):
        # The loader assigns to os.environ itself; register even an originally absent
        # key so fixture teardown also removes values loaded from this temporary file.
        monkeypatch.setenv(key, "fixture-before-load")
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="must use deepseek-flash") as error:
        llm_client.load_credential_file(project_root=tmp_path)
    assert "fixture-secret" not in str(error.value)
    assert path.read_text() == contents


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
