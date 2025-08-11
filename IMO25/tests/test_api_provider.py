import os
import sys
import importlib
import types


def test_send_api_request_routes_to_gemini(monkeypatch):
    code_dir = os.path.join(os.path.dirname(__file__), "..", "code")
    pkg_name = "imo25_code"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [code_dir]
    sys.modules[pkg_name] = pkg

    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")

    config = importlib.import_module(f"{pkg_name}.config")
    api_utils = importlib.import_module(f"{pkg_name}.api_utils")
    agent = importlib.import_module(f"{pkg_name}.agent")

    importlib.reload(config)
    importlib.reload(api_utils)
    importlib.reload(agent)

    assert agent.send_api_request is api_utils.send_api_request

    calls = {"gemini": False, "openrouter": False}

    def fake_gemini(api_key, payload, model_name, agent_type="unknown", telemetry=None):
        calls["gemini"] = True
        return {"choices": [{"message": {"content": ""}}]}

    def fake_open(api_key, payload, model_name, agent_type="unknown", max_retries=3, telemetry=None):
        calls["openrouter"] = True
        return {"choices": [{"message": {"content": ""}}]}

    monkeypatch.setattr(api_utils, "send_gemini_request", fake_gemini)
    monkeypatch.setattr(api_utils, "send_openrouter_request", fake_open)

    agent.send_api_request("key", {}, "model", agent_type="worker")

    assert calls["gemini"]
    assert not calls["openrouter"]
