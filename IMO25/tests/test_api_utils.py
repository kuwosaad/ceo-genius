import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import api_utils


def test_send_api_request_routing(monkeypatch):
    calls = []

    def fake(name):
        def _inner(*args, **kwargs):
            calls.append(name)
            return {}

        return _inner

    monkeypatch.setattr(api_utils, "send_openrouter_request", fake("openrouter"))
    monkeypatch.setattr(api_utils, "send_cerebras_request", fake("cerebras"))
    monkeypatch.setattr(api_utils, "send_gemini_request", fake("gemini"))

    monkeypatch.setattr(api_utils, "MODEL_PROVIDER", "gemini")
    api_utils.send_api_request("key", {}, "model")
    assert calls == ["gemini"]

    calls.clear()
    monkeypatch.setattr(api_utils, "MODEL_PROVIDER", "cerebras")
    api_utils.send_api_request("key", {}, "model")
    assert calls == ["cerebras"]

    calls.clear()
    monkeypatch.setattr(api_utils, "MODEL_PROVIDER", "openrouter")
    api_utils.send_api_request("key", {}, "model")
    assert calls == ["openrouter"]
