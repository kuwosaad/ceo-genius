from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests

try:
    from .config import (
        MODEL_PROVIDER,
        API_URL_BASE,
        CEREBRAS_MODEL_DEFAULT,
        GEMINI_API_URL_BASE,
    )
    from .logging_utils import log_print
except ImportError:
    from config import (
        MODEL_PROVIDER,
        API_URL_BASE,
        CEREBRAS_MODEL_DEFAULT,
        GEMINI_API_URL_BASE,
    )
    from logging_utils import log_print


def get_api_key(agent_type: str) -> str:
    if MODEL_PROVIDER == "cerebras":
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            log_print(
                "Error: CEREBRAS_API_KEY environment variable not set for Cerebras provider."
            )
            log_print("Please check your .env file or set MODEL_PROVIDER=openrouter.")
            sys.exit(1)
        return api_key
    if MODEL_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            log_print(
                "Error: GEMINI_API_KEY environment variable not set for Gemini provider."
            )
            log_print("Please check your .env file or set MODEL_PROVIDER=openrouter.")
            sys.exit(1)
        return api_key
    if agent_type == "strategist":
        api_key = os.getenv("CEO_API_KEY")
        if not api_key:
            log_print("Error: CEO_API_KEY environment variable not set.")
            log_print("Please check your .env file.")
            sys.exit(1)
        return api_key
    if agent_type == "worker":
        api_key = os.getenv("GENIUS_API_KEY")
        if not api_key:
            log_print("Error: GENIUS_API_KEY environment variable not set.")
            log_print("Please check your .env file.")
            sys.exit(1)
        return api_key
    if agent_type == "improver":
        api_key = os.getenv("IMPROVER_API_KEY") or os.getenv("CEO_API_KEY")
        if not api_key:
            log_print(
                "Error: Neither IMPROVER_API_KEY nor CEO_API_KEY environment variables set."
            )
            log_print("Please check your .env file.")
            sys.exit(1)
        return api_key
    api_key = os.getenv("CEO_API_KEY")
    if not api_key:
        log_print("Error: CEO_API_KEY environment variable not set.")
        log_print("Please check your .env file.")
        sys.exit(1)
    return api_key


def build_request_payload(
    system_prompt: str,
    question_prompt: str,
    other_prompts=None,
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_prompt},
    ]
    if other_prompts:
        for prompt in other_prompts:
            messages.append({"role": "user", "content": prompt})
    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def send_openrouter_request(
    api_key: str,
    payload: Dict[str, Any],
    model_name: str,
    agent_type: str = "unknown",
    max_retries: int = 3,
    telemetry=None,
) -> Dict[str, Any]:
    api_url = API_URL_BASE
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lyang36/IMO25",
        "X-Title": f"IMO25-{agent_type}",
    }
    payload["model"] = model_name
    for attempt in range(max_retries):
        log_print(
            f"[{agent_type.upper()}] Sending request to OpenRouter API ({model_name})... (Attempt {attempt + 1}/{max_retries})"
        )
        try:
            start = time.time()
            response = requests.post(
                api_url, headers=headers, data=json.dumps(payload), timeout=(30, 30)
            )
            duration = time.time() - start
            if telemetry:
                telemetry.record_api_call(duration)
            response.raise_for_status()
            response_text = response.text
            preview = (
                response_text
                if len(response_text) <= 500
                else response_text[:500] + "... [truncated]"
            )
            log_print(
                f"[{agent_type.upper()}] API request succeeded. Status: {response.status_code}"
            )
            log_print(f"[{agent_type.upper()}] Response preview: {preview}")
            try:
                if not response_text.strip():
                    log_print(
                        f"[{agent_type.upper()}] Warning: Empty response received"
                    )
                    return {"choices": [{"message": {"content": ""}}]}
                return response.json()
            except json.JSONDecodeError as e:
                log_print(f"[{agent_type.upper()}] JSON decode error: {e}")
                log_print(
                    f"[{agent_type.upper()}] Raw response length: {len(response_text)}"
                )
                try:
                    import re

                    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                    if json_match:
                        partial_json = json_match.group(0)
                        log_print(
                            f"[{agent_type.upper()}] Found potential JSON fragment, length: {len(partial_json)}"
                        )
                        return json.loads(partial_json)
                except json.JSONDecodeError:
                    log_print(f"[{agent_type.upper()}] Failed to parse JSON fragment")
                log_print(
                    f"[{agent_type.upper()}] Raw response (first 1000 chars): {response_text[:1000]}"
                )
                return {
                    "choices": [
                        {"message": {"content": "Error: Failed to parse API response"}}
                    ]
                }
        except requests.exceptions.Timeout:
            duration = time.time() - start if "start" in locals() else 0
            if telemetry:
                telemetry.record_api_call(duration)
            log_print(
                f"[{agent_type.upper()}] API request timed out (Attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                log_print(f"[{agent_type.upper()}] Retrying in 2 seconds...")
                time.sleep(2)
            else:
                log_print(
                    f"[{agent_type.upper()}] All retry attempts failed. API request timed out."
                )
                return {
                    "choices": [
                        {"message": {"content": "Error: API request timed out"}}
                    ]
                }
        except requests.exceptions.RequestException as e:
            duration = time.time() - start if "start" in locals() else 0
            if telemetry:
                telemetry.record_api_call(duration)
            log_print(f"[{agent_type.upper()}] Error during API request: {e}")
            if hasattr(e, "response") and e.response is not None:
                log_print(
                    f"[{agent_type.upper()}] Status code: {e.response.status_code}"
                )
                log_print(f"[{agent_type.upper()}] Response text: {e.response.text}")
            if attempt < max_retries - 1:
                log_print(f"[{agent_type.upper()}] Retrying in 2 seconds...")
                time.sleep(2)
            else:
                log_print(
                    f"[{agent_type.upper()}] All retry attempts failed. API request failed."
                )
                return {
                    "choices": [
                        {
                            "message": {
                                "content": f"Error: API request failed with exception {e}"
                            }
                        }
                    ]
                }
    return {"choices": [{"message": {"content": "Error: Unhandled API failure"}}]}


def send_cerebras_request(
    api_key: str,
    payload: Dict[str, Any],
    model_name: str,
    agent_type: str = "unknown",
    telemetry=None,
) -> Dict[str, Any]:
    start = time.time()
    try:
        try:
            from cerebras.cloud.sdk import Cerebras
        except Exception as import_err:
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"Error: cerebras-cloud-sdk not installed ({import_err})"
                        }
                    }
                ]
            }
        client = Cerebras(api_key=api_key)
        messages = payload.get("messages", [])
        temperature = payload.get("temperature", 0.1)
        top_p = payload.get("top_p", 1.0)
        max_tokens = payload.get("max_tokens")
        res = client.chat.completions.create(
            messages=messages,
            model=model_name or CEREBRAS_MODEL_DEFAULT,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        duration = time.time() - start
        if telemetry:
            telemetry.record_api_call(duration)
        try:
            content = res.choices[0].message.content
        except Exception:
            try:
                content = res["choices"][0]["message"]["content"]
            except Exception:
                content = str(res)
        return {"choices": [{"message": {"content": content}}]}
    except Exception as e:
        duration = time.time() - start
        if telemetry:
            telemetry.record_api_call(duration)
        return {
            "choices": [
                {
                    "message": {
                        "content": f"Error: Cerebras request failed with exception {e}"
                    }
                }
            ]
        }


def _convert_to_gemini_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert our OpenAI-style messages payload to Gemini's generateContent format.
    Gemini expects: {
      "contents": [
        {"role": "user"|"model", "parts": [{"text": "..."}]} , ...
      ]
      , generationConfig: {temperature, topP, maxOutputTokens}
    }
    We'll collapse all system prompts and user prompts into a single user turn in order.
    """
    messages = payload.get("messages", [])
    contents = []
    # Gemini roles: user/model; map system->user, user->user, assistant->model
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        if role not in ("user", "assistant", "system"):
            role = "user"
        gem_role = "user" if role in ("user", "system") else "model"
        contents.append({"role": gem_role, "parts": [{"text": text}]})
    gen_cfg: Dict[str, Any] = {}
    if "temperature" in payload:
        gen_cfg["temperature"] = payload.get("temperature")
    if "top_p" in payload:
        gen_cfg["topP"] = payload.get("top_p")
    if "max_tokens" in payload and payload.get("max_tokens") is not None:
        gen_cfg["maxOutputTokens"] = payload.get("max_tokens")
    req: Dict[str, Any] = {"contents": contents}
    if gen_cfg:
        req["generationConfig"] = gen_cfg
    return req


def send_gemini_request(
    api_key: str,
    payload: Dict[str, Any],
    model_name: str,
    agent_type: str = "unknown",
    telemetry=None,
) -> Dict[str, Any]:
    """
    Send a request to Gemini's REST API using the generateContent endpoint.
    Endpoint: {GEMINI_API_URL_BASE}/models/{model}:generateContent?key=API_KEY
    Response format normalized to {"choices": [{"message": {"content": str}}]} for compatibility.
    """
    # Build endpoint
    base = GEMINI_API_URL_BASE.rstrip("/")
    url = f"{base}/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = _convert_to_gemini_request(payload)
    try:
        start = time.time()
        res = requests.post(
            url, headers=headers, data=json.dumps(body), timeout=(30, 30)
        )
        duration = time.time() - start
        if telemetry:
            telemetry.record_api_call(duration)
        res.raise_for_status()
        data = res.json()
        # Gemini response: candidates[0].content.parts[].text
        text = ""
        try:
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
                text = "\n".join(t for t in texts if t)
        except Exception:
            text = ""
        return {"choices": [{"message": {"content": text}}]}
    except requests.exceptions.RequestException as e:
        log_print(f"[{agent_type.upper()}] Gemini request error: {e}")
        try:
            resp_text = res.text  # type: ignore[name-defined]
            log_print(f"[{agent_type.upper()}] Response text: {resp_text[:500]}")
        except Exception:
            pass
        return {
            "choices": [
                {
                    "message": {
                        "content": f"Error: Gemini request failed with exception {e}"
                    }
                }
            ]
        }


def send_api_request(
    api_key: str,
    payload: Dict[str, Any],
    model_name: str,
    agent_type: str = "unknown",
    max_retries: int = 3,
    telemetry=None,
) -> Dict[str, Any]:
    if MODEL_PROVIDER == "cerebras":
        return send_cerebras_request(
            api_key, payload, model_name, agent_type=agent_type, telemetry=telemetry
        )
    if MODEL_PROVIDER == "gemini":
        return send_gemini_request(
            api_key, payload, model_name, agent_type=agent_type, telemetry=telemetry
        )
    return send_openrouter_request(
        api_key,
        payload,
        model_name,
        agent_type=agent_type,
        max_retries=max_retries,
        telemetry=telemetry,
    )


def extract_text_from_response(response_data: Dict[str, Any]) -> str:
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            return content
    except (KeyError, IndexError, TypeError) as e:
        log_print("Error: Could not extract text from the API response.")
        log_print(f"Reason: {e}")
        log_print("Full API Response:")
        log_print(json.dumps(response_data, indent=2))
    return ""
