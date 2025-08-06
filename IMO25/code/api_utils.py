from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests

from .config import MODEL_PROVIDER, API_URL_BASE, CEREBRAS_MODEL_DEFAULT
from .logging_utils import log_print


def get_api_key(agent_type: str) -> str:
    if MODEL_PROVIDER == "cerebras":
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            log_print("Error: CEREBRAS_API_KEY environment variable not set for Cerebras provider.")
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
            log_print("Error: Neither IMPROVER_API_KEY nor CEO_API_KEY environment variables set.")
            log_print("Please check your .env file.")
            sys.exit(1)
        return api_key
    api_key = os.getenv("CEO_API_KEY")
    if not api_key:
        log_print("Error: CEO_API_KEY environment variable not set.")
        log_print("Please check your .env file.")
        sys.exit(1)
    return api_key


def build_request_payload(system_prompt: str, question_prompt: str, other_prompts=None, temperature: float = 0.1, top_p: float = 1.0, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_prompt},
    ]
    if other_prompts:
        for prompt in other_prompts:
            messages.append({"role": "user", "content": prompt})
    payload: Dict[str, Any] = {"messages": messages, "temperature": temperature, "top_p": top_p}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def send_openrouter_request(api_key: str, payload: Dict[str, Any], model_name: str, agent_type: str = "unknown", max_retries: int = 3, telemetry=None) -> Dict[str, Any]:
    api_url = API_URL_BASE
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lyang36/IMO25",
        "X-Title": f"IMO25-{agent_type}",
    }
    payload["model"] = model_name
    for attempt in range(max_retries):
        log_print(f"[{agent_type.upper()}] Sending request to OpenRouter API ({model_name})... (Attempt {attempt + 1}/{max_retries})")
        try:
            start = time.time()
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=(30, 30))
            duration = time.time() - start
            if telemetry:
                telemetry.record_api_call(duration)
            response.raise_for_status()
            response_text = response.text
            preview = response_text if len(response_text) <= 500 else response_text[:500] + '... [truncated]'
            log_print(f"[{agent_type.upper()}] API request succeeded. Status: {response.status_code}")
            log_print(f"[{agent_type.upper()}] Response preview: {preview}")
            try:
                if not response_text.strip():
                    log_print(f"[{agent_type.upper()}] Warning: Empty response received")
                    return {"choices": [{"message": {"content": ""}}]}
                return response.json()
            except json.JSONDecodeError as e:
                log_print(f"[{agent_type.upper()}] JSON decode error: {e}")
                log_print(f"[{agent_type.upper()}] Raw response length: {len(response_text)}")
                try:
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        partial_json = json_match.group(0)
                        log_print(f"[{agent_type.upper()}] Found potential JSON fragment, length: {len(partial_json)}")
                        return json.loads(partial_json)
                except json.JSONDecodeError:
                    log_print(f"[{agent_type.upper()}] Failed to parse JSON fragment")
                log_print(f"[{agent_type.upper()}] Raw response (first 1000 chars): {response_text[:1000]}")
                return {"choices": [{"message": {"content": "Error: Failed to parse API response"}}]}
        except requests.exceptions.Timeout:
            duration = time.time() - start if 'start' in locals() else 0
            if telemetry:
                telemetry.record_api_call(duration)
            log_print(f"[{agent_type.upper()}] API request timed out (Attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                log_print(f"[{agent_type.upper()}] Retrying in 2 seconds...")
                time.sleep(2)
            else:
                log_print(f"[{agent_type.upper()}] All retry attempts failed. API request timed out.")
                return {"choices": [{"message": {"content": "Error: API request timed out"}}]}
        except requests.exceptions.RequestException as e:
            duration = time.time() - start if 'start' in locals() else 0
            if telemetry:
                telemetry.record_api_call(duration)
            log_print(f"[{agent_type.upper()}] Error during API request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                log_print(f"[{agent_type.upper()}] Status code: {e.response.status_code}")
                log_print(f"[{agent_type.upper()}] Response text: {e.response.text}")
            if attempt < max_retries - 1:
                log_print(f"[{agent_type.upper()}] Retrying in 2 seconds...")
                time.sleep(2)
            else:
                log_print(f"[{agent_type.upper()}] All retry attempts failed. API request failed.")
                return {"choices": [{"message": {"content": f"Error: API request failed with exception {e}"}}]}
    return {"choices": [{"message": {"content": "Error: Unhandled API failure"}}]}


def send_cerebras_request(api_key: str, payload: Dict[str, Any], model_name: str, agent_type: str = "unknown", telemetry=None) -> Dict[str, Any]:
    start = time.time()
    try:
        try:
            from cerebras.cloud.sdk import Cerebras
        except Exception as import_err:
            return {"choices": [{"message": {"content": f"Error: cerebras-cloud-sdk not installed ({import_err})"}}]}
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
        return {"choices": [{"message": {"content": f"Error: Cerebras request failed with exception {e}"}}]}


def send_api_request(api_key: str, payload: Dict[str, Any], model_name: str, agent_type: str = "unknown", max_retries: int = 3, telemetry=None) -> Dict[str, Any]:
    if MODEL_PROVIDER == "cerebras":
        return send_cerebras_request(api_key, payload, model_name, agent_type=agent_type, telemetry=telemetry)
    return send_openrouter_request(api_key, payload, model_name, agent_type=agent_type, max_retries=max_retries, telemetry=telemetry)


def extract_text_from_response(response_data: Dict[str, Any]) -> str:
    try:
        if 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            message = choice.get('message', {})
            content = message.get('content', "")
            return content
    except (KeyError, IndexError, TypeError) as e:
        log_print("Error: Could not extract text from the API response.")
        log_print(f"Reason: {e}")
        log_print("Full API Response:")
        log_print(json.dumps(response_data, indent=2))
    return ""
