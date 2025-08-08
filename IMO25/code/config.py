from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

STRATEGIST_MODEL_NAME = os.getenv("STRATEGIST_MODEL_NAME", "deepseek/deepseek-r1-0528:free")
WORKER_MODEL_NAME = os.getenv("WORKER_MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free")
IMPROVER_MODEL_NAME = os.getenv("IMPROVER_MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free")

ENABLE_NRPA = os.getenv("NRPA_ENABLED", "1") in ("1", "true", "True", "yes", "YES")

# Providers
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter").lower()

# OpenRouter
API_URL_BASE = "https://openrouter.ai/api/v1/chat/completions"

# Cerebras
CEREBRAS_MODEL_DEFAULT = os.getenv("CEREBRAS_MODEL_DEFAULT", "llama-4-scout-17b-16e-instruct")

# Gemini (Google Generative AI)
# Base URL prefix; final endpoint will be models/{model}:generateContent
GEMINI_API_URL_BASE = os.getenv("GEMINI_API_URL_BASE", "https://generativelanguage.googleapis.com/v1beta")
