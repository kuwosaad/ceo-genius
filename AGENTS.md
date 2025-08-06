# AGENTS.md

Build, Lint, Test
- Python >=3.10; install deps: pip install -r requirements.txt (or: pip install requests python-dotenv pytest; optional: cerebras-cloud-sdk)
- Run agent (single): cd IMO25/code && python agent.py ../problems/imo01.txt --verbose
- Parallel: cd IMO25/code && python run_parallel.py ../problems/imo01.txt -n 8 -w 8 -t 1800 -d logs
- Tests (all): cd IMO25 && pytest -q
- Single test file:: cd IMO25 && pytest tests/test_prompts_parsers.py -q
- Single test + node:: cd IMO25 && pytest tests/test_mcts_utils.py -q -k "uct"

Progress Tracking
- Scratchpad: SCRATCHPAD.md (created automatically for task tracking)

Conventions
- Formatting: PEP8; prefer black (line length 100) and ruff for lint if present; keep imports grouped (stdlib, third-party, local) and sorted.
- Types: Add typing for public functions/classes; use from __future__ import annotations; prefer dataclasses for simple structures; avoid Any; use Optional, TypedDict when parsing JSON.
- Naming: snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants; file names snake_case; tests mirror module names.
- Imports: Relative within package (from .mcts_utils import ... when inside code/); absolute from top-level elsewhere (IMO25.code.*). Do not import heavy SDKs unless MODEL_PROVIDER requires; gate cerebras imports with try/except.
- Errors: Never crash on provider issues; return structured error objects or raise ValueError with clear message; handle HTTP 429 with backoff; log via telemetry/log files; avoid printing secrets.
- Logging: Use timestamped logs in logs/ and code/logs/; emit MCTS_START/ITERATION/END via telemetry_ext; keep logs concise and structured.
- Tests: Prefer pytest; small, deterministic, no network; test parsers and MCTS math; use -k to filter; seed randomness when applicable.
- Config: Read env via python-dotenv; keys: MODEL_PROVIDER, *_API_KEY; safe defaults; feature flags via env (MCTS_GENIUS=1/0).
- PR/Commits: Concise messages focused on why; no secrets; include tests for behavior changes.

Cursor/Copilot/CLINE rules
- Follow .clinerules/CLINE.md process (plan → execute → reflect; cautious ops).
- No .cursor/rules or .github/copilot-instructions.md found; if added later, follow them.
