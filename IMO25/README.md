
# IMO25 — CEO–Genius Multi‑Agent Solver with NRPA Strategy Search

Research‑grade framework orchestrating multiple LLM "roles" to solve International Mathematical Olympiad (IMO) problems, enhanced with Nested Rollout Policy Adaptation (NRPA) for intelligent strategy exploration and learning.

Executive Summary
- Roles: Strategist (CEO) → Worker (Genius) → Improver → Verifier.
- NRPA-Enhanced Strategy Search: Before committing to a single plan, enumerate 3–5 strategies, then use Nested Rollout Policy Adaptation to explore strategy sequences, learn what works, and choose the best path through adaptive policy learning.
- Telemetry: Detailed timing, iteration counts, pass/fail stats, strategy changes, and NRPA event logs.
- Parallelization: Optionally launch many independent agents to improve the chance of success.
- Clean Logging: Sequential log naming (IMO1, IMO2, etc.) in a single consolidated directory.

Why this project?
Explaining math proofs with LLMs is brittle if you pursue only one plan. This project adds an intelligent strategic search layer (NRPA-Enhanced) that learns what mathematical approaches work well together, explores strategy sequences adaptively, and focuses computational resources on the most promising paths.

Key Idea
1) Enumerate initial strategies (e.g., Induction, Inversion, Contradiction).
2) Use Nested Rollout Policy Adaptation (NRPA) to explore strategy sequences:
   - Build multi-step strategy paths (e.g., "Induction → Focus on base cases → Bound the inductive step")
   - Run short Worker "sketch" rollouts for each sequence
   - Score sketch viability with a lightweight verifier on [0.0, 1.0]
   - Adaptively learn which strategy combinations work well together
3) Choose the best strategy sequence and hand it to the classical Worker → Improver → Verifier pipeline.

## System Architecture

High‑level pipeline:
1) Strategist (CEO): Produces a high‑level plan, not a full solution.
2) Worker (Genius): Expands the plan into a detailed solution draft.
3) Improver: Self‑corrects, fills gaps, strengthens rigor.
4) Verifier: Acts as a grader; flags critical errors or justification gaps.

NRPA-Enhanced Strategy Search:
- enumerate_initial_strategies: Ask Strategist for 3–5 one‑liner plans.
- generate_refinements: For each strategy path, generate specific refinements and next steps.
- run_strategic_simulation: Short Worker rollout per selected path to produce a sketch.
- lightweight_score_sketch: Score sketch via a strict JSON viability prompt (score + reason).
- NRPA loop: softmax policy rollout → simulate → score → adapt policy weights; choose best sequence at the end.

Components
- code/agent.py: Orchestrator, API routing (OpenRouter/Cerebras), telemetry, backtracking, CLI.
- code/nrpa.py: Nested Rollout Policy Adaptation implementation (policy, rollout, adapt, run_nrpa).
- code/nrpa.py: Nested Rollout Policy Adaptation implementation (policy, rollout, adapt, run_nrpa).
- code/prompts.py: Strategist/Worker/Verifier prompts and robust parsers (list/JSON).
- code/telemetry_ext.py: Non‑invasive NRPA telemetry wrappers (START/ITERATION/END).
- code/run_parallel.py: Parallel runner that launches many agents and aggregates results.

## Data Flow and Prompts

- Strategist prompt (STRATEGIST_ENUM_PROMPT): Returns 3–5 labeled one‑liner strategies. Parser is tolerant to minor format drift.
- Worker sketch prompt (WORKER_SKETCH_PROMPT): Produces a short proof sketch (approach, key lemmas, risks).
- Lightweight verifier prompt (LIGHTWEIGHT_VERIFIER_PROMPT): Returns strict JSON {"score": float, "reason": "..."}; parser falls back to fragments/numeric heuristic.
- Full Worker prompt (step1_prompt + worker_prompt_template): Solicits a complete, rigorous solution with required structure and optional Python checks for constructions.
- Verifier (verification_system_prompt): Generates a Detailed Verification Log and a summary verdict. A separate yes/no follow‑up reduces parsing ambiguity.

## Configuration

Environment variables (.env or shell):
- STRATEGIST_MODEL_NAME: Strategist model (default: deepseek/deepseek-r1-0528:free)
- WORKER_MODEL_NAME: Worker model (default: deepseek/deepseek-chat-v3-0324:free)
- IMPROVER_MODEL_NAME: Improver/Verifier model (default: deepseek/deepseek-chat-v3-0324:free)
- MODEL_PROVIDER: "openrouter" (default) or "cerebras"
- CEO_API_KEY: API key for Strategist/Verifier (OpenRouter)
- GENIUS_API_KEY: API key for Worker (OpenRouter)
- IMPROVER_API_KEY: API key for Improver (falls back to CEO_API_KEY)
- CEREBRAS_API_KEY: API key when MODEL_PROVIDER=cerebras
- CEREBRAS_MODEL_DEFAULT: default Cerebras model id
- NRPA_ENABLED: "1"/"true" to enable NRPA strategy search, "0"/"false" to disable

Advanced NRPA configuration (optional; defaults preserve existing behavior):

- NRPA_LEVELS: nesting depth (default from code)
- NRPA_ITER: iterations per level (default from code)
- NRPA_ALPHA: adaptation step size (default from code)
- NRPA_MAX_DEPTH: maximum sequence depth (default from code)
- NRPA_TEMPERATURE: softmax temperature for policy sampling (default 1.0)
- NRPA_SEED: integer seed for reproducibility (optional)
- NRPA_PATIENCE: early stopping plateau (0 disables; default 0)
- NRPA_MAX_SECONDS: wall-clock budget in seconds (optional)
- NRPA_MAX_CALLS: combined budget for refinement + scoring calls (optional)
- NRPA_BEAM_WIDTH: base-level independent rollouts per iteration (default 1)
- NRPA_MAX_WORKERS: max threads for scoring beam (default 1)
- NRPA_CACHE_ENABLED: enable disk-backed cache for refinements/scores (default on)
- NRPA_CACHE_DIR: directory for cache files (default `IMO25/cache`)

Dependencies:
```bash
pip install requests python-dotenv
# optional:
pip install cerebras-cloud-sdk
# tests:
pip install pytest
```

## Running

Single agent (default provider OpenRouter):
```bash
cd IMO25/code
python agent.py ../problems/imo01.txt --verbose
```

Parallel runs:
```bash
cd IMO25/code
python run_parallel.py ../problems/imo01.txt -n 8 -w 8 -t 1800 -d logs
```

NRPA Strategy Search toggle:
```bash
export NRPA_ENABLED=1   # enable (default)
export NRPA_ENABLED=0   # disable; revert to single-plan flow
```

Provider selection:
```bash
export MODEL_PROVIDER=openrouter
# or
export MODEL_PROVIDER=cerebras
# or
export MODEL_PROVIDER=gemini
# When using Gemini, set your API key and model names, e.g.:
export GEMINI_API_KEY=your_key_here
# Example models: "gemini-1.5-pro" or "gemini-1.5-flash"
export STRATEGIST_MODEL_NAME=gemini-1.5-pro
export WORKER_MODEL_NAME=gemini-1.5-pro
export IMPROVER_MODEL_NAME=gemini-1.5-pro
```
If Cerebras SDK is missing, requests return a structured error note rather than crashing.

## Telemetry and Logs

Single agent logs:
- Sequentially numbered log files (logs/IMO1.log, logs/IMO2.log, etc.).
- Telemetry snapshots as JSON (logs/IMO1_telemetry.json, logs/IMO2_telemetry.json, etc.).
- All logs consolidated in a single directory for easy management.

NRPA events:
- NRPA_START: initial candidates and search parameters
- NRPA_ITERATION: per-iteration selected strategy sequence, score, reason, and policy summary
- NRPA_END: chosen strategy sequence and final policy state

Parallel logs:
- Per‑agent logs in the provided log directory (e.g., logs/agent_00.log).
- Final summary prints success rate and, if found, the winning agent and extracted solution.

## Testing
Basic unit tests exist for:
- NRPA utilities (policy adaptation, rollout scoring)
- Prompt parsers (strategy list parsing, viability JSON parsing)

Run:
```bash
cd IMO25
pytest -q
```

Additional tests cover:
- Reproducibility with fixed seed
- Temperature effect on sampling concentration
- Adaptation increases chosen action probability
- Early stopping by patience on flat landscapes

## Limitations and Future Work
- NRPA search depth and strategy refinement can be further optimized for specific mathematical domains.
- Viability scoring relies on LLM outputs; adding repeated scoring and temperature controls could reduce variance.
- Verification is text‑based; adding programmatic checkers for specific domains (e.g., number theory) would strengthen rigor.
- Cost control can be improved with adaptive iteration counts and early stopping when a path is clearly dominant.

```bash
python agent.py problem.txt [options]
```

With NRPA-Enhanced Strategy Search enabled (default), the run will:
1) Enumerate 3–5 initial strategies
2) Use Nested Rollout Policy Adaptation to explore strategy sequences:
   - Build multi-step strategy paths
   - Perform short Worker rollouts for each sequence
   - Score sketches with a lightweight verifier
   - Adapt policy weights toward promising approaches
3) Choose the best strategy sequence
4) Proceed with the full Worker → Improver → Verifier flow

**Arguments:**
- `problem.txt`: Path to the problem statement file (required); imo2025 problems are in `problems`

**Options:**
- `--log LOG_FILE`: Specify a log file for output (default: prints to console)
- `--other_prompts PROMPTS`: Additional prompts separated by commas

**Example:**
```bash
python agent.py imo2025_p1.txt --log agent_output.log
```

### Parallel Execution (`run_parallel.py`)

Run multiple agents in parallel to increase the chance of finding a solution:

```bash
python run_parallel.py problem.txt [options]
```

**Arguments:**
- `problem.txt`: Path to the problem statement file (required)

**Options:**
- `--num-agents N` or `-n N`: Number of parallel agents (default: 10)
- `--log-dir DIR` or `-d DIR`: Directory for log files (default: logs)
- `--timeout SECONDS` or `-t SECONDS`: Timeout per agent in seconds (default: no timeout)
- `--max-workers N` or `-w N`: Maximum worker processes (default: number of agents)
- `--other_prompts PROMPTS` or `-o PROMPTS`: Additional prompts separated by commas

**Examples:**
```bash
# Run 20 agents with 5-minute timeout each
python run_parallel.py imo2025_p1.txt -n 20 -t 300

# Run 5 agents with custom log directory
python run_parallel.py imo2025_p1.txt -n 5 -d my_logs

# Run with additional prompts
python run_parallel.py imo2025_p1.txt -n 15 -o "focus_on_geometry,use_induction"
```

## Problem Files
See the problems folder (e.g., imo01.txt … imo06.txt). The CLI will resolve a relative path first under the repository problems/ folder, then fall back to the provided path.

## Project Files
- code/agent.py: CEO–Genius orchestrator with NRPA Strategy Search, API routing, telemetry, backtracking, CLI.
- code/nrpa.py: Nested Rollout Policy Adaptation implementation for intelligent strategy exploration.
- code/prompts.py: Prompt templates; robust parsers for strategies list and viability JSON.
- code/telemetry_ext.py: Thin wrappers to emit NRPA telemetry events via TelemetrySystem.
- code/run_parallel.py: Parallel launcher for many agents with status/summary reporting.
- problems/: A small set of sample problem statements.
- tests/: Pytest coverage for NRPA utilities and parsers.

Acknowledgements
- Built on OpenRouter and optionally Cerebras SDK.
- Defaults target DeepSeek models but are configurable.

License
MIT (see headers and LICENSE excerpts in files).

### Single Agent
- Output is printed to console by default
- Use `--log` to save output to a file
- The agent will indicate if a complete solution was found

### NRPA Telemetry

The agent logs NRPA events using the existing telemetry system with richer payloads:
- `NRPA_START`: starting search with N candidate strategies and NRPA parameters
- `NRPA_ITERATION`: iteration index, best_score_so_far, last_score, and policy summary including per-step top‑k actions and entropy (first 3 steps)
- `NRPA_END`: chosen strategy sequence and final policy state

Telemetry is saved as `IMO{N}_telemetry.json` in the `logs` directory with sequential numbering.

### Parallel Execution
- Each agent creates a separate log file in the specified directory
- Progress is shown in real-time
- Final summary shows:
  - Total execution time
  - Number of successful/failed agents
  - Success rate
  - Which agent found a solution (if any)
  - Location of log files

## Understanding the Output

### Solution Detection
The system looks for the phrase "Found a correct solution in run" to identify successful solutions.

### Agent Behavior
- Agents use models from OpenRouter (DeepSeek models by default)
- Each agent follows a structured approach with multiple attempts
- Solutions are verified for completeness and correctness
- Agents can provide partial solutions if complete solutions aren't found

## Tips for Best Results

1. **Problem Formatting**: Ensure your problem file is clear and well-formatted
2. **Parallel Execution**: Use more agents for harder problems (10-20 agents recommended)
3. **Timeout Settings**: Set reasonable timeouts (you may set no timeout)
4. **API Limits**: Be aware of OpenRouter API rate limits and costs
5. **Log Analysis**: Check individual agent logs for detailed reasoning

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your OpenRouter API key is properly set
2. **Timeout Issues**: Increase timeout or reduce number of agents
3. **Memory Issues**: Reduce max-workers if running out of memory
4. **No Solutions Found**: Try running more agents or check problem clarity

### Debug Mode
Add verbose logging by modifying the agent code or check individual log files for detailed output.

## License

MIT License - Copyright (c) 2025 Lin Yang, Yichen Huang

This software is provided as-is. Users are free to copy, modify, and distribute the code with proper attribution.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## Disclaimer

This tool is for educational and research purposes. Success in solving IMO problems depends on problem difficulty and AI model capabilities. Not all problems may be solvable by the current system.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{imo25solver,
  title={IMO Problem Solver using OpenRouter and DeepSeek Models},
  author={Huang, Yichen and Yang, Lin F},
  year={2025}
}
```
