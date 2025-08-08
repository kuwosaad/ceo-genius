from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any

from .logging_utils import log_print
from .nrpa import run_nrpa, NRPA_LEVELS, NRPA_ITER, NRPA_ALPHA, NRPA_MAX_DEPTH, NRPAConfig


class StrategySelector:
    """Abstract base class for strategy selection."""

    def __init__(self, api_client_funcs: Dict[str, Callable], strategist_model_name: str):
        self.api_client_funcs = api_client_funcs
        self.strategist_model_name = strategist_model_name

    def select_strategy(
        self,
        problem_statement: str,
        other_prompts: List[str],
        system_prompt: str,
        telemetry=None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError


class SingleStrategySelector(StrategySelector):
    """Strategy selector that makes a single call to the CEO."""

    def select_strategy(
        self,
        problem_statement: str,
        other_prompts: List[str],
        system_prompt: str,
        telemetry=None,
        **_: Any,
    ) -> str:
        log_print("[STRATEGY] Using single strategy selection.")
        strategist_payload = self.api_client_funcs["build_request_payload"](
            system_prompt=system_prompt,
            question_prompt=problem_statement,
            other_prompts=other_prompts,
        )
        strategist_response = self.api_client_funcs["send_api_request"](
            self.api_client_funcs["get_api_key"]("strategist"),
            strategist_payload,
            model_name=self.strategist_model_name,
            agent_type="strategist",
            telemetry=telemetry,
        )
        return self.api_client_funcs["extract_text_from_response"](strategist_response)


class NRPAStrategySelector(StrategySelector):
    """Strategy selector that runs the full NRPA loop."""

    def select_strategy(
        self,
        problem_statement: str,
        other_prompts: List[str],
        system_prompt: str,
        telemetry=None,
        enumerate_initial_strategies: Optional[Callable[[str, List[str]], List[str]]] = None,
        generate_refinements: Optional[Callable[[List[str], str, Dict[str, Any], Any], List[str]]] = None,
        run_strategic_simulation: Optional[Callable[[str, str, Any], str]] = None,
        lightweight_score_sketch: Optional[Callable[[str, Any], Tuple[float, str]]] = None,
        **_: Any,
    ) -> str:
        log_print("[NRPA] Starting Strategist with NRPA Strategy Search...")
        if not (
            enumerate_initial_strategies
            and generate_refinements
            and run_strategic_simulation
            and lightweight_score_sketch
        ):
            raise ValueError("NRPAStrategySelector requires strategy generation and scoring functions")

        strategies = enumerate_initial_strategies(problem_statement, other_prompts)
        if not strategies:
            log_print("[NRPA] No strategies returned; falling back to original strategist flow.")
            fallback = SingleStrategySelector(self.api_client_funcs, self.strategist_model_name)
            return fallback.select_strategy(problem_statement, other_prompts, system_prompt, telemetry=telemetry)

        # Persistent, file-backed cache configuration
        import threading
        import os
        import json
        import hashlib

        cache: Dict[str, Any] = {}
        cache["nrpa_usage"] = {"refine_calls": 0, "score_calls": 0}
        cache["lock"] = threading.Lock()

        cache_enabled = os.getenv("NRPA_CACHE_ENABLED", "1") in ("1", "true", "True", "yes", "YES")
        cache_dir = os.getenv("NRPA_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))
        os.makedirs(cache_dir, exist_ok=True)

        def _problem_hash(text: str) -> str:
            return hashlib.md5(text.encode()).hexdigest()

        problem_h = _problem_hash(problem_statement)
        refine_store_path = os.path.join(cache_dir, f"refine_{problem_h}.json")
        score_store_path = os.path.join(cache_dir, f"score_{problem_h}.json")

        # Load stores if enabled
        refine_store: Dict[str, Any] = {}
        score_store: Dict[str, Any] = {}
        if cache_enabled:
            try:
                if os.path.exists(refine_store_path):
                    with open(refine_store_path, "r", encoding="utf-8") as f:
                        refine_store = json.load(f)
            except Exception:
                refine_store = {}
            try:
                if os.path.exists(score_store_path):
                    with open(score_store_path, "r", encoding="utf-8") as f:
                        score_store = json.load(f)
            except Exception:
                score_store = {}

        def children_provider(step: int, prefix: Tuple[str, ...]) -> List[str]:
            if step == 0:
                return strategies
            key = "|".join(prefix)
            if cache_enabled and key in refine_store:
                return list(refine_store.get(key, []))
            result = generate_refinements(list(prefix), problem_statement, cache, telemetry)
            if cache_enabled:
                with cache["lock"]:
                    # Limit entries to keep size reasonable
                    if len(refine_store) < 5000:
                        refine_store[key] = result
            # Budget accounting
            usage = cache.get("nrpa_usage")
            if usage is not None:
                usage["refine_calls"] = int(usage.get("refine_calls", 0)) + 1
            return result

        def score_fn(seq: List[str]) -> float:
            if not seq:
                return 0.0
            path_description = " -> ".join(seq)
            # Cache by path description
            if cache_enabled and path_description in score_store:
                score_reason = score_store[path_description]
                score = float(score_reason.get("score", 0.0))
                reason = str(score_reason.get("reason", ""))
            else:
                sketch = run_strategic_simulation(path_description, problem_statement, telemetry)
                score, reason = lightweight_score_sketch(sketch, telemetry)
                if cache_enabled:
                    with cache["lock"]:
                        if len(score_store) < 5000:
                            score_store[path_description] = {"score": score, "reason": reason}
            # Budget accounting
            usage = cache.get("nrpa_usage")
            if usage is not None:
                usage["score_calls"] = int(usage.get("score_calls", 0)) + 1
            log_print(f"[NRPA] Scored sequence: {path_description[:100]}... -> {score:.3f} ({reason[:50]})")
            return score

        # Build configuration (env-driven with defaults)
        config = NRPAConfig.from_env()

        log_print(
            f"[NRPA] Starting search: L={config.levels}, N={config.iterations}, Alpha={config.alpha}, MaxDepth={config.max_depth}, Temp={config.temperature}"
        )
        if telemetry:
            from .telemetry_ext import nrpa_start, nrpa_end

            nrpa_start(
                telemetry,
                {
                    "num_candidates": len(strategies),
                    "levels": config.levels,
                    "iterations": config.iterations,
                    "alpha": config.alpha,
                    "max_depth": config.max_depth,
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "patience": config.patience,
                    "beam_width": config.beam_width,
                    "max_workers": config.max_workers,
                },
            )
        best_score, best_seq = run_nrpa(
            config=config,
            initial_strategies=strategies,
            children_provider=children_provider,
            score_fn=score_fn,
            cache=cache,
            telemetry=telemetry,
        )
        # Persist a context snapshot for cross-session sharing
        try:
            from .context_store import ContextStore

            ctx = ContextStore.from_env()
            snapshot = {
                "problem_hash": hashlib.md5(problem_statement.encode()).hexdigest(),
                "num_candidates": len(strategies),
                "config": {
                    "levels": config.levels,
                    "iterations": config.iterations,
                    "alpha": config.alpha,
                    "max_depth": config.max_depth,
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "patience": config.patience,
                    "max_seconds": config.max_seconds,
                    "max_calls": config.max_calls,
                    "beam_width": config.beam_width,
                    "max_workers": config.max_workers,
                },
                "usage": cache.get("nrpa_usage", {}),
                "best": {
                    "score": best_score,
                    "sequence": best_seq,
                    "chosen": " -> ".join(best_seq) if best_seq else (strategies[0] if strategies else ""),
                },
            }
            ctx.save_snapshot(snapshot)
        except Exception:
            pass
        # Flush persistent caches
        if cache_enabled:
            try:
                with open(refine_store_path, "w", encoding="utf-8") as f:
                    json.dump(refine_store, f, indent=2)
                with open(score_store_path, "w", encoding="utf-8") as f:
                    json.dump(score_store, f, indent=2)
            except Exception:
                pass
        if telemetry:
            from .telemetry_ext import nrpa_end

            nrpa_end(telemetry, {"best_score": best_score})
        chosen = " -> ".join(best_seq) if best_seq else strategies[0]
        log_print(f"[NRPA] Best sequence (score={best_score:.3f}): {chosen}")
        return chosen
