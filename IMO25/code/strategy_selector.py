from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any

from .logging_utils import log_print
from .nrpa import run_nrpa, NRPA_LEVELS, NRPA_ITER, NRPA_ALPHA, NRPA_MAX_DEPTH


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

        cache: Dict[str, Any] = {}

        def children_provider(step: int, prefix: Tuple[str, ...]) -> List[str]:
            if step == 0:
                return strategies
            return generate_refinements(list(prefix), problem_statement, cache, telemetry)

        def score_fn(seq: List[str]) -> float:
            if not seq:
                return 0.0
            path_description = " -> ".join(seq)
            sketch = run_strategic_simulation(path_description, problem_statement, telemetry)
            score, reason = lightweight_score_sketch(sketch, telemetry)
            log_print(f"[NRPA] Scored sequence: {path_description[:100]}... -> {score:.3f} ({reason[:50]})")
            return score

        log_print(
            f"[NRPA] Starting search: L={NRPA_LEVELS}, N={NRPA_ITER}, Alpha={NRPA_ALPHA}, MaxDepth={NRPA_MAX_DEPTH}"
        )
        if telemetry:
            from .telemetry_ext import nrpa_start, nrpa_end

            nrpa_start(
                telemetry,
                {
                    "num_candidates": len(strategies),
                    "levels": NRPA_LEVELS,
                    "iterations": NRPA_ITER,
                    "alpha": NRPA_ALPHA,
                    "max_depth": NRPA_MAX_DEPTH,
                },
            )
        best_score, best_seq = run_nrpa(
            levels=NRPA_LEVELS,
            iterations=NRPA_ITER,
            alpha=NRPA_ALPHA,
            initial_strategies=strategies,
            children_provider=children_provider,
            score_fn=score_fn,
            cache=cache,
        )
        if telemetry:
            from .telemetry_ext import nrpa_end

            nrpa_end(telemetry, {"best_score": best_score})
        chosen = " -> ".join(best_seq) if best_seq else strategies[0]
        log_print(f"[NRPA] Best sequence (score={best_score:.3f}): {chosen}")
        return chosen
