from __future__ import annotations
import json
from typing import List, Dict, Any, Tuple, Optional

try:
    from .prompts import (
        STRATEGIST_ENUM_PROMPT,
        WORKER_SKETCH_PROMPT,
        LIGHTWEIGHT_VERIFIER_PROMPT,
        STRATEGY_REFINEMENT_PROMPT,
        parse_strategies_list,
        parse_viability_score,
    )
    from .telemetry_ext import nrpa_start, nrpa_iteration, nrpa_end
    from .nrpa import run_nrpa, NRPA_LEVELS, NRPA_ITER, NRPA_ALPHA, NRPA_MAX_DEPTH
    from .config import STRATEGIST_MODEL_NAME, WORKER_MODEL_NAME, IMPROVER_MODEL_NAME
    from .api_utils import (
        build_request_payload,
        send_api_request,
        extract_text_from_response,
        get_api_key,
    )
except ImportError:
    from prompts import (
        STRATEGIST_ENUM_PROMPT,
        WORKER_SKETCH_PROMPT,
        LIGHTWEIGHT_VERIFIER_PROMPT,
        STRATEGY_REFINEMENT_PROMPT,
        parse_strategies_list,
        parse_viability_score,
    )
    from telemetry_ext import nrpa_start, nrpa_iteration, nrpa_end
    from nrpa import run_nrpa, NRPA_LEVELS, NRPA_ITER, NRPA_ALPHA, NRPA_MAX_DEPTH
    from config import STRATEGIST_MODEL_NAME, WORKER_MODEL_NAME, IMPROVER_MODEL_NAME
    from api_utils import (
        build_request_payload,
        send_api_request,
        extract_text_from_response,
        get_api_key,
    )


class StrategySelector:
    """Abstract base class for strategy selection."""

    def __init__(self, api_client_funcs, strategist_system_prompt: str):
        self.api_client_funcs = api_client_funcs
        self.strategist_system_prompt = strategist_system_prompt

    def select_strategy(
        self, problem_statement: str, other_prompts, telemetry=None
    ) -> str:
        raise NotImplementedError


class SingleStrategySelector(StrategySelector):
    """Strategy selector that makes a single call to the CEO."""

    def select_strategy(
        self, problem_statement: str, other_prompts, telemetry=None
    ) -> str:
        print("[STRATEGY] Using single strategy selection.")
        strategist_payload = self.api_client_funcs["build_request_payload"](
            system_prompt=self.strategist_system_prompt,
            question_prompt=problem_statement,
            other_prompts=other_prompts,
        )
        strategist_response = self.api_client_funcs["send_api_request"](
            get_api_key("strategist"),
            strategist_payload,
            model_name=STRATEGIST_MODEL_NAME,
            agent_type="strategist",
            telemetry=telemetry,
        )
        return self.api_client_funcs["extract_text_from_response"](strategist_response)


class NRPAStrategySelector(StrategySelector):
    """Strategy selector that runs the full NRPA loop."""

    def select_strategy(
        self, problem_statement: str, other_prompts, telemetry=None
    ) -> str:
        print("[NRPA] Starting Strategist with NRPA Strategy Search...")
        strategies = enumerate_initial_strategies(
            problem_statement, other_prompts, self.strategist_system_prompt
        )
        if not strategies:
            print(
                "[NRPA] No strategies returned; falling back to original strategist flow."
            )
            return self._fallback_strategy(problem_statement, other_prompts, telemetry)
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
            print(
                f"[NRPA] Scored sequence: {path_description[:100]}... -> {score:.3f} ({reason[:50]})"
            )
            return score

        print(
            f"[NRPA] Starting search: L={NRPA_LEVELS}, N={NRPA_ITER}, Alpha={NRPA_ALPHA}, MaxDepth={NRPA_MAX_DEPTH}"
        )
        if telemetry:
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
        chosen = " -> ".join(best_seq) if best_seq else strategies[0]
        print(f"[NRPA] Best sequence (score={best_score:.3f}): {chosen}")
        if telemetry:
            nrpa_end(telemetry, {"best_score": best_score, "best_sequence": chosen})
        return chosen

    def _fallback_strategy(self, problem_statement, other_prompts, telemetry):
        strategist_payload = self.api_client_funcs["build_request_payload"](
            system_prompt=self.strategist_system_prompt,
            question_prompt=problem_statement,
            other_prompts=other_prompts,
        )
        strategist_response = self.api_client_funcs["send_api_request"](
            get_api_key("strategist"),
            strategist_payload,
            model_name=STRATEGIST_MODEL_NAME,
            agent_type="strategist",
            telemetry=telemetry,
        )
        return self.api_client_funcs["extract_text_from_response"](strategist_response)


def run_strategic_simulation(
    path_description: str, problem_statement: str, telemetry=None
) -> str:
    print(f"[NRPA] Running strategic simulation for path: {path_description}")
    worker_prompt = WORKER_SKETCH_PROMPT.format(
        problem_statement=problem_statement[:4000], path_description=path_description
    )
    payload = build_request_payload(
        system_prompt="",
        question_prompt=worker_prompt,
        temperature=0.2,
        top_p=0.95,
        max_tokens=800,
    )
    resp = send_api_request(
        get_api_key("worker"),
        payload,
        model_name=WORKER_MODEL_NAME,
        agent_type="worker",
        telemetry=telemetry,
    )
    sketch_text = extract_text_from_response(resp)
    print(f"[NRPA] DEBUG: Generated sketch (first 500 chars): {sketch_text[:500]}")
    return sketch_text


def generate_refinements(
    path_prefix: List[str],
    problem_statement: str,
    cache: Dict[str, Any],
    telemetry=None,
) -> List[str]:
    from hashlib import md5

    cache_key = md5(f"refine::{'|'.join(path_prefix)}".encode()).hexdigest()
    if cache_key in cache:
        print(
            f"[NRPA] Using cached refinements for prefix: {' -> '.join(path_prefix[:2])}"
        )
        return cache[cache_key]
    prefix_text = " -> ".join(path_prefix) if path_prefix else "(initial strategies)"
    print(f"[NRPA] Generating refinements for prefix: {prefix_text}")
    prompt = STRATEGY_REFINEMENT_PROMPT.format(path_prefix=prefix_text)
    payload = build_request_payload(
        system_prompt="",
        question_prompt=prompt,
        temperature=0.3,
        top_p=0.9,
        max_tokens=600,
    )
    resp = send_api_request(
        get_api_key("strategist"),
        payload,
        model_name=STRATEGIST_MODEL_NAME,
        agent_type="strategist",
        telemetry=telemetry,
    )
    text = extract_text_from_response(resp)
    refinements = parse_strategies_list(text)
    seen = set()
    unique: List[str] = []
    for r in refinements:
        if r not in seen and len(r) > 10 and len(r) < 200:
            seen.add(r)
            unique.append(r)
    result = unique[:5]
    cache[cache_key] = result
    print(f"[NRPA] Generated {len(result)} refinements")
    return result


def lightweight_score_sketch(sketch: str, telemetry=None):
    prompt = LIGHTWEIGHT_VERIFIER_PROMPT.replace("{sketch}", sketch)
    payload = build_request_payload(
        system_prompt="",
        question_prompt=prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=200,
    )
    resp = send_api_request(
        get_api_key("verifier"),
        payload,
        model_name=IMPROVER_MODEL_NAME,
        agent_type="verifier",
        telemetry=telemetry,
    )
    text = ""
    try:
        text = extract_text_from_response(resp)
    except Exception as e:
        text = f"(no-text; extract error: {e})"
    print(f"[NRPA] DEBUG: Verifier response text: {text[:500]}")
    score = 0.0
    reason = ""
    parsed_score = None
    parsed_reason = None
    try:
        print(
            f"[NRPA] DEBUG: Attempting to parse viability score from text: {text[:200]}"
        )
        parsed_score, parsed_reason = parse_viability_score(text or "")
        print(
            f"[NRPA] DEBUG: parse_viability_score returned: score={parsed_score}, reason={parsed_reason}"
        )
        try:
            score = float(parsed_score)
        except Exception as e:
            print(
                f"[NRPA] DEBUG: Failed to convert parsed_score to float: {parsed_score}, error: {e}"
            )
            score = 0.0
        if score < 0.0 or score > 1.0:
            print(f"[NRPA] DEBUG: Score out of range [0,1]: {score}")
            score = 0.0
        reason = (parsed_reason or "").strip()
    except Exception as e:
        print(f"[NRPA] DEBUG: Exception in parse_viability_score: {e}")
        score = 0.0
        reason = f"parse error: {e}"
    if not reason:
        snippet = (text or "").strip()
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        reason = snippet or "no reason"
    if score == 0.0 and "parse error" in reason:
        print(
            f"[NRPA] DEBUG: Failed to parse viability score from verifier response:"
        )
        print(f"[NRPA] DEBUG: Full response text: {text}")
        print(
            f"[NRPA] DEBUG: Parsed score: {parsed_score}, reason: {parsed_reason}"
        )
    print(f"[NRPA] Lightweight score: {score:.3f}. Reason: {reason}")
    return score, reason


def enumerate_initial_strategies(
    problem_statement: str, other_prompts, strategist_system_prompt: str
):
    enum_prompt = STRATEGIST_ENUM_PROMPT.format(problem_statement=problem_statement)
    payload = build_request_payload(
        system_prompt=strategist_system_prompt,
        question_prompt=enum_prompt,
        other_prompts=other_prompts,
        temperature=0.2,
        top_p=0.9,
        max_tokens=600,
    )
    resp = send_api_request(
        get_api_key("strategist"),
        payload,
        model_name=STRATEGIST_MODEL_NAME,
        agent_type="strategist",
    )
    text = extract_text_from_response(resp)
    strategies = parse_strategies_list(text)
    print("[NRPA] Initial strategies:")
    for s in strategies:
        print(f" - {s}")
    return strategies
