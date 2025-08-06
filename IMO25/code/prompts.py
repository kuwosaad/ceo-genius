"""
prompts.py

Prompt templates and robust parsers for the CEO-Genius and NRPA pipeline.

Overview
- Defines the Strategist enumeration prompt that asks for 3–5 concise, labeled strategies.
- Defines the Worker sketch prompt used for cheap simulations during NRPA iterations.
- Defines the Lightweight Verifier prompt that returns a strict JSON viability score.
- Provides parser helpers that are resilient to model formatting drift (list parsing, JSON extraction).

Design Goals
- Deterministic, structured prompts to stabilize downstream parsing.
- Strict output contracts where useful (JSON scoring), with fallback parsing to prevent crashes.
- Minimal surface area so other modules (agent.py) can import and use without additional logic.

Usage
- STRATEGIST_ENUM_PROMPT: called by enumerate_initial_strategies in agent.py
- WORKER_SKETCH_PROMPT: used by run_strategic_simulation for short rollouts
- LIGHTWEIGHT_VERIFIER_PROMPT: used by lightweight_score_sketch to get {"score", "reason"}
- parse_strategies_list, parse_viability_score: tolerant parsing functions feeding NRPA loop.
"""
# Prompt templates and lightweight scoring utilities for NRPA

# Strategist prompt to enumerate 3–5 high-level strategies
STRATEGIST_ENUM_PROMPT = """
You are an IMO problem strategist ("CEO"). Enumerate 3–5 distinct, high-level strategic paths to approach the problem.
Each path must be a concise one-liner, starting with a short label followed by a colon and a description.
Examples: "Proof by Induction: ...", "Geometric Inversion: ...", "Proof by Contradiction: ...", "Algebraic Simplification: ...".
Return ONLY a JSON array of strings. Each string must be a concise, unique strategic approach. Do not include conversational text.

Example response:
["Proof by Induction: Use induction on n", "Geometric Inversion: Apply inversion with respect to unit circle", "Proof by Contradiction: Assume no such configuration exists"]

Problem:
{problem_statement}
"""

# Strategy refinement prompt for NRPA
STRATEGY_REFINEMENT_PROMPT = """
You are an IMO strategist. Given a partial strategy path, propose 3–5 specific refinements or next steps.

The current path is:
{path_prefix}

Each refinement should be a concise one-liner that narrows the focus or adds a specific technique.
Return ONLY a JSON array of strings. Each string must be a unique refinement. No conversational text.

Example response:
["Apply Cauchy-Schwarz inequality to bound the expression", "Consider parity cases for n", "Introduce auxiliary variables to simplify the recurrence"]
"""

# Worker prompt to produce a brief, high-level proof sketch under constraints
WORKER_SKETCH_PROMPT = """
You are an IMO "Worker" asked to perform a short, targeted simulation for a strategic path.

Constraints:
- Produce a HIGH-LEVEL proof sketch (not a full proof).
- Emphasize approach, key lemmas you would try, and feasibility signals.
- Avoid long calculations; focus on the plan's viability.
- Keep it brief and structured (<= ~500 tokens suggested).
- If you propose code, keep it minimal (not required for sketch).

Problem:
{problem_statement}

Selected Strategic Path:
{path_description}

Output:
- Sketch: 5–12 sentences describing the approach.
- Key Lemmas/Checks (bulleted).
- Risks/Unknowns (bulleted).
"""

# Lightweight verifier prompt to score viability of a sketch
LIGHTWEIGHT_VERIFIER_PROMPT = """
You are a ruthless critic. Your task is to find flaws. Score this sketch from 0.0 to 1.0. A score of 1.0 is reserved for a sketch that presents a clear, novel, and mathematically sound path to a solution. A mere restatement of the problem is an immediate 0.0. Explain your score in one sentence.

Return ONLY a single JSON object in this exact format:
{"score": 0.5, "reason": "short one-sentence justification"}

Example response:
{"score": 0.8, "reason": "Good approach but missing key lemma"}

Sketch to score:
{sketch}
"""

# Fallback parser hints for extracting list items or JSON-like content
def parse_strategies_list(text: str) -> list[str]:
    """
    Extract 3–5 strategy lines from text with resilience to format drift.
    
    Now expects JSON array format from updated STRATEGIST_ENUM_PROMPT.
    If JSON parsing fails, falls back to extracting lines with colons.

    Returns:
      A list of up to 5 strategy strings.
    """
    import json
    text = text or ""
    
    # First try to parse as JSON array
    try:
        strategies = json.loads(text)
        if isinstance(strategies, list) and all(isinstance(s, str) for s in strategies):
            # Filter to reasonable length strategies
            cleaned = [s.split(":",1)[1].strip() if ":" in s else s for s in strategies]
            filtered = [s for s in cleaned if 1 <= len(s) <= 200]
            if len(filtered) >= 2:
                return filtered[:5]
    except Exception:
        pass
    
    # Fallback: extract lines (for backward compatibility)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Remove numbering if present (e.g., "1. Label: description" -> "Label: description")
    cleaned_lines = []
    for line in lines:
        # Remove leading numbers and periods
        cleaned_line = line
        if line and line[0].isdigit():
            # Find the first colon after the number
            colon_idx = line.find(':')
            if colon_idx > 0:
                cleaned_line = line[colon_idx+1:].strip()
        cleaned_lines.append(cleaned_line)
    
    # Prefer lines that look like "Label: description"
    labeled = [ln for ln in cleaned_lines if ":" in ln and len(ln.split(":", 1)[0].strip()) <= 60]
    if 2 <= len(labeled) <= 8:
        return labeled[:5]
    # Last fallback: first 5 non-empty lines
    return cleaned_lines[:5]


def parse_viability_score(text: str) -> tuple[float, str]:
    """
    Parse {"score": x.xx, "reason": "..."} from model output with robust fallbacks.

    Strategy:
    1) Attempt strict JSON parsing.
    2) Search for a JSON-like fragment containing "score".
    3) Try loose key extraction even if the value types are strings.
    4) As a last resort, scan for a float in [0.0, 1.0] anywhere in the text.

    Returns:
      (score, reason) where score is clamped to [0.0, 1.0] and reason is a string (possibly empty).
    Never raises; always returns a tuple.
    """
    import json, re
    text = text or ""

    # Helper: clamp & coerce
    def norm_score(x):
        try:
            try:
                v = float(x)
            except Exception:
                return 0.5
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0

    # 1) Strict JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            score = norm_score(data.get("score", 0.0))
            reason = str(data.get("reason", "") or "No reason provided").strip()
            return score, reason
    except Exception:
        pass

    # 2) JSON-like fragment containing "score"
    try:
        match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            frag = match.group(0)
            try:
                data = json.loads(frag)
                if isinstance(data, dict):
                    score = norm_score(data.get("score", 0.0))
                    reason = str(data.get("reason", "") or "No reason provided").strip()
                    return score, reason
            except Exception:
                try:
                    score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', frag)
                    reason_match = re.search(r'"reason"\s*:\s*"(.*?)"', frag, flags=re.DOTALL)
                    score = norm_score(score_match.group(1)) if score_match else 0.0
                    reason = (reason_match.group(1).strip() if reason_match else "No reason provided").strip()
                    return score, reason
                except Exception:
                    pass
    except Exception:
        pass

    # 3) Loose key extraction anywhere in text
    try:
        score_match = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE)
        reason_match = re.search(r'"?reason"?\s*[:=]\s*"?([^"\n]+)"?', text, flags=re.IGNORECASE)
        if score_match:
            score = norm_score(score_match.group(1))
            reason = (reason_match.group(1).strip() if reason_match else text).strip()
            return score, reason
    except Exception:
        pass

    # 4) Fallback heuristic: any number in [0,1]
    try:
        for m in re.finditer(r"(\d+(\.\d+)?)", text):
            try:
                v = float(m.group(1))
                if 0.0 <= v <= 1.0:
                    return v, ""
            except Exception:
                continue
    except Exception:
        pass
        
    return 0.5, text
