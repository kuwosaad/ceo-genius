"""
agent.py

High-level Orchestrator for the IMO25 CEO-Genius pipeline with NRPA strategy search.

This module coordinates a multi-agent system to solve IMO-style problems using LLMs:
1) Strategist ("CEO") produces a high-level plan.
2) Worker ("Genius") expands the plan into a full solution.
3) Improver refines and self-corrects the solution.
4) Verifier evaluates rigor and flags critical errors or justification gaps.

NRPA Strategy Search (feature-flagged via NRPA_ENABLED=1) sits on top of the Strategist step to enumerate
multiple strategic candidates, run lightweight simulations (sketches) with the Worker model,
score their viability using a lightweight verifier, and choose the best path using Nested Rollout 
Policy Adaptation. Only the selected strategy is then passed to the classic
Worker → Improver → Verifier pipeline to minimize cost while improving search over strategies.

Key capabilities:
- Environment configuration via .env (model provider, API keys per role, model names)
- Provider routing (OpenRouter default; Cerebras optional)
- Robust API interaction with retries, JSON parsing fallback, and conservative timeouts
- TelemetrySystem for metrics/events + non-invasive NRPA telemetry wrappers
- BacktrackingManager to escalate repeated verification failures and request strategy reassessment
- CLI interface to run against a problem file with logging, verbosity, and repeated runs

Logs/Telemetry:
- Timestamped console + log file output
- Telemetry JSON files recording durations, counts, events, and solution state

Main entrypoint:
    python agent.py problems/imo01.txt --verbose
    python agent.py problems/imo01.txt --log logs/run.log --other_prompts "hint1,hint2"

This file is intentionally comprehensive; the core NRPA utilities are isolated in nrpa.py,
prompt templates and parsers in prompts.py, and telemetry wrappers in telemetry_ext.py.
"""

import os
import sys
import json
import argparse
import time
import subprocess
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# NRPA additions and prompt imports
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
# Config and logging imports with fallback
try:
    from .config import (
        STRATEGIST_MODEL_NAME,
        WORKER_MODEL_NAME,
        IMPROVER_MODEL_NAME,
        ENABLE_NRPA,
        API_URL_BASE,
        MODEL_PROVIDER,
        CEREBRAS_MODEL_DEFAULT,
    )
    from .logging_utils import (
        log_print,
        debug_print,
        initialize_logging,
        set_log_file,
        close_log_file,
        set_verbose_mode,
        get_next_log_number,
    )
except ImportError:
    from config import (
        STRATEGIST_MODEL_NAME,
        WORKER_MODEL_NAME,
        IMPROVER_MODEL_NAME,
        ENABLE_NRPA,
        API_URL_BASE,
        MODEL_PROVIDER,
        CEREBRAS_MODEL_DEFAULT,
    )
    from logging_utils import (
        log_print,
        debug_print,
        initialize_logging,
        set_log_file,
        close_log_file,
        set_verbose_mode,
        get_next_log_number,
    )
from .api_utils import (
    get_api_key,
    build_request_payload,
    send_api_request,
    extract_text_from_response,
)
from .strategy_selector import SingleStrategySelector, NRPAStrategySelector

print = log_print


def execute_python_code(code):
    """
    Execute Python code in a subprocess and return the result.
    This provides a safer way to run code snippets for verification, isolating
    runtime errors and timeouts from the main agent process. The subprocess is
    killed after a short timeout to avoid hanging executions from incorrect or
    adversarial code emitted by the Worker/Improver models.
    """
    try:
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Execute the code in a subprocess with a timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        # Clean up the temporary file
        os.unlink(temp_file)
        
        # Return the result
        if result.returncode == 0:
            return {"success": True, "output": result.stdout, "error": None}
        else:
            return {"success": False, "output": result.stdout, "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "Code execution timed out"}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

def initialize_scratchpad(problem_statement):
    """
    Initialize the shared memory scratchpad with problem information.
    """
    # Extract key information from the problem statement
    # This is a simplified version - in practice, we might want to use an LLM to parse this
    scratchpad = f"""--- WORKING THEORY SCRATCHPAD ---
**Problem Statement:** {problem_statement[:200]}... (truncated)

**Key Definitions:**
- S: Set of points (a,b) where a,b are positive integers and a+b ≤ n+1
- T_n: Total number of points in S = n(n+1)/2
- Sunny line: Line not parallel to x-axis, y-axis, or x+y=0

**Proven Facts:**
- None yet established

**Disproven Hypotheses:**
- None yet disproven

**Current Central Obstacle:**
- Need to determine all possible values of k for given n

--- END SCRATCHPAD ---"""
    return scratchpad

def update_scratchpad(scratchpad, new_fact=None, disproven_hypothesis=None, obstacle=None):
    """
    Update the scratchpad with new information.
    """
    lines = scratchpad.split('\n')
    
    # Find sections
    proven_section = -1
    disproven_section = -1
    obstacle_section = -1
    
    for i, line in enumerate(lines):
        if line.startswith('**Proven Facts:**'):
            proven_section = i
        elif line.startswith('**Disproven Hypotheses:**'):
            disproven_section = i
        elif line.startswith('**Current Central Obstacle:**'):
            obstacle_section = i
    
    # Update sections
    if new_fact and proven_section != -1:
        # Insert after the "Proven Facts:" line
        lines.insert(proven_section + 1, f"- {new_fact}")
    
    if disproven_hypothesis and disproven_section != -1:
        # Insert after the "Disproven Hypotheses:" line
        lines.insert(disproven_section + 1, f"- {disproven_hypothesis}")
    
    if obstacle and obstacle_section != -1:
        # Replace the obstacle line
        lines[obstacle_section] = f"**Current Central Obstacle:** {obstacle}"
    
    return '\n'.join(lines)

class TelemetrySystem:
    """
    Tracks and logs agent performance metrics during execution.

    What is measured:
    - total_api_calls and api_call_durations: per-request timing to estimate cost/latency
    - agent_iterations: outer verification-improvement loop cycles
    - verification_passes / verification_failures: verifier outcomes across iterations
    - strategy_changes: number of CEO strategy reassessments triggered by repeated failures
    - solution_found: whether the pipeline converged to a correct solution (stability check)

    The system stores an events list for human-readable milestones (SESSION_START/END,
    STRATEGY_CHANGE, SOLUTION_FOUND) and also NRPA_* events emitted via telemetry_ext.py.
    At session end, a metrics JSON snapshot is persisted under logs/ with sequential naming.
    """
    def __init__(self, log_directory="../logs"):
        self.log_directory = log_directory
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "total_api_calls": 0,
            "api_call_durations": [],
            "agent_iterations": 0,
            "verification_passes": 0,
            "verification_failures": 0,
            "strategy_changes": 0,
            "solution_found": False
        }
        self.events = []
    
    def start_session(self):
        """Mark the start of a session."""
        self.metrics["start_time"] = datetime.now().isoformat()
        self.log_event("SESSION_START", "Telemetry session started")
    
    def end_session(self):
        """Mark the end of a session."""
        self.metrics["end_time"] = datetime.now().isoformat()
        self.log_event("SESSION_END", "Telemetry session ended")
        self.save_metrics()
    
    def log_event(self, event_type, description):
        """Log a significant event during execution."""
        self.events.append({
            "type": event_type,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_api_call(self, duration):
        """Record an API call with its duration."""
        self.metrics["total_api_calls"] += 1
        self.metrics["api_call_durations"].append(duration)
    
    def record_iteration(self):
        """Record an agent iteration."""
        self.metrics["agent_iterations"] += 1
    
    def record_verification_result(self, passed):
        """Record a verification result."""
        if passed:
            self.metrics["verification_passes"] += 1
        else:
            self.metrics["verification_failures"] += 1
    
    def record_strategy_change(self):
        """Record a strategy change."""
        self.metrics["strategy_changes"] += 1
        self.log_event("STRATEGY_CHANGE", "Agent strategy was reassessed by CEO")
    
    def record_solution_found(self):
        """Record that a solution was found."""
        self.metrics["solution_found"] = True
        self.log_event("SOLUTION_FOUND", "Agent found a correct solution")
    
    def save_metrics(self):
        """Save metrics to a JSON file with sequential naming."""
        # Use the same sequential numbering as the main log
        log_number = get_next_log_number() - 1  # Use the same number as the main log
        metrics_file_path = os.path.join(self.log_directory, f"IMO{log_number}_telemetry.json")
        
        # Calculate additional metrics
        total_duration = 0
        if self.metrics["start_time"] and self.metrics["end_time"]:
            start = datetime.fromisoformat(self.metrics["start_time"])
            end = datetime.fromisoformat(self.metrics["end_time"])
            total_duration = (end - start).total_seconds()
        
        avg_api_duration = 0
        if self.metrics["api_call_durations"]:
            avg_api_duration = sum(self.metrics["api_call_durations"]) / len(self.metrics["api_call_durations"])
        
        # Add calculated metrics
        full_metrics = {
            **self.metrics,
            "total_duration_seconds": total_duration,
            "average_api_call_duration": avg_api_duration,
            "events": self.events
        }
        
        try:
            with open(metrics_file_path, 'w') as f:
                json.dump(full_metrics, f, indent=2)
            print(f"[TELEMETRY] Metrics saved to {metrics_file_path}")
        except Exception as e:
            print(f"[TELEMETRY] Error saving metrics: {e}")

class BacktrackingManager:
    """
    Manages strategic backtracking when repeated verification failures occur.

    Behavior:
    - Increments a failure counter on each failed verification
    - Once a threshold is reached (default: 3), escalates to request a new strategy
      from the CEO/Strategist with a failure summary and history for context.
    - Resets when a new strategy is adopted to avoid premature further escalation.
    """
    def __init__(self, max_failures=3):
        self.failure_count = 0
        self.max_failures = max_failures
        self.failure_history = []
    
    def record_failure(self, error_type, context, scratchpad_state):
        """
        Record a failure and return whether escalation to CEO is needed.
        """
        self.failure_count += 1
        self.failure_history.append({
            "type": error_type,
            "context": context,
            "scratchpad_state": scratchpad_state,
            "timestamp": datetime.now().isoformat()
        })
        return self.failure_count >= self.max_failures
    
    def generate_ceo_reassessment_prompt(self, original_strategy, problem_statement):
        """
        Generate a prompt for the CEO to reassess the strategy.
        """
        recent_failures = self.failure_history[-3:]  # Last 3 failures
        failure_summary = "\n".join([
            f"- {f['type']}: {f['context'][:100]}..." for f in recent_failures
        ])
        
        return f"""
STRATEGIC REASSESSMENT REQUEST

The current strategy has failed {self.failure_count} times. 
Please reassess and provide a new approach.

Original Problem:
{problem_statement}

Original Strategy:
{original_strategy}

Recent Failures:
{failure_summary}

Failure History:
{json.dumps(self.failure_history, indent=2)}

Please provide a new strategic plan that addresses these failures.
"""

strategist_system_prompt = """
You are a world-class mathematician and a brilliant strategist. Your role is to act as the "CEO" or "Strategist" in a two-agent team tasked with solving an International Mathematical Olympiad (IMO) problem.

Your task is NOT to solve the problem fully. Instead, you must produce a high-level, conceptual plan that a "Genius" or "Worker" agent can use to construct the detailed, rigorous proof.

Your plan must be clear, insightful, and guide the worker agent effectively. Focus on the core logic and structure of the argument.

Instructions

Deconstruct the Problem: Break down the problem into its core components and constraints.

Identify Key Concepts: Pinpoint the mathematical fields and theorems that are likely relevant (e.g., Number Theory, Euclidean Geometry, Combinatorics, Inequalities).

Propose Methodologies: Suggest one or more promising lines of attack. This could be proof by induction, proof by contradiction, casework, using a specific geometric transformation, or setting up an algebraic system.

Outline the Argument: Provide a step-by-step sketch of the logical flow. This is the most critical part. Define key lemmas or intermediate results that need to be proven.

Be Concise and Strategic: Avoid getting bogged down in detailed calculations. Your output is a blueprint, not the final building.

Output Format

Your response must be a single block of text containing the strategic plan.
"""

worker_prompt_template = """
You are a brilliant mathematician, a "Genius" or "Worker" agent. You have been given a high-level strategic plan from your "CEO" or "Strategist" agent to solve an International Mathematical Olympiad (IMO) problem.

Your task is to take this strategic plan and expand it into a complete and rigorously justified solution. You must follow the provided strategy, filling in all the details, proofs, and calculations necessary.

When proposing constructions or making claims, you MUST provide Python code to verify your constructions. This code will be executed to validate your solution.

The Problem Statement:
{problem_statement}

The Strategic Plan from your CEO:
--- STRATEGY START ---
{strategy}
--- STRATEGY END ---

Working Theory Scratchpad:
{scratchpad}

You must now produce the full solution, strictly following the Core Instructions and Output Format provided to you.
If you propose a construction, include Python code to verify it.
"""

step1_prompt = """

Core Instructions

Rigor is Paramount: Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.

Honesty About Completeness: If you cannot find a complete solution, you must not guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:

Proving a key lemma.

Fully resolving one or more cases within a logically sound case-based proof.

Establishing a critical property of the mathematical objects in the problem.

For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.

Use Markdown for Mathematics: All mathematical variables, expressions, and relations must be enclosed in markdown delimiters (e.g., Let *n* be an integer.).

Output Format

Your response MUST be structured into the following sections, in this exact order.

1. Summary

Provide a concise overview of your findings. This section must contain two parts:

a. Verdict: State clearly whether you have found a complete solution or a partial solution.

For a complete solution: State the final answer, e.g., "I have successfully solved the problem. The final answer is..."

For a partial solution: State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."

b. Method Sketch: Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:

A narrative of your overall strategy.

The full and precise mathematical statements of any key lemmas or major intermediate results.

If applicable, describe any key constructions or case splits that form the backbone of your argument.

2. Detailed Solution

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

Self-Correction Instruction

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.
"""

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt.

When making claims or proposing constructions, provide Python code to verify them. This will help validate your solution.

Working Theory Scratchpad:
{scratchpad}

If you propose a construction, include Python code to verify it.
"""

correction_prompt = """
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.

When making claims or proposing constructions, provide Python code to verify them. This will help validate your solution.

Working Theory Scratchpad:
{scratchpad}

If you propose a construction, include Python code to verify it.
"""

verification_system_prompt = """
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct only if every step is rigorously justified. A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

When evaluating solutions, you should also check any Python code provided for correctness and execution results.

Instructions

1. Core Instructions

Your sole task is to find and report all issues in the provided solution. You must act as a verifier, NOT a solver. Do NOT attempt to correct the errors or fill the gaps you find.

You must perform a step-by-step check of the entire solution. This analysis will be presented in a Detailed Verification Log, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

2. How to Handle Issues in the Solution
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

a. Critical Error:
This is any error that breaks the logical chain of the proof. This includes both logical fallacies (e.g., claiming that A>B, C>D implies A-C>B-D) and factual errors (e.g., a calculation error like 2+3=6).

Procedure:

Explain the specific error and state that it invalidates the current line of reasoning.

Do NOT check any further steps that rely on this error.

You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

b. Justification Gap:
This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.

Procedure:

Explain the gap in the justification.

State that you will assume the step's conclusion is true for the sake of argument.

Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

3. Enhanced Feedback Requirements
For each issue identified, you MUST also provide:
- Suggestion for Fix: How the issue could be resolved
- Alternative Approach: A different method that could be used
- "What-If" Question: A thought-provoking question about the approach

4. Code Verification
If the solution includes Python code:
- Check that the code is syntactically correct
- Verify that the code logically supports the mathematical claims
- If execution results are provided, check that they are consistent with the code

5. Output Format
Your response MUST be structured into two main sections: a Summary followed by the Detailed Verification Log.

a. Summary
This section MUST be at the very beginning of your response. It must contain two components:

Final Verdict: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."

List of Findings: A bulleted list that summarizes every issue you discovered. For each finding, you must provide:

Location: A direct quote of the key phrase or equation where the issue occurs.

Issue: A brief description of the problem and its classification (Critical Error or Justification Gap).

Suggestion for Fix: How the issue could be resolved.

Alternative Approach: A different method that could be used.

"What-If" Question: A thought-provoking question about the approach.

b. Detailed Verification Log
Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, quote the relevant text to make your reference clear before providing your detailed analysis of that part.

Example of the Required Summary Format
This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.

Final Verdict: The solution is invalid because it contains a Critical Error.

List of Findings:

Location: "By interchanging the limit and the integral, we get..."

Issue: Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.

Suggestion for Fix: Provide a proof of uniform convergence or use a different method that doesn't require interchanging limits.

Alternative Approach: Use the Dominated Convergence Theorem if applicable.

"What-If" Question: What if the function sequence doesn't converge uniformly? Would the result still hold?
"""

verification_remider = """

Verification Task Reminder

Your task is to act as an IMO grader. Now, generate the summary and the step-by-step verification log for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""

def read_file_content(filepath):
    """
    Reads and returns the content of a file.
    Exits if the file cannot be read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)


def extract_detailed_solution(solution, marker='Detailed Solution', after=True):
    """
    Extracts the text after '### Detailed Solution ###' from the solution string.
    Returns the substring after the marker, stripped of leading/trailing whitespace.
    If the marker is not found, returns an empty string.
    """
    idx = solution.find(marker)
    if idx == -1:
        return ''
    if(after):
        return solution[idx + len(marker):].strip()
    else:
        return solution[:idx].strip()

def verify_solution(problem_statement, solution, verbose=True):
    """
    Submit the 'Detailed Solution' section of the model output to the Verifier.

    Steps:
    1) Build a verifier prompt that includes the problem and the extracted detailed solution.
    2) Ask the verifier to produce a detailed verification log with a Summary and findings.
    3) Ask a quick follow-up yes/no check to determine if the solution passes.
    4) Return the bug_report (if any) and the yes/no verdict string to drive the loop.

    The follow-up yes/no check intentionally uses a separate short prompt to reduce
    parsing ambiguity and keep the main verifier output rich and human-auditable.
    """
    dsol = extract_detailed_solution(solution)

    newst = f"""
======================================================================

Problem

{problem_statement}

======================================================================

Solution

{dsol}

{verification_remider}
"""
    if(verbose):
        print("[VERIFIER] Start verification.")
    p2 = build_request_payload(system_prompt=verification_system_prompt,
                            question_prompt=newst
                            )
    
    if(verbose):
        print("[VERIFIER] Verification prompt:")
        print(json.dumps(p2, indent=4))
    
    res = send_api_request(get_api_key("verifier"), p2, model_name=IMPROVER_MODEL_NAME, agent_type="verifier")
    out = extract_text_from_response(res) 
    
    if(verbose):
        print("[VERIFIER] Verification results:")
        print(json.dumps(out, indent=4))
    
    check_correctness = """Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?""" \
            + "\n\n" + out 
    prompt = build_request_payload(system_prompt="", question_prompt=check_correctness)
    r = send_api_request(get_api_key("verifier"), prompt, model_name=IMPROVER_MODEL_NAME, agent_type="verifier")
    o = extract_text_from_response(r) 
    
    if(verbose):
        print("[VERIFIER] Is verification good?")
        print(json.dumps(o, indent=4))
        
    bug_report = ""
    
    if("yes" not in o.lower()):
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)
    
    if(verbose):
        print("[VERIFIER] Bug report:")
        print(json.dumps(bug_report, indent=4))
    
    return bug_report, o

def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
"""

    p1 = build_request_payload(system_prompt="",    question_prompt=check_complete_prompt)
    r = send_api_request(get_api_key("improver"), p1, model_name=IMPROVER_MODEL_NAME, agent_type="improver")
    o = extract_text_from_response(r)

    print(o)
    return "yes" in o.lower()

def run_strategic_simulation(path_description: str, problem_statement: str, telemetry=None) -> str:
    """
    Short Worker rollout to gauge a strategy's viability.

    Design:
    - Keep tokens small and temperature low for stability and speed.
    - Produce a structured sketch (approach, key lemmas, risks) rather than a full proof.
    - Used only inside the NRPA loop to minimize cost before selecting one path.
    """
    """
    Use the Worker model to perform a short, targeted simulation (proof sketch) for a given strategic path.
    Constrained by tokens/time via API settings.
    """
    print(f"[NRPA] Running strategic simulation for path: {path_description}")
    worker_prompt = WORKER_SKETCH_PROMPT.format(
        problem_statement=problem_statement[:4000],  # keep prompt bounded
        path_description=path_description
    )
    payload = build_request_payload(
        system_prompt="", 
        question_prompt=worker_prompt,
        temperature=0.2,
        top_p=0.95,
        max_tokens=800  # sketch-sized
    )
    resp = send_api_request(get_api_key("worker"), payload, model_name=WORKER_MODEL_NAME, agent_type="worker", telemetry=telemetry)
    sketch_text = extract_text_from_response(resp)
    print(f"[NRPA] DEBUG: Generated sketch (first 500 chars): {sketch_text[:500]}")
    return sketch_text


def generate_refinements(path_prefix: List[str], problem_statement: str, cache: Dict[str, Any], telemetry=None) -> List[str]:
    """
    Generate refinements for a strategy path prefix using the Strategist model.
    
    Args:
        path_prefix: List of strategy steps so far
        problem_statement: The IMO problem
        cache: Shared cache to avoid duplicate LLM calls
        telemetry: Telemetry system for logging
        
    Returns:
        List of refined strategy steps
    """
    from hashlib import md5
    cache_key = md5(f"refine::{'|'.join(path_prefix)}".encode()).hexdigest()
    if cache_key in cache:
        print(f"[NRPA] Using cached refinements for prefix: {' -> '.join(path_prefix[:2])}")
        return cache[cache_key]
        
    prefix_text = " -> ".join(path_prefix) if path_prefix else "(initial strategies)"
    print(f"[NRPA] Generating refinements for prefix: {prefix_text}")
    
    prompt = STRATEGY_REFINEMENT_PROMPT.format(path_prefix=prefix_text)
    payload = build_request_payload(
        system_prompt="", 
        question_prompt=prompt,
        temperature=0.3,
        top_p=0.9,
        max_tokens=600
    )
    resp = send_api_request(get_api_key("strategist"), payload, model_name=STRATEGIST_MODEL_NAME, agent_type="strategist", telemetry=telemetry)
    text = extract_text_from_response(resp)
    
    # Parse refinements (expecting JSON array)
    refinements = parse_strategies_list(text)
    # Deduplicate and limit
    seen = set()
    unique = []
    for r in refinements:
        if r not in seen and len(r) > 10 and len(r) < 200:
            seen.add(r)
            unique.append(r)
    result = unique[:5]
    cache[cache_key] = result
    print(f"[NRPA] Generated {len(result)} refinements")
    return result


def lightweight_score_sketch(sketch: str, telemetry=None):
    """
    Score a high-level sketch on [0.0, 1.0] using a lightweight verifier prompt.

    Robustness:
    - Attempts strict JSON parsing first, falls back to JSON fragment detection,
      then numeric heuristic to avoid exceptions from imperfect LLM outputs.
    - Always returns (score, reason) with safe defaults to keep the NRPA loop progressing.
    - Guards against empty/malformed API responses to prevent KeyError/TypeError crashes.
    - Adds extra debugging for failed parses to help diagnose model output issues.
    """
    # Use replace instead of format to avoid issues with curly braces in the sketch
    prompt = LIGHTWEIGHT_VERIFIER_PROMPT.replace('{sketch}', sketch)
    payload = build_request_payload(system_prompt="", question_prompt=prompt, temperature=0.0, top_p=1.0, max_tokens=200)
    resp = send_api_request(get_api_key("verifier"), payload, model_name=IMPROVER_MODEL_NAME, agent_type="verifier", telemetry=telemetry)

    # Ensure we have text content; if not, default safely
    text = ""
    try:
        text = extract_text_from_response(resp)
    except Exception as e:
        text = f"(no-text; extract error: {e})"

    # Debug: Print the actual response we're trying to parse
    print(f"[NRPA] DEBUG: Verifier response text: {text[:500]}")
    
    # Parse defensively
    score = 0.0
    reason = ""
    parsed_score = None
    parsed_reason = None
    
    try:
        print(f"[NRPA] DEBUG: Attempting to parse viability score from text: {text[:200]}")
        parsed_score, parsed_reason = parse_viability_score(text or "")
        print(f"[NRPA] DEBUG: parse_viability_score returned: score={parsed_score}, reason={parsed_reason}")
        # Normalize parsed values
        try:
            score = float(parsed_score)
        except Exception as e:
            print(f"[NRPA] DEBUG: Failed to convert parsed_score to float: {parsed_score}, error: {e}")
            score = 0.0
        if score < 0.0 or score > 1.0:
            print(f"[NRPA] DEBUG: Score out of range [0,1]: {score}")
            score = 0.0
        reason = (parsed_reason or "").strip()
    except Exception as e:
        print(f"[NRPA] DEBUG: Exception in parse_viability_score: {e}")
        score = 0.0
        reason = f"parse error: {e}"

    # Fallback reason from raw text if needed
    if not reason:
        snippet = (text or "").strip()
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        reason = snippet or "no reason"

    # Debug output for failed parses
    if score == 0.0 and "parse error" in reason:
        print(f"[NRPA] DEBUG: Failed to parse viability score from verifier response:")
        print(f"[NRPA] DEBUG: Full response text: {text}")
        print(f"[NRPA] DEBUG: Parsed score: {parsed_score}, reason: {parsed_reason}")

    print(f"[NRPA] Lightweight score: {score:.3f}. Reason: {reason}")
    return score, reason


def enumerate_initial_strategies(problem_statement: str, other_prompts):
    """
    Ask the Strategist to list 3–5 distinct high-level paths (one-liners with labels).
    The parser in prompts.py tolerates minor formatting deviations to keep the system robust.
    """
    """
    Ask the Strategist model to enumerate 3–5 high-level strategy candidates.
    """
    enum_prompt = STRATEGIST_ENUM_PROMPT.format(problem_statement=problem_statement)
    payload = build_request_payload(system_prompt=strategist_system_prompt, question_prompt=enum_prompt, other_prompts=other_prompts, temperature=0.2, top_p=0.9, max_tokens=600)
    resp = send_api_request(get_api_key("strategist"), payload, model_name=STRATEGIST_MODEL_NAME, agent_type="strategist")
    text = extract_text_from_response(resp)
    strategies = parse_strategies_list(text)
    print("[NRPA] Initial strategies:")
    for s in strategies:
        print(f" - {s}")
    return strategies


def init_explorations(problem_statement, verbose=True, other_prompts=[], backtracker=None, telemetry=None):
    """
    Entry-stage orchestration using strategy selector pattern.

    Returns conversation payloads, solution text, verification artifacts, scratchpad, and chosen strategy.
    """
    # Initialize the shared memory scratchpad
    scratchpad = initialize_scratchpad(problem_statement)

    # Create API client functions dictionary for strategy selectors
    api_client_funcs = {
        'build_request_payload': build_request_payload,
        'send_api_request': send_api_request,
        'extract_text_from_response': extract_text_from_response,
        'get_api_key': get_api_key
    }

    # Select strategy based on NRPA flag
    if ENABLE_NRPA:
        strategy_selector = NRPAStrategySelector(api_client_funcs, STRATEGIST_MODEL_NAME)
    else:
        strategy_selector = SingleStrategySelector(api_client_funcs, STRATEGIST_MODEL_NAME)

    # Select the best strategy
    strategy = strategy_selector.select_strategy(
        problem_statement,
        other_prompts,
        strategist_system_prompt,
        telemetry=telemetry,
        enumerate_initial_strategies=enumerate_initial_strategies,
        generate_refinements=generate_refinements,
        run_strategic_simulation=run_strategic_simulation,
        lightweight_score_sketch=lightweight_score_sketch,
    )

    print("[CEO] Strategist's Plan / Chosen Path:")
    print(strategy)
    print("-" * 50)

    # --- Proceed with original pipeline using the selected strategy ---
    print("[GENIUS] Worker is implementing the plan...")
    worker_question_prompt = worker_prompt_template.format(
        problem_statement=problem_statement,
        strategy=strategy,
        scratchpad=scratchpad
    )

    worker_payload = build_request_payload(
        system_prompt=step1_prompt, # The original detailed prompt for solution formatting
        question_prompt=worker_question_prompt
    )

    print("[GENIUS] Worker prompt:")
    print(json.dumps(worker_payload, indent=4))

    worker_response = send_api_request(get_api_key("worker"), worker_payload, model_name=WORKER_MODEL_NAME, agent_type="worker", telemetry=telemetry)
    initial_solution = extract_text_from_response(worker_response)

    print("[GENIUS] Worker's Initial Solution:")
    print(json.dumps(initial_solution, indent=4))

    # --- Step 3: Proceed with the original pipeline (Self-Improvement & Verification) ---
    print("[IMPROVER] Self improvement start (on Worker's solution):")

    # Create a new payload for the conversation history
    conversation_history_payload = {
        "messages": worker_payload["messages"] + [
            {"role": "assistant", "content": initial_solution},
            {"role": "user", "content": self_improvement_prompt.format(scratchpad=scratchpad)}
        ],
        "temperature": 0.1,
        "top_p": 1.0
    }

    # Now call the improver model
    improver_response = send_api_request(get_api_key("improver"), conversation_history_payload, model_name=IMPROVER_MODEL_NAME, agent_type="improver", telemetry=telemetry)
    solution = extract_text_from_response(improver_response)
    print("[IMPROVER] Self-Improved solution: ")
    print(json.dumps(solution, indent=4))

    # Create payload for the correction loop
    correction_payload = {
        "messages": conversation_history_payload["messages"] + [
            {"role": "assistant", "content": solution}
        ],
        "temperature": 0.1,
        "top_p": 1.0
    }

    print("[IMPROVER] Check if solution is complete:")
    is_complete = check_if_solution_claimed_complete(solution) # Check the improved solution
    if not is_complete:
        print("[IMPROVER] Solution is not complete. Failed.")
        if telemetry:
            telemetry.record_verification_result(False)
        return None, None, None, None

    print("[VERIFIER] Verifying the self-improved solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)

    print("[VERIFIER] Initial verification: ")
    print(json.dumps(verify, indent=4))
    print(f"[VERIFIER] verify results: {good_verify}")

    if telemetry:
        telemetry.record_verification_result("yes" in good_verify.lower())

    return correction_payload, solution, verify, good_verify, scratchpad, strategy

def agent(problem_statement, other_prompts=[]):
    """
    Full outer loop driver that:
      - Starts telemetry
      - Calls init_explorations to get an initial solution attempt
      - Runs verification-improvement iterations with escalation after repeated failures
      - Emits final solution when stability threshold is met
    """
    # Initialize telemetry system
    # Use default telemetry log directory; logging_utils already manages file outputs
    telemetry = TelemetrySystem()
    telemetry.start_session()
    
    # Initialize the shared memory scratchpad
    scratchpad = initialize_scratchpad(problem_statement)
    
    # Initialize the backtracking manager
    backtracker = BacktrackingManager()
    
    # Get initial explorations
    init_result = init_explorations(problem_statement, True, other_prompts, backtracker, telemetry)
    
    # Handle case where init_explorations returns None (failed to find complete solution)
    if init_result is None or len(init_result) < 6:
        print("[AGENT] Failed in finding a complete solution.")
        telemetry.end_session()
        return None
    
    # Unpack the results
    conversation_history_payload, solution, verify, good_verify, scratchpad, strategy = init_result
    
    if solution is None:
        print("[AGENT] Failed in finding a complete solution.")
        return None
        
    error_count = 0
    correct_count = 0
    if "yes" in good_verify.lower():
        correct_count = 1
        
    for i in range(30):
        print(f"[AGENT] Number of iterations: {i}, number of corrects: {correct_count}, number of errors: {error_count}")
        
        if "yes" not in good_verify.lower():
            # Clear counters
            correct_count = 0
            error_count += 1
            
            # Record failure in backtracker
            should_escalate = backtracker.record_failure(
                "verification_failure", 
                verify[:200] if verify else "No verification details", 
                scratchpad
            )
            
            # Check if we should escalate to CEO for strategy reassessment
            if should_escalate:
                print(f"[AGENT] {backtracker.failure_count} failures reached threshold. Escalating to CEO for strategy reassessment.")
                
                # Generate CEO reassessment prompt
                ceo_prompt = backtracker.generate_ceo_reassessment_prompt(strategy, problem_statement)
                
                # Request new strategy from CEO
                print("[AGENT] Requesting new strategy from CEO...")
                strategist_payload = build_request_payload(
                    system_prompt=strategist_system_prompt,
                    question_prompt=ceo_prompt
                )
                
                strategist_response = send_api_request(
                    get_api_key("strategist"), 
                    strategist_payload, 
                    model_name=STRATEGIST_MODEL_NAME, 
                    agent_type="strategist",
                    telemetry=telemetry
                )
                if telemetry:
                    telemetry.record_strategy_change()
                
                new_strategy = extract_text_from_response(strategist_response)
                print("[AGENT] New strategy from CEO:")
                print(new_strategy)
                
                # Update strategy for next iteration
                strategy = new_strategy
                
                # Reset backtracker for new strategy
                backtracker = BacktrackingManager()
                
                # Reinitialize with new strategy
                worker_question_prompt = worker_prompt_template.format(
                    problem_statement=problem_statement,
                    strategy=strategy,
                    scratchpad=scratchpad
                )
                
                worker_payload = build_request_payload(
                    system_prompt=step1_prompt,
                    question_prompt=worker_question_prompt
                )
                
                worker_response = send_api_request(
                    get_api_key("worker"), 
                    worker_payload, 
                    model_name=WORKER_MODEL_NAME, 
                    agent_type="worker",
                    telemetry=telemetry
                )
                
                solution = extract_text_from_response(worker_response)
                
                # Reset conversation history with new strategy
                conversation_history_payload = {
                    "messages": worker_payload["messages"] + [
                        {"role": "assistant", "content": solution},
                        {"role": "user", "content": self_improvement_prompt.format(scratchpad=scratchpad)}
                    ],
                    "temperature": 0.1,
                    "top_p": 1.0
                }
                
                # Get improved solution
                improver_response = send_api_request(
                    get_api_key("improver"), 
                    conversation_history_payload, 
                    model_name=IMPROVER_MODEL_NAME, 
                    agent_type="improver",
                    telemetry=telemetry
                )
                
                solution = extract_text_from_response(improver_response)
                
                # Reset counters after strategy change
                error_count = 0
                correct_count = 0
                
                # Check if new solution is complete
                is_complete = check_if_solution_claimed_complete(solution)
                if not is_complete:
                    print("[AGENT] New solution from CEO strategy is not complete. Continuing iterations.")
                    continue
            else:
                # Continue with normal correction process
                print("[IMPROVER] Verification does not pass, correcting ...")
                
                # Add correction prompt to the conversation
                conversation_history_payload["messages"].append(
                    {"role": "user", "content": correction_prompt.format(scratchpad=scratchpad) + "\n\n" + verify}
                )
                
                print("[IMPROVER] New prompt:")
                print(json.dumps(conversation_history_payload, indent=4))
                response2 = send_api_request(
                    get_api_key("improver"), 
                    conversation_history_payload, 
                    model_name=IMPROVER_MODEL_NAME, 
                    agent_type="improver",
                    telemetry=telemetry
                )
                
                solution = extract_text_from_response(response2)
                
                # Add the model's response to the conversation
                conversation_history_payload["messages"].append(
                    {"role": "assistant", "content": solution}
                )
                
                print("[IMPROVER] Corrected solution:")
                print(json.dumps(solution, indent=4))
                
                print("[IMPROVER] Check if solution is complete:")
                is_complete = check_if_solution_claimed_complete(solution)
                if not is_complete:
                    print("[IMPROVER] Solution is not complete. Continuing iterations.")
                    continue
                
        print("[VERIFIER] Verify the solution.")
        verify, good_verify = verify_solution(problem_statement, solution)
        
        if telemetry:
            telemetry.record_verification_result("yes" in good_verify.lower())
        
        if "yes" in good_verify.lower():
            print("[VERIFIER] Solution is good, verifying again ...")
            correct_count += 1
            error_count = 0
            
        if correct_count >= 5:
            print("[AGENT] Correct solution found.")
            if telemetry:
                telemetry.record_solution_found()
                telemetry.end_session()
            print(json.dumps(solution, indent=4))
            return solution
            
        elif error_count >= 10:
            print("[AGENT] Failed in finding a correct solution after 10 errors.")
            if telemetry:
                telemetry.end_session()
            return None
            
    print("[AGENT] Failed in finding a correct solution within 30 iterations.")
    if telemetry:
        telemetry.end_session()
    return None

if __name__ == "__main__":
    """
    CLI runner.

    Usage:
      python agent.py problems/imo01.txt
      python agent.py problems/imo01.txt --log logs/run.log --max_runs 5 --verbose
      python agent.py problems/imo05.txt --other_prompts "use parity,try bounding"

    Notes:
      - Logs are written to logs/ by default with a timestamped filename unless --log is provided.
      - Problem file resolution:
          * If a relative path is given, the tool first looks under the repository 'problems/' directory.
          * Falls back to the provided path if not found there.
      - Environment:
          * .env supplies API keys (CEO_API_KEY, GENIUS_API_KEY, IMPROVER_API_KEY) and model/provider settings.
          * NRPA_ENABLED toggles the NRPA-enhanced strategy selection.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='IMO Problem Solver Agent')
    parser.add_argument('problem_file', nargs='?', default='problem_statement.txt',
                        help='Path to the problem statement file (default: problem_statement.txt)')
    parser.add_argument('--log', '-l', type=str, help='Path to log file (optional)')
    parser.add_argument('--other_prompts', '-o', type=str, help='Other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=10, help='Maximum number of runs (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose mode for debugging')
    
    args = parser.parse_args()
    
    # Set verbose mode
    set_verbose_mode(args.verbose)
    
    max_runs = args.max_runs
    
    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(',')
    
    print("[MAIN] Other prompts:")
    print(other_prompts)
    
    # Set up logging
    if args.log:
        log_file_path = args.log
    else:
        log_file_path = initialize_logging()
    
    if not set_log_file(log_file_path):
        sys.exit(1)
    print(f"[MAIN] Logging to file: {log_file_path}")
    
    # Handle file path correctly
    if not os.path.isabs(args.problem_file):
        # If relative path, look in the problems directory first
        problems_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "problems")
        problem_path = os.path.join(problems_dir, args.problem_file)
        if os.path.exists(problem_path):
            problem_statement = read_file_content(problem_path)
        else:
            # Fall back to the provided path
            problem_statement = read_file_content(args.problem_file)
    else:
        # Absolute path provided
        problem_statement = read_file_content(args.problem_file)
    
    for i in range(max_runs):
        print(f"\n\n[MAIN] >>>>>>>>>>>>>>>>>>>>>>>>>> Run {i} of {max_runs} ...")
        try:
            sol = agent(problem_statement, other_prompts)
            if(sol is not None):
                print(f"[MAIN] >>>>>>> Found a correct solution in run {i}.")
                print(json.dumps(sol, indent=4))
                break
        except Exception as e:
            print(f"[MAIN] >>>>>>> Error in run {i}: {e}")
            continue
    
    # Close log file if it was opened
    close_log_file()
