"""
telemetry_ext.py

Non-invasive telemetry helpers for recording NRPA lifecycle events.

Purpose
- Provide thin wrappers that annotate the TelemetrySystem (defined in agent.py) with NRPA-specific events
  without importing or tightly coupling to the TelemetrySystem class.
- Keep agent.py clean by centralizing event naming and payload formatting for:
    * NRPA_START: initial candidate summary (e.g., number of strategies)
    * NRPA_ITERATION: per-iteration selection, score, and policy summary
    * NRPA_END: final chosen strategy and policy state

Why separate?
- Decouples logging/telemetry responsibilities from core search and orchestration logic.
- Allows alternate telemetry systems to be injected without code changes here; wrappers no-op if telemetry is None.

Usage
- Called from agent.init_explorations() around the NRPA loop.

Resilience
- Each function guards against missing telemetry or logging failures to avoid breaking the solve loop.
"""
from __future__ import annotations
from typing import Any, Dict


# Lightweight wrappers to record NRPA-related telemetry without modifying the core TelemetrySystem class.
# The TelemetrySystem in agent.py exposes:
#   - log_event(event_type: str, description: str)
#   - metrics dict and save_metrics()
#
# These helpers simply format and forward NRPA events via log_event.


def nrpa_start(telemetry: Any, summary: Dict[str, Any]) -> None:
    """
    Log NRPA session start with an initial summary payload.
    Example summary: {"num_candidates": 5}
    """
    if telemetry is None:
        return
    try:
        telemetry.log_event("NRPA_START", f"Start NRPA: {summary}")
    except Exception:
        # Be resilient to formatting/logging failures
        pass


def nrpa_iteration(telemetry: Any, iteration: int, data: Dict[str, Any]) -> None:
    """
    Log one iteration of NRPA with selection, score, and policy summaries.
    Example data: {"selected_path": "...", "score": 0.63, "reason": "...", "policy": {...}}
    """
    if telemetry is None:
        return
    try:
        payload = {"iteration": iteration, **data}
        telemetry.log_event("NRPA_ITERATION", f"Iter {iteration}: {payload}")
    except Exception:
        pass


def nrpa_end(telemetry: Any, result: Dict[str, Any]) -> None:
    """
    Log NRPA session end with final best strategy and policy state.
    Example result: {"chosen": "Inversion: ...", "policy": {...}}
    """
    if telemetry is None:
        return
    try:
        telemetry.log_event("NRPA_END", f"End NRPA: {result}")
    except Exception:
        pass
