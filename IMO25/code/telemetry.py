from __future__ import annotations

"""Telemetry and backtracking utilities for the agent."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

from .logging_utils import get_next_log_number


class TelemetrySystem:
    """Tracks and logs agent performance metrics during execution."""

    def __init__(self, log_directory: str = "../logs"):
        self.log_directory = log_directory
        self.metrics: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "total_api_calls": 0,
            "api_call_durations": [],
            "agent_iterations": 0,
            "verification_passes": 0,
            "verification_failures": 0,
            "strategy_changes": 0,
            "solution_found": False,
        }
        self.events: List[Dict[str, Any]] = []

    def start_session(self) -> None:
        self.metrics["start_time"] = datetime.now().isoformat()
        self.log_event("SESSION_START", "Telemetry session started")

    def end_session(self) -> None:
        self.metrics["end_time"] = datetime.now().isoformat()
        self.log_event("SESSION_END", "Telemetry session ended")
        self.save_metrics()

    def log_event(self, event_type: str, description: str) -> None:
        self.events.append(
            {"type": event_type, "description": description, "timestamp": datetime.now().isoformat()}
        )

    def record_api_call(self, duration: float) -> None:
        self.metrics["total_api_calls"] += 1
        self.metrics["api_call_durations"].append(duration)

    def record_iteration(self) -> None:
        self.metrics["agent_iterations"] += 1

    def record_verification_result(self, passed: bool) -> None:
        if passed:
            self.metrics["verification_passes"] += 1
        else:
            self.metrics["verification_failures"] += 1

    def record_strategy_change(self) -> None:
        self.metrics["strategy_changes"] += 1
        self.log_event("STRATEGY_CHANGE", "Agent strategy was reassessed by CEO")

    def record_solution_found(self) -> None:
        self.metrics["solution_found"] = True
        self.log_event("SOLUTION_FOUND", "Agent found a correct solution")

    def save_metrics(self) -> None:
        log_number = get_next_log_number() - 1
        metrics_file_path = os.path.join(self.log_directory, f"IMO{log_number}_telemetry.json")

        total_duration = 0
        if self.metrics["start_time"] and self.metrics["end_time"]:
            start = datetime.fromisoformat(self.metrics["start_time"])
            end = datetime.fromisoformat(self.metrics["end_time"])
            total_duration = (end - start).total_seconds()

        avg_api_duration = 0
        if self.metrics["api_call_durations"]:
            avg_api_duration = sum(self.metrics["api_call_durations"]) / len(self.metrics["api_call_durations"])

        full_metrics = {
            **self.metrics,
            "total_duration_seconds": total_duration,
            "average_api_call_duration": avg_api_duration,
            "events": self.events,
        }

        try:
            with open(metrics_file_path, "w") as f:
                json.dump(full_metrics, f, indent=2)
        except Exception as e:
            print(f"[TELEMETRY] Error saving metrics: {e}")


class BacktrackingManager:
    """Manages strategic backtracking when repeated verification failures occur."""

    def __init__(self, max_failures: int = 3):
        self.failure_count = 0
        self.max_failures = max_failures
        self.failure_history: List[Dict[str, Any]] = []

    def record_failure(self, error_type: str, context: str, scratchpad_state: str) -> bool:
        self.failure_count += 1
        self.failure_history.append(
            {
                "type": error_type,
                "context": context,
                "scratchpad_state": scratchpad_state,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return self.failure_count >= self.max_failures

    def generate_ceo_reassessment_prompt(self, original_strategy: str, problem_statement: str) -> str:
        history_summary = "\n".join(
            f"- [{f['timestamp']}] {f['type']}: {f['context']}" for f in self.failure_history
        )
        return (
            "We have encountered repeated verification failures while following the current strategy.\n"
            f"Original strategy: {original_strategy}\n"
            f"Problem: {problem_statement}\n\n"
            "Failure history:\n"
            f"{history_summary}\n\n"
            "Please reassess the strategy and suggest a revised approach."
        )

    def reset(self) -> None:
        self.failure_count = 0
        self.failure_history.clear()
