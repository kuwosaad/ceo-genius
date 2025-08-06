from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any

try:
    from .logging_utils import get_next_log_number
except ImportError:
    from logging_utils import get_next_log_number


class TelemetrySystem:
    """Track and persist agent metrics and events."""

    def __init__(self, log_directory: str = "../logs") -> None:
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
            "solution_found": False,
        }
        self.events: list[dict[str, Any]] = []

    def start_session(self) -> None:
        self.metrics["start_time"] = datetime.now().isoformat()
        self.log_event("SESSION_START", "Telemetry session started")

    def end_session(self) -> None:
        self.metrics["end_time"] = datetime.now().isoformat()
        self.log_event("SESSION_END", "Telemetry session ended")
        self.save_metrics()

    def log_event(self, event_type: str, description: str) -> None:
        self.events.append(
            {
                "type": event_type,
                "description": description,
                "timestamp": datetime.now().isoformat(),
            }
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
        metrics_file_path = os.path.join(
            self.log_directory, f"IMO{log_number}_telemetry.json"
        )
        total_duration = 0.0
        if self.metrics["start_time"] and self.metrics["end_time"]:
            start = datetime.fromisoformat(self.metrics["start_time"])
            end = datetime.fromisoformat(self.metrics["end_time"])
            total_duration = (end - start).total_seconds()
        avg_api_duration = 0.0
        if self.metrics["api_call_durations"]:
            avg_api_duration = sum(self.metrics["api_call_durations"]) / len(
                self.metrics["api_call_durations"]
            )
        full_metrics = {
            **self.metrics,
            "total_duration_seconds": total_duration,
            "average_api_call_duration": avg_api_duration,
            "events": self.events,
        }
        try:
            with open(metrics_file_path, "w") as f:
                json.dump(full_metrics, f, indent=2)
            print(f"[TELEMETRY] Metrics saved to {metrics_file_path}")
        except Exception as e:
            print(f"[TELEMETRY] Error saving metrics: {e}")
