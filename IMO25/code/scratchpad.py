from __future__ import annotations

"""Utility functions for managing the shared scratchpad state."""

from typing import Optional


def initialize_scratchpad(problem_statement: str) -> str:
    """Create an initial scratchpad for a problem statement."""
    return f"""--- WORKING THEORY SCRATCHPAD ---
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


def update_scratchpad(
    scratchpad: str,
    new_fact: Optional[str] = None,
    disproven_hypothesis: Optional[str] = None,
    obstacle: Optional[str] = None,
) -> str:
    """Update sections of the scratchpad with new information."""
    lines = scratchpad.split("\n")
    proven_section = disproven_section = obstacle_section = -1
    for i, line in enumerate(lines):
        if line.startswith("**Proven Facts:**"):
            proven_section = i
        elif line.startswith("**Disproven Hypotheses:**"):
            disproven_section = i
        elif line.startswith("**Current Central Obstacle:**"):
            obstacle_section = i
    if new_fact and proven_section != -1:
        lines.insert(proven_section + 1, f"- {new_fact}")
    if disproven_hypothesis and disproven_section != -1:
        lines.insert(disproven_section + 1, f"- {disproven_hypothesis}")
    if obstacle and obstacle_section != -1:
        lines[obstacle_section] = f"**Current Central Obstacle:** {obstacle}"
    return "\n".join(lines)
