"""Reliability gate node: compute calibrated confidence, decide ANSWER vs ABSTAIN.

Confidence is derived from three signals:
  1. Whether the model emitted [ABSTAIN] (confidence = 0.0 immediately)
  2. Whether SQL executed without error (boosts confidence)
  3. Whether a repair was needed (penalizes confidence proportionally)
"""

from __future__ import annotations

from ehrcopilot import config
from ehrcopilot.agents.state import CopilotState


def reliability_gate_node(state: CopilotState) -> CopilotState:
    # Already abstained upstream (model, guard, or small-cell)
    if state.get("abstain"):
        return {
            **state,
            "confidence": 0.0,
            "abstain": True,
            "abstain_reason": state.get("abstain_reason") or "upstream_abstain",
        }

    exec_result = state.get("exec_result")
    exec_error = state.get("exec_error")
    repair_count = state.get("repair_count", 0)
    guard_result = state.get("guard_result")

    # Base confidence: 1.0 if execution succeeded, 0.0 if not
    if exec_error or exec_result is None:
        return {
            **state,
            "confidence": 0.0,
            "abstain": True,
            "abstain_reason": exec_error or "execution_failed",
        }

    confidence = 1.0

    # Penalty per repair attempt (each repair indicates uncertainty)
    confidence -= 0.2 * repair_count

    # Guard result passed but noted a borderline issue
    if guard_result and not guard_result.passed:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    threshold = config.CONFIDENCE_THRESHOLD
    if confidence < threshold:
        return {
            **state,
            "confidence": confidence,
            "abstain": True,
            "abstain_reason": f"confidence_below_threshold({confidence:.2f}<{threshold})",
        }

    return {**state, "confidence": confidence, "abstain": False}
