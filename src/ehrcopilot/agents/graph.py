"""LangGraph supervisor graph for the Clinical-Analytics Copilot.

Why agents here: each node has distinct responsibilities (plan, link, generate, guard,
execute, assess, summarize) that benefit from typed state handoffs and conditional routing.
The graph makes the data flow auditable and each node independently testable. This is
different from a simple request-response chain where agents would be over-engineering.

Graph flow:
  planner → [schema_linker if answerable | summarizer if not]
  schema_linker → sql_writer
  sql_writer → guard_validator
  guard_validator → [executor if passed | repair if failed + retries left | reliability_gate if exhausted]
  repair → sql_writer  (retry loop)
  executor → reliability_gate
  reliability_gate → clinical_safe_summarizer
  clinical_safe_summarizer → END
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END  # type: ignore[import]

from ehrcopilot import config
from ehrcopilot.agents.state import CopilotState
from ehrcopilot.agents.nodes.planner import planner_node
from ehrcopilot.agents.nodes.schema_linker import schema_linker_node
from ehrcopilot.agents.nodes.sql_writer import sql_writer_node
from ehrcopilot.agents.nodes.guard_validator import guard_validator_node
from ehrcopilot.agents.nodes.repair import repair_node
from ehrcopilot.agents.nodes.executor import executor_node
from ehrcopilot.agents.nodes.reliability_gate import reliability_gate_node
from ehrcopilot.agents.nodes.clinical_safe_summarizer import clinical_safe_summarizer_node


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_planner(state: CopilotState) -> Literal["schema_linker", "clinical_safe_summarizer"]:
    if not state.get("answerable", True):
        return "clinical_safe_summarizer"
    return "schema_linker"


def _route_guard(
    state: CopilotState,
) -> Literal["executor", "repair", "reliability_gate"]:
    # Model already abstained (sql=None before guard ran)
    if state.get("abstain"):
        return "reliability_gate"

    guard_result = state.get("guard_result")
    if guard_result and not guard_result.passed:
        repair_count = state.get("repair_count", 0)
        if repair_count < config.MAX_REPAIR_ATTEMPTS:
            return "repair"
        # Exhausted retries
        return "reliability_gate"

    return "executor"


def _route_executor(
    state: CopilotState,
) -> Literal["reliability_gate", "repair"]:
    if state.get("exec_error") and not state.get("abstain"):
        repair_count = state.get("repair_count", 0)
        if repair_count < config.MAX_REPAIR_ATTEMPTS:
            return "repair"
    return "reliability_gate"


def _route_repair(
    state: CopilotState,
) -> Literal["sql_writer", "reliability_gate"]:
    if state.get("repair_count", 0) >= config.MAX_REPAIR_ATTEMPTS:
        return "reliability_gate"
    return "sql_writer"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    g: StateGraph = StateGraph(CopilotState)  # type: ignore[type-var]

    g.add_node("planner", planner_node)
    g.add_node("schema_linker", schema_linker_node)
    g.add_node("sql_writer", sql_writer_node)
    g.add_node("guard_validator", guard_validator_node)
    g.add_node("repair", repair_node)
    g.add_node("executor", executor_node)
    g.add_node("reliability_gate", reliability_gate_node)
    g.add_node("clinical_safe_summarizer", clinical_safe_summarizer_node)

    g.set_entry_point("planner")

    g.add_conditional_edges(
        "planner",
        _route_planner,
        {
            "schema_linker": "schema_linker",
            "clinical_safe_summarizer": "clinical_safe_summarizer",
        },
    )

    g.add_edge("schema_linker", "sql_writer")
    g.add_edge("sql_writer", "guard_validator")

    g.add_conditional_edges(
        "guard_validator",
        _route_guard,
        {
            "executor": "executor",
            "repair": "repair",
            "reliability_gate": "reliability_gate",
        },
    )

    g.add_conditional_edges(
        "executor",
        _route_executor,
        {
            "reliability_gate": "reliability_gate",
            "repair": "repair",
        },
    )

    g.add_conditional_edges(
        "repair",
        _route_repair,
        {
            "sql_writer": "sql_writer",
            "reliability_gate": "reliability_gate",
        },
    )

    g.add_edge("reliability_gate", "clinical_safe_summarizer")
    g.add_edge("clinical_safe_summarizer", END)

    return g


# Compiled graph — import this for inference
copilot_graph = build_graph().compile()


def run(question: str, role: str = "clinician") -> CopilotState:
    """Run the copilot graph for a single question. Returns the final state."""
    initial: CopilotState = {
        "question": question,
        "role": role,
        "repair_count": 0,
        "abstain": False,
        "abstain_reason": None,
        "cache_hit": False,
        "confidence": 0.0,
    }
    return copilot_graph.invoke(initial)
