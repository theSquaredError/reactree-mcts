"""
Outer MCTS tree serialization, visualization, and artifact export.

Public entry point:
    export_outer_tree_artifacts(cfg, root_node, task_name) -> Dict[str, str]
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional

try:
    from graphviz import Digraph  # type: ignore[import-not-found]
except ImportError:
    Digraph = None

from .types import DecompositionAction

logger = logging.getLogger(__name__)

NodeType = Any


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "hmt_task").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "hmt_task"


def _resolve_hmt_artifact_dir(cfg) -> str:
    output_dir = getattr(cfg, "out_dir", None)
    if not output_dir or (isinstance(output_dir, str) and "${" in output_dir):
        output_dir = getattr(getattr(cfg, "dataset", object()), "collect_dir", None)
    if not output_dir:
        output_dir = os.path.join("output", "hmt_collect")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Tree → dict serialization
# ---------------------------------------------------------------------------

def _outer_node_to_dict(node: NodeType) -> Dict[str, Any]:
    state = node.get_state()
    action: Optional[DecompositionAction] = getattr(node, "decomposition_action", None)
    return {
        "id": str(id(node)),
        "node_role": getattr(node, "node_role", "goal"),
        "quality_value": node.quality_value,
        "visit_count": node.visit_count,
        "num_children": len(node.get_children()),
        "state": {
            "goal": state.goal,
            "depth": state.depth,
            "action_history": list(state.action_history),
            "executed_steps": list(state.executed_steps),
            "succeed_subgoals": state.succeed_subgoals,
            "failed_subgoals": state.failed_subgoals,
            "is_terminal": state.is_terminal,
            "generated_candidates": [c.signature() for c in state.get_available_actions()],
            "candidate_generation_done": state.candidate_generation_done,
        },
        "trajectory": list(state.trajectory),
        "mcts_attempts": list(state.mcts_attempts),
        "final_success_trajectory": list(getattr(state, "final_success_trajectory", [])),
        "subgoal_step": {
            "index": getattr(node, "subgoal_index", None),
            "text": getattr(node, "subgoal_text", None),
        },
        "decomposition_action": {
            "control_flow": action.control_flow,
            "subgoals": list(action.subgoals),
            "prior_prob": action.prior_prob,
            "signature": action.signature(),
        } if action is not None else None,
        "children": [_outer_node_to_dict(child) for child in node.get_children()],
    }


def _save_outer_tree(root_node: NodeType, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as fh:
        json.dump(_outer_node_to_dict(root_node), fh, indent=2)


# ---------------------------------------------------------------------------
# Graphviz rendering helpers
# ---------------------------------------------------------------------------

def _add_outer_tree_node(
    graph: Any, node_dict: Dict[str, Any], parent_id: Optional[str] = None
) -> None:
    node_id = node_dict["id"]
    state = node_dict.get("state", {})
    action = node_dict.get("decomposition_action")
    node_role = node_dict.get("node_role", "goal")
    subgoal_step = node_dict.get("subgoal_step", {})

    if action is None and node_role == "subgoal_step":
        action_line = f"subgoal[{subgoal_step.get('index', 0)}]: {subgoal_step.get('text', '')}"
    elif action is None:
        action_line = f"node: {node_role.upper()}"
    else:
        action_line = f"action: {action['control_flow']} | prior={action['prior_prob']:.2f}"

    subgoals = action["subgoals"] if action is not None else []
    label = (
        f"goal: {state.get('goal', '')}\n"
        f"depth: {state.get('depth', 0)} | children: {node_dict.get('num_children', 0)}\n"
        f"Q/N: {node_dict.get('quality_value', 0.0):.2f}/{node_dict.get('visit_count', 0)}\n"
        f"{action_line}\n"
        f"subgoals: {subgoals}\n"
        f"terminal: {state.get('is_terminal', False)}"
    )
    graph.node(node_id, label)
    if parent_id is not None:
        graph.edge(parent_id, node_id)
    for child in node_dict.get("children", []):
        _add_outer_tree_node(graph, child, node_id)


def _add_outer_tree_node_simple(
    graph: Any, node_dict: Dict[str, Any], parent_id: Optional[str] = None
) -> None:
    node_id = node_dict["id"]
    action = node_dict.get("decomposition_action")
    node_role = node_dict.get("node_role", "goal")
    subgoal_step = node_dict.get("subgoal_step", {})

    if action is None and node_role == "subgoal_step":
        label = f"subgoal[{subgoal_step.get('index', 0)}]"
    elif action is None:
        label = node_role.upper()
    else:
        label = f"{action['control_flow']}\nchildren={node_dict.get('num_children', 0)}"

    graph.node(node_id, label)
    if parent_id is not None:
        graph.edge(parent_id, node_id)
    for child in node_dict.get("children", []):
        _add_outer_tree_node_simple(graph, child, node_id)


def _render_outer_tree(tree_json_path: str, output_prefix: str) -> Optional[Dict[str, str]]:
    if Digraph is None:
        logger.warning("graphviz not installed; skipping HMT tree rendering")
        return None

    with open(tree_json_path, "r", encoding="utf-8") as fh:
        tree_json = json.load(fh)

    detailed = Digraph(format="pdf")
    detailed.attr(rankdir="TB", nodesep="0.5", ranksep="1.5")
    _add_outer_tree_node(detailed, tree_json)
    detailed_pdf = detailed.render(output_prefix, format="pdf", cleanup=True)
    detailed_png = detailed.render(output_prefix, format="png", cleanup=True)

    simple_prefix = output_prefix + "_simple"
    simple = Digraph(format="pdf")
    simple.attr(rankdir="TB", nodesep="0.5", ranksep="1.5")
    _add_outer_tree_node_simple(simple, tree_json)
    simple_pdf = simple.render(simple_prefix, format="pdf", cleanup=True)
    simple_png = simple.render(simple_prefix, format="png", cleanup=True)

    return {
        "pdf": detailed_pdf,
        "png": detailed_png,
        "simple_pdf": simple_pdf,
        "simple_png": simple_png,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_outer_tree_artifacts(cfg, root_node: NodeType, task_name: str) -> Dict[str, str]:
    artifact_dir = _resolve_hmt_artifact_dir(cfg)
    base_name = _sanitize_filename(task_name)
    output_prefix = os.path.join(artifact_dir, f"{base_name}_outer_mcts_tree")
    json_path = output_prefix + ".json"
    _save_outer_tree(root_node, json_path)
    artifacts: Dict[str, str] = {"json": json_path}

    render_tree = bool(
        getattr(getattr(cfg, "hmt", object()), "render_tree_artifacts", False)
    )
    if render_tree:
        rendered = _render_outer_tree(json_path, output_prefix)
        if rendered is not None:
            artifacts.update(rendered)
    else:
        logger.info("Skipping HMT tree rendering (hmt.render_tree_artifacts=false); JSON only")

    logger.info("Exported HMT outer tree artifacts: %s", artifacts)
    return artifacts


def log_outer_tree_structure(root_node: NodeType, depth: int = 0) -> None:
    """Recursively log the outer tree for debugging."""
    state = root_node.get_state()
    logger.info(
        "%sNode: goal=%s, depth=%s, Q/N=%s/%s",
        "  " * depth,
        state.goal,
        state.depth,
        root_node.quality_value,
        root_node.visit_count,
    )
    for child in root_node.get_children():
        log_outer_tree_structure(child, depth + 1)
