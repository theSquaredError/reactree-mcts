"""
ALFRED action constants, regex patterns, and the Node class loader.
All other hmt modules import from here rather than re-defining these.
"""

import os
import re
import importlib.util
from typing import Any, Pattern, Tuple


# ---------------------------------------------------------------------------
# ALFRED primitive action vocabulary
# ---------------------------------------------------------------------------

ALFRED_PRIMITIVE_ACTION_PREFIXES: Tuple[str, ...] = (
    "go to ",
    "pick up ",
    "put down ",
    "open ",
    "close ",
    "turn on ",
    "turn off ",
    "slice ",
)

ALFRED_TERMINAL_ACTIONS: Tuple[str, ...] = ("done", "failure")

DECOMPOSITION_CONNECTOR_PATTERN = re.compile(
    r"\b(?:and then|then|and|after that|afterwards)\b|,",
    flags=re.IGNORECASE,
)

ALFRED_PRIMITIVE_ACTION_PATTERNS: Tuple[Pattern[str], ...] = tuple(
    re.compile(rf"^{re.escape(prefix.strip())}\s+\S.+$")
    for prefix in ALFRED_PRIMITIVE_ACTION_PREFIXES
)


def _looks_like_compound_goal(goal: str) -> bool:
    return bool(DECOMPOSITION_CONNECTOR_PATTERN.search(goal or ""))


# ---------------------------------------------------------------------------
# Node class loader (tries several import paths before falling back to direct
# file load, so the hmt package works regardless of how it is invoked)
# ---------------------------------------------------------------------------

def _load_node_class() -> Any:
    try:
        from src.mcts.node import Node as ImportedNode  # type: ignore
        return ImportedNode
    except Exception:
        pass

    try:
        from mcts.node import Node as ImportedNode  # type: ignore
        return ImportedNode
    except Exception:
        pass

    node_path = os.path.join(os.path.dirname(__file__), "..", "mcts", "node.py")
    spec = importlib.util.spec_from_file_location("hmt_mcts_node", node_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate mcts.node module for HMT")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Node


Node = _load_node_class()
NodeType = Any
