"""
Hierarchical MCTS (HMT): Two-Level Tree Search
- Outer MCTS: explores task decomposition candidates (Expand choices)
- Inner MCTS: explores primitive action sequences for each subgoal

Usage:
    hmt = HierarchicalMCTS(cfg, llm_agent, env, outer_budget=5, inner_budget=10)
    terminate_info = hmt.collect_llm(task_d, args_dict)
"""

import logging
import copy
import math
import os
import json
import random
import importlib.util
import re
import sys
from typing import Any, Dict, List, Pattern, Tuple, Optional
from dataclasses import dataclass, field

try:
    from graphviz import Digraph  # type: ignore[import-not-found]
except ImportError:
    Digraph = None


def _load_node_class():
    # Try regular imports first.
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

    # Fallback: load node.py directly without importing mcts package __init__.
    node_path = os.path.join(os.path.dirname(__file__), "mcts", "node.py")
    spec = importlib.util.spec_from_file_location("hmt_mcts_node", node_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate mcts.node module for HMT")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Node


Node = _load_node_class()
NodeType = Any


from alfred.alfred_reactree import AlfredReactree  
from alfred.utils import dotdict, load_task_json
import alfred.utils as alfred_utils  


try:
    import hydra  # type: ignore[import-not-found]
except ImportError:
    hydra = None

logger = logging.getLogger(__name__)


ALFRED_PRIMITIVE_ACTION_PREFIXES: Tuple[str, ...] = (
    "go to ",
    "pick up ",
    "put down ",
    "open ",
    "close ",
    "turn on ",
    "turn off ",
    "slice ",
    "recall location of ",
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


def _configure_logging(cfg) -> str:
    """Configure root logging once and write HMT logs to a dedicated file."""
    log_dir = getattr(cfg, "out_dir", None)
    if not log_dir or (isinstance(log_dir, str) and "${" in log_dir):
        log_dir = getattr(getattr(cfg, "dataset", object()), "collect_dir", None)
    if not log_dir:
        log_dir = os.path.join("output", "hmt_collect")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "hmt.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler_exists = False
    stream_handler_exists = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == os.path.abspath(log_path):
            handler.setFormatter(formatter)
            file_handler_exists = True
        elif isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)
            stream_handler_exists = True

    if not file_handler_exists:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if not stream_handler_exists:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    logger.info("Logging initialized")
    logger.info("HMT log file: %s", log_path)
    return log_path


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


def _outer_node_to_dict(node: NodeType) -> Dict[str, Any]:
    state = node.get_state()
    action = getattr(node, "decomposition_action", None)
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
            "generated_candidates": [candidate.signature() for candidate in state.get_available_actions()],
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
    with open(filename, "w", encoding="utf-8") as file_obj:
        json.dump(_outer_node_to_dict(root_node), file_obj, indent=2)


def _add_outer_tree_node(graph: Any, node_dict: Dict[str, Any], parent_id: Optional[str] = None) -> None:
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


def _add_outer_tree_node_simple(graph: Any, node_dict: Dict[str, Any], parent_id: Optional[str] = None) -> None:
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
        logger.warning("graphviz is not installed; skipping HMT tree rendering")
        return None

    with open(tree_json_path, "r", encoding="utf-8") as file_obj:
        tree_json = json.load(file_obj)

    detailed_graph = Digraph(format="pdf")
    detailed_graph.attr(rankdir="TB", nodesep="0.5", ranksep="1.5")
    _add_outer_tree_node(detailed_graph, tree_json)
    detailed_pdf = detailed_graph.render(output_prefix, format="pdf", cleanup=True)
    detailed_png = detailed_graph.render(output_prefix, format="png", cleanup=True)

    simple_prefix = output_prefix + "_simple"
    simple_graph = Digraph(format="pdf")
    simple_graph.attr(rankdir="TB", nodesep="0.5", ranksep="1.5")
    _add_outer_tree_node_simple(simple_graph, tree_json)
    simple_pdf = simple_graph.render(simple_prefix, format="pdf", cleanup=True)
    simple_png = simple_graph.render(simple_prefix, format="png", cleanup=True)

    return {
        "pdf": detailed_pdf,
        "png": detailed_png,
        "simple_pdf": simple_pdf,
        "simple_png": simple_png,
    }


def export_outer_tree_artifacts(cfg, root_node: NodeType, task_name: str) -> Dict[str, str]:
    artifact_dir = _resolve_hmt_artifact_dir(cfg)
    base_name = _sanitize_filename(task_name)
    output_prefix = os.path.join(artifact_dir, f"{base_name}_outer_mcts_tree")
    json_path = output_prefix + ".json"
    _save_outer_tree(root_node, json_path)
    artifacts = {"json": json_path}

    render_tree = bool(getattr(getattr(cfg, "hmt", object()), "render_tree_artifacts", False))
    if render_tree:
        rendered_paths = _render_outer_tree(json_path, output_prefix)
        if rendered_paths is not None:
            artifacts.update(rendered_paths)
    else:
        logger.info("Skipping HMT tree rendering (hmt.render_tree_artifacts=false); JSON only")

    logger.info("Exported HMT outer tree artifacts: %s", artifacts)
    return artifacts


# ============================================================================
# OUTER LEVEL: Decomposition State & MCTS
# ============================================================================

class DecompositionState:
    """
    Represents a state at the decomposition level.
    
    Attributes:
        env_snapshot: environment state (world objects, agent position, etc.)
        goal: current task goal (string)
        depth: decomposition depth
        executed_steps: list of committed (executed) steps so far
        succeed_subgoals: number of successfully completed subgoals
        failed_subgoals: number of failed subgoals
    """
    
    def __init__(self, 
                 goal: str,
                 env_snapshot: Dict,
                 depth: int = 0,
                 executed_steps: Optional[List] = None,
                 action_history: Optional[List[str]] = None,
                 succeed_subgoals: int = 0,
                 failed_subgoals: int = 0,
                 max_candidate_count: int = 3,
                 generated_candidates: Optional[List["DecompositionAction"]] = None,
                 candidate_generation_done: bool = False,
                 trajectory: Optional[List[str]] = None,
                 mcts_attempts: Optional[List[Dict]] = None,
                 final_success_trajectory: Optional[List[Dict[str, Any]]] = None):
        self.goal = goal
        self.env_snapshot = copy.deepcopy(env_snapshot)
        self.depth = depth
        self.executed_steps = executed_steps or []
        self.action_history = action_history or []
        self.succeed_subgoals = succeed_subgoals
        self.failed_subgoals = failed_subgoals
        self.max_candidate_count = max(1, int(max_candidate_count))
        self.generated_candidates = list(generated_candidates or [])
        self.candidate_generation_done = candidate_generation_done
        self.is_terminal = False
        self.trajectory = list(trajectory or [])  # actions taken to execute this node
        self.mcts_attempts = list(mcts_attempts or [])  # all MCTS iterations with candidates and simulated actions
        self.final_success_trajectory = list(final_success_trajectory or [])  # [{subgoal, primitive_actions, ...}, ...]
    
    def clone(self):
        """Create a deep copy for branching exploration."""
        return DecompositionState(
            goal=self.goal,
            env_snapshot=copy.deepcopy(self.env_snapshot),
            depth=self.depth,
            executed_steps=copy.deepcopy(self.executed_steps),
            action_history=copy.deepcopy(self.action_history),
            succeed_subgoals=self.succeed_subgoals,
            failed_subgoals=self.failed_subgoals,
            max_candidate_count=self.max_candidate_count,
            generated_candidates=copy.deepcopy(self.generated_candidates),
            candidate_generation_done=self.candidate_generation_done,
            trajectory=copy.deepcopy(self.trajectory),
            mcts_attempts=copy.deepcopy(self.mcts_attempts),
            final_success_trajectory=copy.deepcopy(self.final_success_trajectory),
        )

    def get_available_actions(self) -> List["DecompositionAction"]:
        return self.generated_candidates


class DecompositionAction:
    """
    Represents an Expand action at decomposition level.
    
    Attributes:
        control_flow: 'sequence' | 'fallback' | 'parallel'
        subgoals: ordered list of subgoal strings
        prior_prob: LLM probability/confidence
    """
    
    def __init__(self, 
                 control_flow: str,
                 subgoals: List[str],
                 prior_prob: float = 0.5):
        self.control_flow = control_flow
        self.subgoals = subgoals
        self.prior_prob = prior_prob
    
    def __repr__(self):
        return f"DecompAction({self.control_flow}: {self.subgoals[:2]}...)"

    def signature(self) -> str:
        return f"{self.control_flow}::{'||'.join(self.subgoals)}"


class OuterMCTSPlanner:
    """
    Outer MCTS loop over decomposition candidates.
    
    For each state (partially executed task):
    1. Sample K decomposition candidates from LLM.
    2. For each candidate, run inner MCTS to solve subgoals.
    3. Backup reward to outer tree.
    4. Select best candidate and commit some prefix.
    5. Execute committed prefix in real environment.
    6. Move to next state and repeat.
    """
    
    def __init__(self, 
                 cfg,
                 llm_agent,
                 env,
                 outer_budget: int = 2,
                 inner_budget: int = 5,
                 decomp_candidate_count: int = 3):
        """
        Args:
            cfg: config object
            llm_agent: LLM agent for prompting
            env: ALFRED environment (ThorConnector)
            outer_budget: number of outer MCTS simulations per state
            inner_budget: number of inner MCTS simulations per subgoal (action level)
            decomp_candidate_count: number of candidate decompositions to sample
        """
        self.cfg = cfg
        self.llm_agent = llm_agent
        self.env = env
        self.outer_budget = outer_budget
        self.inner_budget = inner_budget
        self.decomp_candidate_count = decomp_candidate_count
        self.logger = logger
        self.action_mcts = ActionMCTSWrapper(
            cfg=cfg,
            llm_agent=llm_agent,
            env=env,
            budget=inner_budget,
        )
        # Runtime execution state — set by AlfredReactreeWithHMT.collect_llm_with_hmt
        # before calling execute_goal_with_mcts so that _execute_subgoal_with_reactree
        # can drive the real environment.
        self._current_traj_data: Optional[Dict] = None
        self._cur_step_id: int = 1
        self._cur_decision_id: int = 1
        # Scene restore info for simulation — set by collect_llm_with_hmt so that
        # outer_default_policy can reset the environment to a clean initial state
        # before each outer MCTS simulation, making reward signals comparable.
        self._sim_restore_info: Optional[Dict] = None

    @staticmethod
    def _node_role(node: NodeType) -> str:
        return getattr(node, "node_role", "goal")

    def _get_decomposition_children(self, node: NodeType) -> List[NodeType]:
        return [child for child in node.get_children() if self._node_role(child) == "decomposition"]

    def _get_decomposition_paths(self, node: NodeType) -> List[List[str]]:
        all_paths: List[List[str]] = []

        def dfs(current_node: NodeType):
            children = self._get_decomposition_children(current_node)
            if not children:
                all_paths.append(list(current_node.get_state().action_history))
                return
            for child in children:
                dfs(child)

        dfs(node)
        return all_paths

    def _make_goal_node(
        self,
        goal: str,
        env_snapshot: Dict,
        depth: int,
        action_history: List[str],
        executed_steps: Optional[List[str]] = None,
        node_role: str = "goal",
        subgoal_index: Optional[int] = None,
        subgoal_text: Optional[str] = None,
    ) -> NodeType:
        state = DecompositionState(
            goal=goal,
            env_snapshot=copy.deepcopy(env_snapshot),
            depth=depth,
            executed_steps=copy.deepcopy(executed_steps or []),
            action_history=copy.deepcopy(action_history),
            max_candidate_count=self.decomp_candidate_count,
        )
        node = Node()
        node.set_state(state)
        node.node_role = node_role
        if subgoal_index is not None:
            node.subgoal_index = subgoal_index
        if subgoal_text is not None:
            node.subgoal_text = subgoal_text
        return node

    def _record_terminal_result(self, node: NodeType, reward: float, success: bool):
        state = node.get_state()
        state.is_terminal = success
        if node.visit_count == 0:
            node.increment_visit_count()
        node.update_quality_value(reward)

    def _format_action_metadata(self, action: Optional[DecompositionAction]) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        return {
            "control_flow": action.control_flow,
            "subgoals": list(action.subgoals),
            "prior_prob": action.prior_prob,
            "signature": action.signature(),
        }

    @staticmethod
    def _normalize_goal_text(goal: str) -> str:
        return " ".join((goal or "").strip().lower().split())

    def _max_depth(self) -> int:
        return int(getattr(getattr(self.cfg, "llm_agent", object()), "max_depth", 4))

    def _is_primitive_goal(self, goal: str) -> bool:
        normalized_goal = self._normalize_goal_text(goal)
        if not normalized_goal:
            return False
        if normalized_goal in ALFRED_TERMINAL_ACTIONS:
            return True
        if _looks_like_compound_goal(normalized_goal):
            return False
        return any(pattern.match(normalized_goal) for pattern in ALFRED_PRIMITIVE_ACTION_PATTERNS)

    def _has_decomposition_progress(self, goal: str, subgoals: List[str]) -> bool:
        normalized_goal = self._normalize_goal_text(goal)
        normalized_subgoals = [self._normalize_goal_text(subgoal) for subgoal in subgoals if subgoal.strip()]
        if not normalized_subgoals:
            return False
        if len(normalized_subgoals) == 1 and normalized_subgoals[0] == normalized_goal:
            return False
        return True

    def _is_state_terminal(self, state: DecompositionState) -> bool:
        return state.is_terminal or state.depth >= self._max_depth()

    @staticmethod
    def _candidate_subgoal_key(action: DecompositionAction) -> Tuple[str, ...]:
        return tuple(action.subgoals)

    def _llm_generate_expand_candidate(
        self,
        goal: str,
        obs_text: str,
        task_type: str,
        depth: int,
    ) -> Optional[DecompositionAction]:
        """Call the LLM once and return a DecompositionAction if it outputs Expand.

        Mirrors how AlfredAgentNode.collect_llm asks plan_next_step() in a loop:
        the LLM may Think first before deciding to Expand.  We allow up to
        *max_try* iterations to get an Expand response.  If the LLM instead
        chooses Act or Error, we return None so the caller falls back to
        heuristic candidates.
        """
        nl_inst_info = {
            "nl_inst": goal,
            "message": None,
            "task_type": task_type,
            "depth": depth,
        }
        # A minimal skill_set is intentional: we want the LLM to focus on
        # decomposing (Expand) rather than picking a primitive action.
        minimal_skill_set = ["done", "failure"]
        max_try = 6  # LLM may emit Think steps before Expand
        try:
            self.llm_agent.reset(nl_inst_info, obs_text)
            for attempt in range(max_try):
                next_step_info = self.llm_agent.plan_next_step(minimal_skill_set)
                next_step_class = next_step_info["next_step_class"]
                next_step = next_step_info["next_step"]
                self.logger.info(
                    "LLM expand query | attempt=%d/%d | goal='%s' | class=%s",
                    attempt + 1, max_try, goal, next_step_class,
                )
                if next_step_class == "Expand":
                    control_flow = next_step["control_flow"]
                    raw_subgoals = next_step["conditions"].split(", ")
                    subgoals = [s.strip() for s in raw_subgoals if s.strip()]
                    if self._has_decomposition_progress(goal, subgoals):
                        self.logger.info(
                            "LLM decomposition | control_flow=%s | subgoals=%s | prior=0.80",
                            control_flow, subgoals,
                        )
                        return DecompositionAction(control_flow, subgoals, prior_prob=0.80)
                    self.logger.info("LLM Expand had no decomposition progress, retrying")
                elif next_step_class == "Think":
                    # LLM is reasoning — let it continue
                    self.logger.info(
                        "LLM Think | goal='%s' | thought='%s'",
                        goal,
                        next_step,
                    )
                    continue
                else:
                    # LLM chose Act or Error — treat goal as primitive from LLM's view
                    self.logger.info(
                        "LLM chose '%s' instead of Expand for goal='%s'; no decomposition",
                        next_step_class, goal,
                    )
                    break
        except Exception as exc:
            self.logger.warning("_llm_generate_expand_candidate failed: %s", exc)
        return None

    def _execute_subgoal_with_reactree(
        self,
        goal: str,
        env_snapshot: Dict,
        depth: int,
    ) -> Tuple[bool, float, int, List[str]]:
        """Execute *goal* by running the full ReAcTree Act/Think/Expand loop in the
        real environment, exactly as AlfredAgentNode.collect_llm does.

        Returns (success, reward, steps_taken, trajectory).
        Falls back to the lightweight heuristic if traj_data is not set.
        """
        if self._current_traj_data is None:
            self.logger.warning(
                "_execute_subgoal_with_reactree: _current_traj_data not set, using heuristic"
            )
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(goal, env_snapshot.get("observation", ""))
            return success, reward, steps, trajectory

        traj_data = self._current_traj_data
        task_type = env_snapshot.get("task_type", "unknown")
        nl_inst_info = {
            "nl_inst": goal,
            "message": None,
            "task_type": task_type,
            "depth": depth,
        }

        try:
            obs = self.env.init_reset(traj_data)
            obs_text = obs["text"]
            self.llm_agent.reset(nl_inst_info, obs_text)
            skill_set = self.llm_agent.update_skill_set(obs)
        except Exception as exc:
            self.logger.warning("_execute_subgoal_with_reactree init failed: %s", exc)
            return False, -0.5, 0, []

        max_decisions = int(getattr(getattr(self.cfg, "llm_agent", object()), "max_decisions", 50))
        steps_taken = 0
        try_count = 0
        max_try = 5
        current_obs_text = obs_text
        trajectory: List[str] = []

        self.logger.info(
            "ReactTree execution START | goal='%s' | depth=%d | obs='%.80s'",
            goal, depth, obs_text,
        )

        while True:
            if self._cur_decision_id > max_decisions:
                self.logger.info("ReactTree max_decisions reached | goal='%s'", goal)
                return False, -0.5, steps_taken, trajectory

            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class = next_step_info["next_step_class"]
                next_step = next_step_info["next_step"]
                try_count = 0
            except Exception as exc:
                self.logger.warning("plan_next_step error (try %d): %s", try_count + 1, exc)
                try_count += 1
                if try_count >= max_try:
                    return False, -0.5, steps_taken, trajectory
                continue

            self.logger.info(
                "ReactTree step | goal='%s' | %s: %s",
                goal, next_step_class, next_step,
            )

            if next_step_class == "Think":
                self.logger.info(
                    "ReactTree Think | goal='%s' | thought='%s'",
                    goal,
                    next_step,
                )
                self._cur_decision_id += 1

            elif next_step_class == "Act":
                self._cur_decision_id += 1
                act_goal = str(next_step).strip()
                if act_goal == "done":
                    self.logger.info(
                        "ReactTree done | goal='%s' | steps=%d", goal, steps_taken
                    )
                    return True, 1.0, steps_taken, trajectory
                elif act_goal == "failure":
                    self.logger.info(
                        "ReactTree failure | goal='%s' | steps=%d", goal, steps_taken
                    )
                    return False, -0.5, steps_taken, trajectory
                else:
                    self.logger.info(
                        "ReactTree Act delegated to inner MCTS | goal='%s' | act='%s'",
                        goal,
                        act_goal,
                    )
                    act_success, act_reward, act_steps, act_trajectory = self.inner_mcts_solve_subgoal(
                        act_goal,
                        current_obs_text,
                    )
                    if act_trajectory:
                        trajectory.extend(act_trajectory)
                    else:
                        trajectory.append(act_goal)
                    steps_taken += act_steps
                    self._cur_step_id += act_steps

                    try:
                        scene_name = self.env.last_event.metadata.get("sceneName", "FloorPlan1")
                        obs_pack = self.env.llm_skill_interact(None, scene_name)
                        current_obs_text = obs_pack.get("message", current_obs_text)
                        self.llm_agent.add_obs(current_obs_text)
                        skill_set = self.llm_agent.update_skill_set(obs_pack)
                    except Exception as exc:
                        self.logger.warning("Failed to refresh obs/skills after inner MCTS Act: %s", exc)

                    self.logger.info(
                        "ReactTree Act via inner MCTS result | act='%s' | success=%s | reward=%.3f | steps=%d | actions=%s",
                        act_goal,
                        act_success,
                        act_reward,
                        act_steps,
                        act_trajectory,
                    )
                    if not act_success:
                        return False, act_reward, steps_taken, trajectory

            elif next_step_class == "Expand":
                # LLM wants to decompose further — delegate back through HMT so that
                # the outer MCTS handles the sub-decomposition properly.
                self._cur_decision_id += 1
                control_flow = next_step["control_flow"]
                raw_subgoals = next_step["conditions"].split(", ")
                subgoals = [s.strip() for s in raw_subgoals if s.strip()]
                self.logger.info(
                    "ReactTree nested Expand | goal='%s' | control_flow=%s | subgoals=%s",
                    goal, control_flow, subgoals,
                )
                sub_snapshot = {**env_snapshot, "observation": current_obs_text}
                nested_node = self._make_goal_node(
                    goal=goal,
                    env_snapshot=sub_snapshot,
                    depth=depth,
                    action_history=[],
                    node_role="goal",
                )
                # Pre-seed candidate so the outer MCTS evaluates the LLM's suggestion first
                llm_action = DecompositionAction(control_flow, subgoals, prior_prob=0.80)
                nested_node.get_state().generated_candidates = [llm_action]
                result = self.execute_goal_with_mcts(nested_node)
                return result["success"], result["reward"], result["steps"], result.get("trajectory", [])

            elif next_step_class == "Error":
                self._cur_decision_id += 1
                try_count += 1
                if try_count >= max_try:
                    self.logger.warning(
                        "ReactTree max errors reached | goal='%s'", goal
                    )
                    return False, -0.5, steps_taken, trajectory

            else:
                self.logger.warning(
                    "ReactTree unknown step class '%s' | goal='%s'",
                    next_step_class, goal,
                )
                break

        return False, -0.3, steps_taken, trajectory

    def _simulate_goal(self, goal: str, env_snapshot: Dict, depth: int, action_history: List[str]) -> Dict[str, Any]:
        normalized_goal = self._normalize_goal_text(goal)
        if not normalized_goal:
            return {
                "success": False,
                "reward": -1.0,
                "steps": 1,
                "mode": "empty_goal",
                "goal": goal,
                "trajectory": [],
            }

        if depth >= self._max_depth():
            return {
                "success": False,
                "reward": -1.0,
                "steps": 1,
                "mode": "max_depth_failure",
                "goal": goal,
                "trajectory": [],
            }

        if self._is_primitive_goal(goal):
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(goal, env_snapshot.get("observation", ""))
            return {
                "success": success,
                "reward": reward,
                "steps": steps,
                "mode": "primitive",
                "goal": goal,
                "trajectory": trajectory,
            }

        subgoal_root = self._make_goal_node(
            goal=goal,
            env_snapshot=env_snapshot,
            depth=depth,
            action_history=action_history,
            executed_steps=[],
            node_role="goal",
        )

        best_subgoal_node, _ = self.outer_monte_carlo_tree_search(subgoal_root)
        best_action = getattr(best_subgoal_node, "decomposition_action", None)
        if best_action is None:
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(goal, env_snapshot.get("observation", ""))
            return {
                "success": success,
                "reward": reward,
                "steps": steps,
                "mode": "primitive_fallback",
                "goal": goal,
                "trajectory": trajectory,
            }

        if not self._has_decomposition_progress(goal, best_action.subgoals):
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(goal, env_snapshot.get("observation", ""))
            return {
                "success": success,
                "reward": reward,
                "steps": steps,
                "mode": "no_progress_fallback",
                "goal": goal,
                "trajectory": trajectory,
            }

        nested_result = self._evaluate_decomposition_action(
            best_action,
            env_snapshot=env_snapshot,
            depth=depth + 1,
            action_history=action_history + [best_action.signature()],
        )
        nested_result["mode"] = "decomposed"
        nested_result["goal"] = goal
        nested_result["trajectory"] = list(nested_result.get("simulated_actions", []))
        nested_result["chosen_action"] = {
            "control_flow": best_action.control_flow,
            "subgoals": list(best_action.subgoals),
            "prior_prob": best_action.prior_prob,
        }
        return nested_result

    def _evaluate_decomposition_action(
        self,
        action: DecompositionAction,
        env_snapshot: Dict,
        depth: int,
        action_history: List[str],
    ) -> Dict[str, Any]:
        total_reward = 0.0
        total_steps = 0
        success_count = 0
        fail_count = 0
        subgoal_results: List[Dict[str, Any]] = []
        simulated_actions: List[str] = []  # Collect all actions from all subgoal simulations

        for subgoal in action.subgoals:
            result = self._simulate_goal(
                subgoal,
                env_snapshot=env_snapshot,
                depth=depth,
                action_history=action_history,
            )
            subgoal_results.append(result)
            total_reward += float(result["reward"])
            total_steps += int(result["steps"])
            # Collect simulated actions/trajectory from this subgoal
            if "trajectory" in result:
                simulated_actions.extend(result["trajectory"])
            # Also collect from nested subgoal_results if present
            subgoal_nested = result.get("subgoal_results", [])
            for nested_result in subgoal_nested:
                if "trajectory" in nested_result:
                    simulated_actions.extend(nested_result["trajectory"])

            if result["success"]:
                success_count += 1
            else:
                fail_count += 1

            self.logger.info(
                "Recursive rollout | goal='%s' | subgoal='%s' | mode=%s | success=%s | reward=%.3f | steps=%d",
                env_snapshot.get("observation", ""),
                subgoal,
                result.get("mode", "unknown"),
                result["success"],
                result["reward"],
                result["steps"],
            )

            if action.control_flow == "sequence" and not result["success"]:
                break
            if action.control_flow == "fallback" and result["success"]:
                total_reward += 0.25
                break

        total_reward += 0.4 * success_count
        total_reward -= 0.5 * fail_count
        total_reward -= 0.02 * total_steps

        overall_success = fail_count == 0 and success_count >= max(1, len(action.subgoals))
        if action.control_flow == "fallback":
            overall_success = success_count > 0

        return {
            "success": overall_success,
            "reward": total_reward,
            "steps": total_steps,
            "success_count": success_count,
            "fail_count": fail_count,
            "subgoal_results": subgoal_results,
            "simulated_actions": simulated_actions,
        }

    def execute_goal_with_mcts(self, goal_node: NodeType) -> Dict[str, Any]:
        state = goal_node.get_state()
        goal = state.goal
        role = self._node_role(goal_node)
        self.logger.info(
            ">> execute_goal_with_mcts | role=%s | goal='%s' | depth=%d",
            role, goal, state.depth,
        )

        if not goal:
            self._record_terminal_result(goal_node, -1.0, False)
            self.logger.info("  -> empty goal, returning failure")
            return {
                "success": False,
                "reward": -1.0,
                "steps": 1,
                "mode": "empty_goal",
                "goal": goal,
                "chosen_action": None,
                "subgoal_results": [],
                "trajectory": [],
            }

        if state.depth >= self._max_depth():
            success, reward, steps, trajectory = False, -1.0, 1, []
            reason = "max_depth_failure"
            self._record_terminal_result(goal_node, reward, success)
            goal_node.get_state().trajectory = trajectory
            self.logger.info(
                "  -> %s | goal='%s' | success=%s | reward=%.3f | steps=%d",
                reason, goal, success, reward, steps,
            )
            return {
                "success": success,
                "reward": reward,
                "steps": steps,
                "mode": reason,
                "goal": goal,
                "chosen_action": None,
                "subgoal_results": [],
                "trajectory": trajectory,
            }

        if self._is_primitive_goal(goal):
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(
                goal,
                state.env_snapshot.get("observation", ""),
            )
            reason = "primitive"
            self._record_terminal_result(goal_node, reward, success)
            goal_node.get_state().trajectory = trajectory
            self.logger.info(
                "  -> %s | goal='%s' | success=%s | reward=%.3f | steps=%d | trajectory=%s",
                reason, goal, success, reward, steps, trajectory,
            )
            return {
                "success": success,
                "reward": reward,
                "steps": steps,
                "mode": reason,
                "goal": goal,
                "chosen_action": None,
                "subgoal_results": [],
                "trajectory": trajectory,
            }

        best_node, _ = self.outer_monte_carlo_tree_search(goal_node)
        chosen_action = getattr(best_node, "decomposition_action", None)
        goal_node.selected_child = best_node
        self.logger.info(
            "  -> MCTS chose | control_flow=%s | subgoals=%s | q=%.3f | visits=%d",
            chosen_action.control_flow if chosen_action else "(none)",
            chosen_action.subgoals if chosen_action else [],
            best_node.quality_value,
            best_node.visit_count,
        )

        if chosen_action is None:
            success, reward, steps, trajectory = self._execute_subgoal_with_reactree(
                goal, state.env_snapshot, state.depth
            )
            self._record_terminal_result(goal_node, reward, success)
            goal_node.get_state().trajectory = trajectory
            self.logger.info(
                "  -> primitive_fallback (MCTS found no decomp) | goal='%s' | success=%s | reward=%.3f | trajectory=%s",
                goal, success, reward, trajectory,
            )
            return {
                "success": success,
                "reward": reward,
                "steps": steps,
                "mode": "primitive_fallback",
                "goal": goal,
                "chosen_action": None,
                "subgoal_results": [],
                "trajectory": trajectory,
            }

        self.logger.info(
            "  -> executing committed decomposition | control_flow=%s | %d subgoals: %s",
            chosen_action.control_flow, len(chosen_action.subgoals), chosen_action.subgoals,
        )
        execution_result = self.execute_committed_decomposition(best_node)
        state.trajectory = list(execution_result.get("trajectory", []))
        state.final_success_trajectory = copy.deepcopy(execution_result.get("final_success_trajectory", []))
        state.succeed_subgoals += execution_result["success_count"]
        state.failed_subgoals += execution_result["fail_count"]
        state.is_terminal = execution_result["success"]
        self.logger.info(
            "<< execute_goal_with_mcts DONE | goal='%s' | success=%s | reward=%.3f | succeeded=%d | failed=%d",
            goal, execution_result["success"], execution_result["reward"],
            execution_result["success_count"], execution_result["fail_count"],
        )
        execution_result.update(
            {
                "mode": "mcts",
                "goal": goal,
                "chosen_action": self._format_action_metadata(chosen_action),
            }
        )
        return execution_result

    def _generate_next_candidate(self, state: DecompositionState) -> Optional[DecompositionAction]:
        """Lazily generate one new decomposition candidate for *state*.

        Strategy:
        - Use only LLM Expand outputs.
        - Never fall back to heuristic text splitting.
        - Retry a few times per candidate request to obtain a unique decomposition.
        """
        if state.candidate_generation_done:
            return None

        obs = state.env_snapshot.get("observation", "") if state.env_snapshot else ""
        task_type = state.env_snapshot.get("task_type", "unknown") if state.env_snapshot else "unknown"
        next_index = len(state.generated_candidates)
        seen_subgoal_keys = {self._candidate_subgoal_key(c) for c in state.generated_candidates}

        candidate: Optional[DecompositionAction] = None
        max_llm_attempts = 4
        for _ in range(max_llm_attempts):
            sampled = self._llm_generate_expand_candidate(
                state.goal, obs, task_type, state.depth
            )
            if sampled is None:
                continue
            if self._candidate_subgoal_key(sampled) in seen_subgoal_keys:
                continue
            candidate = sampled
            break

        if candidate is None:
            state.candidate_generation_done = True
            self.logger.info(
                "No unique LLM decomposition candidate for goal='%s' (index=%d). Marking fully generated.",
                state.goal,
                next_index,
            )
            return None

        state.generated_candidates.append(candidate)
        if len(state.generated_candidates) >= state.max_candidate_count:
            state.candidate_generation_done = True

        self.logger.info(
            "Generated outer candidate %d/%d for goal '%s' | control_flow=%s | prior=%.2f | subgoals=%s",
            len(state.generated_candidates),
            state.max_candidate_count,
            state.goal,
            candidate.control_flow,
            candidate.prior_prob,
            candidate.subgoals,
        )
        return candidate

    def _is_fully_expanded(self, node: NodeType) -> bool:
        state = node.get_state()
        if not state.candidate_generation_done:
            return False
        return len(self._get_decomposition_children(node)) >= len(state.get_available_actions())
    
    def outer_tree_policy(self, node: NodeType) -> NodeType:
        """
        Outer MCTS tree policy: Selection + Expansion at decomposition level.
        
        Similar to standard MCTS tree_policy but operates on decomposition actions.
        """
        while not self._is_state_terminal(node.get_state()):
            role = self._node_role(node)
            state = node.get_state()
            self.logger.debug(
                "tree_policy | role=%s | depth=%d | fully_expanded=%s | terminal=%s",
                role, state.depth, self._is_fully_expanded(node), state.is_terminal,
            )
            # Decomposition nodes are leaves in the outer tree — never expand further.
            if role == "decomposition":
                self.logger.info(
                    "tree_policy | reached decomposition leaf | depth=%d | control_flow=%s",
                    state.depth,
                    getattr(getattr(node, "decomposition_action", None), "control_flow", "?"),
                )
                return node
            if self._is_fully_expanded(node):
                # Select best child using UCT + LLM prior
                next_node = self.outer_best_child(node, is_exploration=True)
                if next_node is None or next_node is node:
                    # No progress possible (no decomp children or all exhausted)
                    self.logger.info(
                        "tree_policy | no progress from best_child (node returned itself) | depth=%d",
                        state.depth,
                    )
                    return node
                self.logger.info(
                    "tree_policy | selected child | role=%s | depth=%d | control_flow=%s",
                    self._node_role(next_node),
                    next_node.get_state().depth,
                    getattr(getattr(next_node, "decomposition_action", None), "control_flow", "?"),
                )
                node = next_node
            else:
                # Expand: materialize one new decomposition candidate.
                self.logger.info(
                    "tree_policy | expanding new candidate at depth=%d | goal='%s'",
                    state.depth, state.goal,
                )
                expanded_node = self.outer_expand(node)
                return expanded_node
        self.logger.info(
            "tree_policy | terminal node reached | depth=%d | is_terminal=%s",
            node.get_state().depth, node.get_state().is_terminal,
        )
        return node
    
    def outer_expand(self, node: NodeType) -> NodeType:
        """
        Outer MCTS: Expansion
        Outer expand: generate a new decomposition candidate.
        Gets decomposition candidates from LLM and adds one as child.
        """
        state = node.get_state()
        tried_paths = self._get_decomposition_paths(node)
        selected = self.outer_expand_action(state, tried_paths)

        if selected is None:
            return node

        child_state = state.clone()
        child_state.depth += 1
        child_state.action_history.append(selected.signature())
        child_state.executed_steps.append(f"Expand({selected.control_flow}): {selected.subgoals}")

        child_node = Node()
        child_node.set_state(child_state)
        child_node.node_role = "decomposition"
        child_node.decomposition_action = selected
        node.add_child(child_node)
        self.logger.info(
            "Expanded outer node | parent_depth=%d | child_depth=%d | children_now=%d | control_flow=%s | subgoals=%s",
            state.depth,
            child_state.depth,
            len(node.get_children()),
            selected.control_flow,
            selected.subgoals,
        )
        return child_node

    def outer_expand_action(self, state: DecompositionState, tried_paths: List[List[str]]) -> Optional[DecompositionAction]:
        current_prefix = list(state.action_history)

        for candidate in state.get_available_actions():
            candidate_path = current_prefix + [candidate.signature()]
            if candidate_path not in [path[: len(candidate_path)] for path in tried_paths]:
                return candidate

        while not state.candidate_generation_done:
            candidate = self._generate_next_candidate(state)
            if candidate is None:
                break
            candidate_path = current_prefix + [candidate.signature()]
            if candidate_path not in [path[: len(candidate_path)] for path in tried_paths]:
                return candidate

        return None
    
    def _restore_env_for_simulation(self) -> None:
        """Restore the environment to the initial scene state before a simulation rollout.

        Called at the start of each outer_default_policy invocation so that every
        candidate decomposition is evaluated from the same starting world state,
        making their reward signals directly comparable.
        """
        info = self._sim_restore_info
        if info is None:
            self.logger.warning(
                "_restore_env_for_simulation: no restore info set; "
                "simulations may start from a dirty environment state"
            )
            return
        try:
            self.env.restore_scene(
                info["object_poses"],
                info["object_toggles"],
                info["dirty_and_empty"],
            )
            self.env.step(info["init_action"])
            self.env.set_task(info["traj_data"], info["model_args"], reward_type="dense")
            if getattr(getattr(self.cfg, "llm_agent", object()), "working_memory", False):
                self.env.reset_working_memory()
            self.logger.info("_restore_env_for_simulation: scene restored to initial state")
        except Exception as exc:
            self.logger.warning("_restore_env_for_simulation failed: %s", exc)

    def outer_default_policy(self, node: NodeType) -> Tuple[float, NodeType]:
        """
        Outer default policy (simulation): rollout decomposition.

        Takes the decomposition in this node and:
        1. Executes each subgoal in order (using inner MCTS for primitive actions).
        2. Accumulates reward from successes/failures.
        3. Returns (reward, final_node).
        """
        action = getattr(node, "decomposition_action", None)
        if action is None:
            # Root or no action selected yet.
            return 0.0, node

        # Restore the environment to its initial state before every simulation so
        # that each candidate decomposition is evaluated from the same world state.
        self._restore_env_for_simulation()

        evaluation = self._evaluate_decomposition_action(
            action,
            env_snapshot=node.get_state().env_snapshot,
            depth=node.get_state().depth,
            action_history=list(node.get_state().action_history),
        )
        total_reward = evaluation["reward"]
        total_steps = evaluation["steps"]
        success_count = evaluation["success_count"]
        fail_count = evaluation["fail_count"]

        state = node.get_state()
        state.succeed_subgoals += success_count
        state.failed_subgoals += fail_count
        # Always mark as terminal after simulation — avoids re-simulation and tree_policy loops.
        state.is_terminal = True

        self.logger.info(
            "Outer default policy result | control_flow=%s | success=%s | successes=%d | failures=%d | total_steps=%d | total_reward=%.3f",
            action.control_flow,
            evaluation["success"],
            success_count,
            fail_count,
            total_steps,
            total_reward,
        )
        
        # Log simulated actions to node's mcts_attempts
        simulated_actions = evaluation.get("simulated_actions", [])
        node.get_state().mcts_attempts.append({
            "decomposition": action.signature(),
            "reward": total_reward,
            "simulated_actions": simulated_actions,
        })
        
        return total_reward, node
    
    def inner_mcts_solve_subgoal(self, 
                                  subgoal: str,
                                  current_obs: str) -> Tuple[bool, float, int, List[str]]:
        """
        Inner MCTS: Search for primitive action sequence to solve one subgoal.
        
        Uses existing MCTSAlgorithm to explore action trees for primitive skills.
        This replaces the current one-shot ReAct loop.
        
        Args:
            subgoal: single decomposed subgoal (e.g., "pick up apple")
            current_obs: current environment observation text
        
        Returns:
            (success: bool, reward: float, steps_used: int, action_sequence: List[str])
        """
        success, reward, action_sequence = self.action_mcts.solve_subgoal(subgoal, current_obs)
        steps_used = len(action_sequence) if action_sequence else 1
        self.logger.info(
            "Inner MCTS solve | subgoal='%s' | success=%s | reward=%.3f | steps=%d | actions=%s",
            subgoal,
            success,
            reward,
            steps_used,
            action_sequence,
        )
        return success, reward, steps_used, action_sequence
    
    def outer_best_child(self, node: NodeType, is_exploration: bool) -> NodeType:
        """
        Outer select: UCT-based child selection incorporating LLM prior.
        
        Similar to standard MCTS but with LLM prior weighted.
        """
        children = self._get_decomposition_children(node)
        if not children:
            return node

        best_score = -float("inf")
        best_node = children[0]

        parent_visits = max(1, node.visit_count)
        c = 1.0 / math.sqrt(2.0) if is_exploration else 0.0

        for child in children:
            q = child.quality_value
            n = child.visit_count
            exploitation = q / n if n > 0 else 0.0
            exploration = c * math.sqrt(math.log(parent_visits + 1) / (n + 1))

            cand = getattr(child, "decomposition_action", None)
            prior = cand.prior_prob if cand is not None else 0.5
            score = exploitation + prior * exploration

            if score > best_score:
                best_score = score
                best_node = child

        selected_action = getattr(best_node, "decomposition_action", None)
        if selected_action is not None:
            self.logger.info(
                "Selected outer child | exploration=%s | score=%.3f | q=%.3f | visits=%d | control_flow=%s | subgoals=%s",
                is_exploration,
                best_score,
                best_node.quality_value,
                best_node.visit_count,
                selected_action.control_flow,
                selected_action.subgoals,
            )

        return best_node
    
    def outer_backup(self, node: NodeType, reward: float):
        """Outer backup: propagate reward up the tree."""
        gamma = float(getattr(getattr(self.cfg, "hmt", object()), "gamma", 0.95))
        discount = 1.0
        cur = node

        while cur is not None:
            cur.increment_visit_count()
            cur.update_quality_value(reward * discount)
            self.logger.info(
                "Backup node | reward=%.3f | discount=%.3f | new_q=%.3f | visits=%d",
                reward,
                discount,
                cur.quality_value,
                cur.visit_count,
            )
            discount *= gamma
            cur = cur.get_parent()
    
    def outer_monte_carlo_tree_search(self, node: NodeType) -> Tuple[NodeType, List[NodeType]]:
        """
        Full outer MCTS: tree_policy -> default_policy -> backup.
        
        Returns best child and all expanded nodes.
        """
        root_state = node.get_state()
        self.logger.info(
            "=== outer_monte_carlo_tree_search START | goal='%s' | depth=%d | budget=%d ===",
            root_state.goal, root_state.depth, self.outer_budget,
        )
        all_expand_nodes = []
        for i in range(self.outer_budget):
            self.logger.info(
                "--- Outer MCTS iteration %d/%d | goal='%s' | depth=%d | decomp_children=%d ---",
                i + 1, self.outer_budget, root_state.goal, root_state.depth,
                len(self._get_decomposition_children(node)),
            )
            
            # Tree policy (selection + expansion)
            expand_node = self.outer_tree_policy(node)
            expand_action = getattr(expand_node, "decomposition_action", None)
            self.logger.info(
                "  tree_policy result | role=%s | depth=%d | action=%s",
                self._node_role(expand_node),
                expand_node.get_state().depth,
                expand_action.signature() if expand_action else "(none — root/fallback)",
            )
            all_expand_nodes.append(expand_node)
            
            # Default policy (simulation with inner MCTS rollouts)
            reward, leaf_node = self.outer_default_policy(expand_node)
            self.logger.info(
                "  default_policy reward=%.3f | node_role=%s",
                reward, self._node_role(leaf_node),
            )
            
            # Backup
            self.outer_backup(leaf_node, reward)
        
        # Select best decomposition for commitment
        best_node = self.outer_best_child(node, is_exploration=False)
        best_action = getattr(best_node, "decomposition_action", None)
        self.logger.info(
            "=== outer_monte_carlo_tree_search END | chosen: %s | subgoals=%s | q=%.3f | visits=%d ===",
            best_action.control_flow if best_action else "(none)",
            best_action.subgoals if best_action else [],
            best_node.quality_value,
            best_node.visit_count,
        )
        return best_node, all_expand_nodes
    
    def execute_committed_decomposition(self, decomp_node: NodeType) -> Dict:
        """
        Execute the chosen decomposition in the real environment.
        
        Apply Reactree semantics:
        - sequence: execute subgoals in order, fail if any fails
        - fallback: try each until one succeeds
        - parallel: execute all (but environment is single-agent, so not practical)
        
        Returns:
            {
                'success': bool,
                'subgoal_results': [{'subgoal': str, 'success': bool}, ...],
                'steps_executed': int
            }
        """
        action = getattr(decomp_node, "decomposition_action", None)
        if action is None:
            return {
                "success": False,
                "reward": -1.0,
                "subgoal_results": [],
                "steps_executed": 0,
                "success_count": 0,
                "fail_count": 0,
                "reason": "no_decomposition_action",
            }

        total_reward = 0.0
        total_steps = 0
        success_count = 0
        fail_count = 0
        subgoal_results: List[Dict[str, Any]] = []
        final_success_trajectory: List[Dict[str, Any]] = []
        flat_primitive_trajectory: List[str] = []
        state = decomp_node.get_state()

        for index, subgoal in enumerate(action.subgoals, start=1):
            step_history = list(state.action_history) + [f"subgoal[{index}]::{subgoal}"]
            step_executed = list(state.executed_steps) + [f"Subgoal[{index}/{len(action.subgoals)}]: {subgoal}"]
            step_node = self._make_goal_node(
                goal=subgoal,
                env_snapshot=state.env_snapshot,
                depth=state.depth + 1,
                action_history=step_history,
                executed_steps=step_executed,
                node_role="subgoal_step",
                subgoal_index=index,
                subgoal_text=subgoal,
            )
            decomp_node.add_child(step_node)

            self.logger.info(
                "Executing subgoal [%d/%d] | parent_goal='%s' | subgoal='%s' | control_flow=%s",
                index, len(action.subgoals), state.goal, subgoal, action.control_flow,
            )
            step_result = self.execute_goal_with_mcts(step_node)
            primitive_actions = list(step_result.get("trajectory", []))
            flat_primitive_trajectory.extend(primitive_actions)
            subgoal_results.append(
                {
                    "subgoal_index": index,
                    "subgoal": subgoal,
                    **step_result,
                }
            )
            final_success_trajectory.append(
                {
                    "subgoal_index": index,
                    "subgoal": subgoal,
                    "success": bool(step_result.get("success", False)),
                    "mode": step_result.get("mode", "unknown"),
                    "primitive_actions": primitive_actions,
                }
            )
            total_reward += float(step_result["reward"])
            total_steps += int(step_result["steps"])

            if step_result["success"]:
                success_count += 1
            else:
                fail_count += 1

            self.logger.info(
                "Subgoal [%d/%d] result | parent_goal='%s' | subgoal='%s' | mode=%s | success=%s | reward=%.3f | steps=%d | cumulative: ok=%d fail=%d",
                index, len(action.subgoals), state.goal, subgoal,
                step_result.get("mode", "unknown"),
                step_result["success"],
                step_result["reward"],
                step_result["steps"],
                success_count, fail_count,
            )
            self.logger.info(
                "Subgoal [%d/%d] primitive actions | subgoal='%s' | actions=%s",
                index,
                len(action.subgoals),
                subgoal,
                primitive_actions,
            )

            if action.control_flow == "sequence" and not step_result["success"]:
                # Keep executing remaining subgoals so all LLM subgoal traces are visible.
                # Final sequence success is still computed strictly at the end.
                self.logger.info(
                    "Sequence subgoal failed at index=%d, continuing execution of remaining subgoals for traceability",
                    index,
                )
            if action.control_flow == "fallback" and step_result["success"]:
                total_reward += 0.25
                break

        total_reward += 0.4 * success_count
        total_reward -= 0.5 * fail_count
        total_reward -= 0.02 * total_steps

        overall_success = fail_count == 0 and success_count >= max(1, len(action.subgoals))
        if action.control_flow == "fallback":
            overall_success = success_count > 0

        state.succeed_subgoals += success_count
        state.failed_subgoals += fail_count
        state.is_terminal = overall_success
        state.trajectory = list(flat_primitive_trajectory)
        state.final_success_trajectory = copy.deepcopy(final_success_trajectory)

        self.logger.info(
            "Committed decomposition primitive trajectory | goal='%s' | overall_success=%s | subgoals=%s",
            state.goal,
            overall_success,
            final_success_trajectory,
        )

        return {
            "success": overall_success,
            "subgoal_results": subgoal_results,
            "reward": total_reward,
            "steps": total_steps,
            "steps_executed": total_steps,
            "success_count": success_count,
            "fail_count": fail_count,
            "control_flow": action.control_flow,
            "trajectory": flat_primitive_trajectory,
            "final_success_trajectory": final_success_trajectory,
        }


# ============================================================================
# INTEGRATION: Wrapper for existing AlfredReactree
# ============================================================================

class AlfredReactreeWithHMT(AlfredReactree):
    """
    Extended AlfredReactree that uses outer MCTS for decomposition search.
    
    Replaces the one-shot Expand logic with outer MCTS exploration.
    """
    
    def __init__(self, cfg, llm_agent, env, use_hmt: bool = True):
        super().__init__(cfg, llm_agent, env)
        self.logger = logger
        self.use_hmt = use_hmt
        
        if self.use_hmt:
            hmt_cfg = getattr(cfg, "hmt", None)
            outer_budget = int(getattr(hmt_cfg, "outer_budget", 2))
            inner_budget = int(getattr(hmt_cfg, "inner_budget", 3))
            decomp_candidate_count = int(getattr(hmt_cfg, "decomp_candidate_count", 3))
            self.logger.info(
                "Initializing HMT | outer_budget=%d | inner_budget=%d | decomp_candidate_count=%d",
                outer_budget,
                inner_budget,
                decomp_candidate_count,
            )
            self.outer_mcts = OuterMCTSPlanner(
                cfg=cfg,
                llm_agent=llm_agent,
                env=env,
                outer_budget=outer_budget,
                inner_budget=inner_budget,
                decomp_candidate_count=decomp_candidate_count,
            )
    
    def collect_llm(self, task_d, args_dict):
        """
        Entry point: original AlfredReactree.collect_llm but with HMT.
        """
        if self.use_hmt:
            return self.collect_llm_with_hmt(task_d, args_dict)

        # Fallback to original single-shot ReacTree
        return super().collect_llm(task_d, args_dict)
    
    def collect_llm_with_hmt(self, task_d, args_dict):
        """
        Hierarchical collection: outer decomposition search + inner action search.
        """
        self.logger.info("Starting HMT collection | task=%s | repeat_idx=%s", task_d.get("task"), task_d.get("repeat_idx"))
        traj_data = load_task_json(task_d)
        repeat_idx = task_d["repeat_idx"]
        nl_inst = traj_data["turk_annotations"]["anns"][repeat_idx]["task_desc"]
        model_args = dotdict(args_dict)
        self.logger.info("Natural language instruction: %s", nl_inst)

        # ---- Scene / environment setup (mirrors AlfredReactree.collect_llm) ----
        try:
            scene_num = traj_data["scene"]["scene_num"]
            object_poses = traj_data["scene"]["object_poses"]
            dirty_and_empty = traj_data["scene"]["dirty_and_empty"]
            object_toggles = traj_data["scene"]["object_toggles"]
            scene_name = "FloorPlan%d" % scene_num
            self.env.reset(scene_name)
            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            self.env.step(dict(traj_data["scene"]["init_action"]))
            self.env.set_task(traj_data, model_args, reward_type="dense")
            if self.cfg.llm_agent.working_memory:
                self.env.reset_working_memory()
        except Exception:
            import traceback
            self.logger.exception("Scene setup failed")
            traceback.print_exc()

        # Get initial observation from the env
        try:
            init_obs = self.env.init_reset(traj_data)
            init_obs_text = init_obs["text"]
        except Exception:
            self.logger.exception("env.init_reset failed; using instruction as fallback obs")
            init_obs_text = nl_inst

        # ---- Wire runtime state into the OuterMCTSPlanner ----
        self.outer_mcts._current_traj_data = traj_data
        self.outer_mcts._cur_step_id = 1
        self.outer_mcts._cur_decision_id = 1
        # Store scene restore info so outer_default_policy can reset the environment
        # to this exact initial state before each simulation rollout.
        self.outer_mcts._sim_restore_info = {
            "object_poses": traj_data["scene"]["object_poses"],
            "object_toggles": traj_data["scene"]["object_toggles"],
            "dirty_and_empty": traj_data["scene"]["dirty_and_empty"],
            "init_action": dict(traj_data["scene"]["init_action"]),
            "traj_data": traj_data,
            "model_args": model_args,
        }

        # Build root decomposition state with the real initial observation
        root_state = DecompositionState(
            goal=nl_inst,
            env_snapshot={
                "observation": init_obs_text,
                "task_type": traj_data.get("task_type", "unknown"),
            },
            depth=0,
            max_candidate_count=self.outer_mcts.decomp_candidate_count,
        )
        root_node = Node()
        root_node.set_state(root_state)
        root_node.node_role = "goal"

        execution_result = self.outer_mcts.execute_goal_with_mcts(root_node)
        chosen = execution_result.get("chosen_action")
        tree_artifacts = export_outer_tree_artifacts(
            self.cfg,
            root_node,
            task_d.get("task", nl_inst),
        )
        if chosen is not None:
            self.logger.info(
                "Chosen HMT decomposition | control_flow=%s | subgoals=%s | q=%.3f | visits=%d",
                chosen["control_flow"],
                chosen["subgoals"],
                getattr(getattr(root_node, "selected_child", root_node), "quality_value", root_node.quality_value),
                getattr(getattr(root_node, "selected_child", root_node), "visit_count", root_node.visit_count),
            )

        terminate_info = {
            "success": execution_result["success"],
            "terminate": execution_result.get("mode", "hmt"),
            "steps": execution_result["steps"],
            "nl_inst": nl_inst,
        }
        self.logger.info("HMT terminate_info: %s", terminate_info)

        if chosen is not None:
            terminate_info["hmt_outer"] = {
                "control_flow": chosen["control_flow"],
                "subgoals": chosen["subgoals"],
                "prior_prob": chosen["prior_prob"],
                "value": getattr(getattr(root_node, "selected_child", root_node), "quality_value", root_node.quality_value),
                "visits": getattr(getattr(root_node, "selected_child", root_node), "visit_count", root_node.visit_count),
                "reward": execution_result["reward"],
                "subgoal_results": execution_result["subgoal_results"],
                "tree_artifacts": tree_artifacts,
            }
        else:
            terminate_info["hmt_outer"] = {
                "control_flow": None,
                "subgoals": [],
                "prior_prob": 0.0,
                "value": 0.0,
                "visits": 0,
                "reward": execution_result["reward"],
                "subgoal_results": execution_result["subgoal_results"],
                "tree_artifacts": tree_artifacts,
            }

        return terminate_info

    def run(self, task_d, args_dict, log):
        """Evaluator-compatible entry point.

        Wraps ``collect_llm_with_hmt`` so that ``AlfredEvaluator`` can call
        ``tp.run(task_d, args_dict, log)`` without modification.  Success is
        authoritative from the environment (``env.get_goal_satisfied()``), not
        from HMT's internal tracking.
        """
        from PIL import Image as _Image
        log.info(task_d)

        # Initialise vis_log the same way AlfredReactree.run does so that
        # save_vis_log in the evaluator has a valid list to write.
        try:
            self.env.vis_log = [
                {"action": "init", "images": _Image.fromarray(self.env.last_event.frame)}
            ]
        except Exception:
            self.env.vis_log = []

        terminate_info = self.collect_llm_with_hmt(task_d, args_dict)

        # Ground-truth success from the simulator (matches what AlfredReactree.run does).
        terminate_info["success"] = self.env.get_goal_satisfied()
        return terminate_info


# ============================================================================
# INNER LEVEL: LLM adapter — provides chat() interface for MCTSAlgorithm
# ============================================================================

class _MCTSChatAdapter:
    """Wraps AlfredLlmAgent to expose the chat() interface MCTSAlgorithm.llm_inference expects.

    Captures agent.llm at construction time (before any ReAcTree prompts are loaded) as a
    clean base context.  guidance's + operator is non-mutating — each chat() call extends
    from that base without touching agent.llm or reloading the model.
    """

    def __init__(self, llm_agent) -> None:
        self._agent = llm_agent
        # Snapshot the guidance model before any prompt is loaded onto it.
        # For HuggingFace this is the bare Transformers wrapper; for OpenAI it is
        # the freshly constructed API client.  Both serve as a clean starting point.
        self._base = llm_agent.llm

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, float]:
        import ast
        import re as _re

        # prompt_template() embeds available_commands in the last user message as:
        # "The candidate actions are ['action a', 'action b', ...]"
        available_commands: List[str] = []
        for msg in reversed(messages):
            if msg["role"] == "user":
                m = _re.search(r"candidate actions are (\[.+?\])", msg["content"], _re.DOTALL)
                if m:
                    try:
                        available_commands = ast.literal_eval(m.group(1))
                    except Exception:
                        pass
                break
        if not available_commands:
            return {}

        # Build a flat prompt: message contents already carry role-specific labels
        # ("Init environment:", "Current observation:", etc.) set by prompt_template().
        prompt_str = "\n".join(msg["content"] for msg in messages) + "\n"

        try:
            import guidance
            # Extend from the clean base context — non-mutating, no model re-init.
            result = self._base + prompt_str + guidance.select(available_commands, name="mcts_action")
            return {result["mcts_action"]: 1.0}
        except Exception as exc:
            logger.warning("_MCTSChatAdapter.chat failed: %s", exc)
            return {}


# ============================================================================
# INNER LEVEL: Reuse existing MCTS for action search
# ============================================================================

class ActionMCTSWrapper:
    """
    Wrapper to use existing MCTSAlgorithm for primitive action search.
    
    Adapts existing State/Node/MCTSAlgorithm to subgoal-level search.
    """
    
    def __init__(self, 
                 cfg,
                 llm_agent,
                 env,
                 budget: int = 10):
        """
        Args:
            cfg: config
            llm_agent: LLM agent
            env: ALFRED environment
            budget: inner MCTS computation budget
        """
        self.cfg = cfg
        self.llm_agent = llm_agent
        self.env = env
        self.budget = budget
        self.mcts_unavailable_reason = ""

        # Lazy import to avoid hard dependency failures in environments where
        self.mcts = None
        try:
            try:
                from src.mcts.algorithm import MCTSAlgorithm  # type: ignore
            except ImportError:
                from mcts.algorithm import MCTSAlgorithm  # type: ignore

            self.mcts = MCTSAlgorithm(
                computation_budget=budget,
                max_sim_round_number=10,
                LLM=_MCTSChatAdapter(llm_agent),
                play_round=5,
            )

            # Replace prompt_template with a version that handles an empty action_history
            # (the original crashes with IndexError on the first inner-MCTS step).
            import types as _types

            def _prompt_template_fixed(_self_m, state, available_commands):
                try:
                    from src.llm.prompt import SYSTEM_PROMPT as _SYS  # type: ignore
                except ImportError:
                    try:
                        from llm.prompt import SYSTEM_PROMPT as _SYS  # type: ignore
                    except ImportError:
                        _SYS = "You are a robot agent. Pick the best action from the candidate list."
                msgs: List[Dict[str, Any]] = []
                if "Welcome to TextWorld, ALFRED!" in (state.obs or ""):
                    msgs.append({"role": "user", "content": f"Init environment: {state.obs}\n The candidate actions are {available_commands}"})
                else:
                    obs_list = list(getattr(state, "obs_list", [state.obs]))
                    action_history = list(getattr(state, "action_history", []))
                    for i, obs in enumerate(obs_list):
                        prefix = "Init environment: " if i == 0 else "Current observation: "
                        is_last = (i == len(obs_list) - 1)
                        if is_last:
                            msgs.append({"role": "user", "content": f"{prefix}{obs}\nThe candidate actions are {available_commands}"})
                        else:
                            msgs.append({"role": "user", "content": f"{prefix}{obs}"})
                            if i < len(action_history):
                                msgs.append({"role": "assistant", "content": action_history[i]})
                return _SYS, msgs

            self.mcts.prompt_template = _types.MethodType(_prompt_template_fixed, self.mcts)

        except Exception as e:
            self.mcts_unavailable_reason = str(e)
            logger.warning("ActionMCTSWrapper backend unavailable, will use LLM rollout fallback: %s", e)

        # _MCTSChatAdapter provides chat() so the backend is usable whenever MCTSAlgorithm loaded.
        self._can_use_mcts_backend = self.mcts is not None

    class _RolloutState:
        def __init__(self, obs: str, obs_list: List[str], action_history: List[str]):
            self.obs = obs
            self.obs_list = list(obs_list)
            self.action_history = list(action_history)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _current_skill_set(self) -> List[str]:
        try:
            scene_name = self.env.last_event.metadata.get("sceneName", "FloorPlan1")
            obs_pack = self.env.llm_skill_interact(None, scene_name)
            skill_set = self.llm_agent.update_skill_set(obs_pack)
            deduped: List[str] = []
            seen = set()
            for cmd in skill_set:
                if cmd == "look":
                    continue
                if cmd not in seen:
                    seen.add(cmd)
                    deduped.append(cmd)
            return deduped
        except Exception as exc:
            logger.warning("_current_skill_set failed: %s", exc)
            return ["done", "failure"]

    def _is_subgoal_satisfied(self, subgoal: str, action: str, obs_text: str) -> bool:
        sg = self._normalize_text(subgoal)
        act = self._normalize_text(action)
        obs = self._normalize_text(obs_text)

        if act in ALFRED_TERMINAL_ACTIONS:
            return True
        if sg and act == sg:
            return True

        for prefix in ALFRED_PRIMITIVE_ACTION_PREFIXES:
            if sg.startswith(prefix) and act.startswith(prefix):
                target = sg[len(prefix):].strip()
                return (not target) or (target in act) or (target in obs)

        return False

    def _solve_with_reactexpand_rollout(self, subgoal: str, obs: str) -> Tuple[bool, float, List[str]]:
        """LLM-only fallback rollout when MCTS backend cannot be used.

        This still uses guidance/plan_next_step loop (no heuristic score shortcuts).
        """
        logger.info("LLM fallback rollout ENTRY | subgoal='%s' | obs_preview='%s...'", subgoal, (obs or "")[:100])

        task_type = "unknown"
        try:
            if getattr(self.env, "last_event", None) is not None:
                task_type = self.env.last_event.metadata.get("taskType", "unknown")
        except Exception:
            pass

        nl_inst_info = {
            "nl_inst": subgoal,
            "message": None,
            "task_type": task_type,
            "depth": 0,
        }

        cur_obs = obs or "No observation available."
        action_history: List[str] = []
        action_prob_history: List[float] = []

        logger.info("LLM fallback initializing | task_type='%s'", task_type)
        try:
            self.llm_agent.reset(nl_inst_info, cur_obs)
            logger.info("LLM fallback reset successful")
        except Exception as exc:
            logger.warning("LLM rollout reset failed for '%s': %s", subgoal, exc)
            return False, -1.0, []

        max_steps = max(1, int(self.budget))
        think_budget = max_steps * 3
        think_count = 0
        loop_count = 0

        logger.info("LLM fallback starting action loop | subgoal='%s' | max_steps=%d | think_budget=%d", subgoal, max_steps, think_budget)

        for step_idx in range(max_steps):
            loop_count += 1
            skill_set = self._current_skill_set()
            logger.info(
                "LLM fallback loop iteration %d | subgoal='%s' | available_skills=%d | history=%s",
                loop_count, subgoal, len(skill_set) if skill_set else 0, action_history
            )

            try:
                logger.info("LLM fallback calling plan_next_step | obs_preview='%s...'", cur_obs[:80])
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class = next_step_info["next_step_class"]
                next_step = next_step_info["next_step"]
                logger.info(
                    "LLM fallback plan result | next_step_class='%s' | next_step_preview='%s'",
                    next_step_class, str(next_step)[:100]
                )
            except Exception as exc:
                logger.warning("LLM fallback plan_next_step failed for '%s': %s", subgoal, exc)
                return False, -1.0, action_history

            if next_step_class == "Think":
                think_count += 1
                logger.info(
                    "LLM fallback Think | subgoal='%s' | thought='%s' | count=%d/%d",
                    subgoal,
                    next_step,
                    think_count,
                    think_budget,
                )
                if think_count >= think_budget:
                    logger.info("LLM fallback exceeded think budget | returning None")
                    return False, -1.0, action_history
                continue

            if next_step_class == "Error":
                logger.info("LLM fallback got Error from plan_next_step")
                return False, -1.0, action_history

            if next_step_class == "Expand":
                # Use first decomposed subgoal as actionable proxy in rollout.
                raw_conditions = next_step.get("conditions", "")
                expanded = [s.strip() for s in raw_conditions.split(",") if s.strip()]
                logger.info("LLM fallback Expand | raw='%s' | decomposed=%s", raw_conditions, expanded)
                if not expanded:
                    logger.info("LLM fallback Expand yielded no subgoals")
                    return False, -1.0, action_history
                next_step_class = "Act"
                next_step = expanded[0]
                logger.info("LLM fallback converting Expand to Act | next_step='%s'", next_step)

            if next_step_class != "Act":
                logger.info("LLM fallback unexpected step class | class='%s'", next_step_class)
                return False, -1.0, action_history

            action = str(next_step)
            action_history.append(action)
            action_prob_history.append(0.5)
            logger.info("LLM fallback attempting action | action='%s' | step_total=%d", action, len(action_history))

            if action == "failure":
                logger.info("LLM fallback got failure action | exiting")
                return False, -1.0, action_history
            if action == "done":
                mean_prob = sum(action_prob_history) / max(1, len(action_prob_history))
                logger.info(
                    "LLM fallback got done action | mean_prob=%.3f | actions=%s",
                    mean_prob, action_history
                )
                return True, mean_prob, action_history

            logger.info("LLM fallback executing action environment | action='%s'", action)
            if action.startswith("recall location of "):
                target_obj = action.split("recall location of ")[1]
                cur_obs = alfred_utils.recall_working_memory(self.env.working_memory, target_obj)
                logger.info("LLM fallback recall result | target='%s' | obs_len=%d", target_obj, len(cur_obs))
            else:
                try:
                    obs_ret = self.env.llm_skill_interact(action)
                    cur_obs = obs_ret.get("message", "")
                    logger.info("LLM fallback skill result | action='%s' | obs_preview='%s...'", action, cur_obs[:80])
                except Exception as exc:
                    logger.warning("LLM fallback env action failed | action='%s' | err=%s", action, exc)
                    return False, -1.0, action_history

            self.llm_agent.add_obs(cur_obs)

            logger.info("LLM fallback checking satisfaction | subgoal='%s' | action='%s'", subgoal, action)
            is_satisfied = self._is_subgoal_satisfied(subgoal, action, cur_obs)
            logger.info("LLM fallback satisfaction result | satisfied=%s", is_satisfied)

            if is_satisfied:
                mean_prob = sum(action_prob_history) / max(1, len(action_prob_history))
                logger.info(
                    "LLM fallback satisfaction EXIT | subgoal='%s' | mean_prob=%.3f | actions=%s",
                    subgoal, mean_prob, action_history
                )
                return True, mean_prob, action_history

        mean_prob = sum(action_prob_history) / max(1, len(action_prob_history)) if action_prob_history else 0.0
        logger.info(
            "LLM fallback max steps EXIT | subgoal='%s' | mean_prob=%.3f | total_actions=%d | actions=%s",
            subgoal, mean_prob, len(action_history), action_history
        )
        return False, -mean_prob, action_history
    
    def solve_subgoal(self, subgoal: str, obs: str) -> Tuple[bool, float, List[str]]:
        """
        Solve one subgoal using inner MCTS action search.
        
        Args:
            subgoal: goal string (e.g., "pick up apple")
            obs: current observation
        
        Returns:
            (success, reward, action_sequence)
        """
        sg = (subgoal or "").strip()
        if not sg:
            logger.info("Inner MCTS solve_subgoal | subgoal=EMPTY | returning early")
            return False, -1.0, []

        logger.info("Inner MCTS ENTRY | subgoal='%s' | obs_preview='%s...'", sg, (obs or "")[:150])

        if not self._can_use_mcts_backend:
            logger.info(
                "Inner rollout using LLM fallback for '%s' (MCTS backend unavailable or incompatible: %s)",
                sg,
                self.mcts_unavailable_reason or "llm_agent has no chat()",
            )
            result = self._solve_with_reactexpand_rollout(sg, obs)
            logger.info("Inner MCTS FALLBACK EXIT | subgoal='%s' | success=%s | reward=%.3f | steps=%d", sg, result[0], result[1], len(result[2]))
            return result

        action_history: List[str] = []
        action_prob_history: List[float] = []
        obs_list: List[str] = [obs or "No observation available."]
        cur_obs = obs_list[-1]
        max_steps = max(1, int(self.budget))

        logger.info("Inner MCTS starting search | subgoal='%s' | max_steps=%d | budget=%.1f", sg, max_steps, self.budget)

        for step_idx in range(max_steps):
            available_commands = self._current_skill_set()
            if not available_commands:
                logger.info("Inner MCTS no available commands for subgoal '%s' at step %d/%d", sg, step_idx + 1, max_steps)
                return False, -1.0, action_history

            logger.info(
                "Inner MCTS step %d/%d START | subgoal='%s' | available_commands=%d | history_so_far=%s",
                step_idx + 1, max_steps, sg, len(available_commands), action_history
            )

            state = self._RolloutState(
                obs=cur_obs,
                obs_list=obs_list,
                action_history=action_history,
            )
            logger.info("Inner MCTS calling LLM inference | state_obs_preview='%s...'", cur_obs[:80])
            action, action_prob = self.mcts.llm_inference(state, available_commands)
            action_history.append(action)
            action_prob_history.append(float(action_prob))

            logger.info(
                "Inner MCTS step %d/%d | subgoal='%s' | action='%s' | prob=%.3f | history=%s",
                step_idx + 1, max_steps, sg, action, float(action_prob), action_history,
            )

            if action == "failure":
                logger.info("Inner MCTS early exit | action='failure' | step=%d/%d | subgoal='%s'", step_idx + 1, max_steps, sg)
                return False, -1.0, action_history
            if action == "done":
                mean_prob = sum(action_prob_history) / max(1, len(action_prob_history))
                logger.info(
                    "Inner MCTS early exit | action='done' | step=%d/%d | subgoal='%s' | mean_prob=%.3f",
                    step_idx + 1, max_steps, sg, mean_prob
                )
                return True, mean_prob, action_history

            logger.info("Inner MCTS executing action | action='%s' | type=%s", action,
                       "recall" if action.startswith("recall location of ") else "skill")

            if action.startswith("recall location of "):
                target_obj = action.split("recall location of ")[1]
                cur_obs = alfred_utils.recall_working_memory(self.env.working_memory, target_obj)
                logger.info("Inner MCTS recall result | target='%s' | obs_preview='%s...'", target_obj, cur_obs[:80])
            else:
                try:
                    logger.info("Inner MCTS env.llm_skill_interact starting | action='%s'", action)
                    obs_ret = self.env.llm_skill_interact(action)
                    cur_obs = obs_ret.get("message", "")
                    logger.info("Inner MCTS env.llm_skill_interact result | new_obs_preview='%s...'", cur_obs[:80])
                except Exception as exc:
                    logger.warning("Inner MCTS env action failed | action='%s' | err=%s", action, exc)
                    return False, -1.0, action_history

            obs_list.append(cur_obs)
            self.llm_agent.add_obs(cur_obs)

            logger.info("Inner MCTS checking satisfaction | subgoal='%s' | action='%s'", sg, action)
            is_satisfied = self._is_subgoal_satisfied(sg, action, cur_obs)
            logger.info("Inner MCTS satisfaction check result | subgoal='%s' | satisfied=%s", sg, is_satisfied)

            if is_satisfied:
                mean_prob = sum(action_prob_history) / max(1, len(action_prob_history))
                logger.info(
                    "Inner MCTS satisfaction exit | subgoal='%s' | step=%d/%d | mean_prob=%.3f | actions=%s",
                    sg, step_idx + 1, max_steps, mean_prob, action_history
                )
                return True, mean_prob, action_history

            logger.info(
                "Inner MCTS step %d/%d END | subgoal='%s' | continuing to next action",
                step_idx + 1, max_steps, sg
            )

        mean_prob = sum(action_prob_history) / max(1, len(action_prob_history)) if action_prob_history else 0.0
        logger.info(
            "Inner MCTS max steps reached | subgoal='%s' | mean_prob=%.3f | actions=%s",
            sg, mean_prob, action_history
        )
        return False, -mean_prob, action_history


# ============================================================================
# UTILITY: logging and debugging
# ============================================================================
# ============================================================================
# MAIN: Example pseudocode flow
# ============================================================================

def _main_impl(cfg):
    import pprint
    import time
    import datetime
    from tqdm import tqdm
    from alfred.alfred_env import ThorConnector
    from alfred.alfred_llm_agent import AlfredLlmAgent
    from alfred.utils import dotdict, save_vis_log
    from alfred.data.preprocess import Dataset

    log_path = _configure_logging(cfg)

    try:
        from omegaconf import OmegaConf  # type: ignore
        logger.info("Loaded config:\n%s", OmegaConf.to_yaml(cfg))
    except Exception:
        logger.info("Loaded config object: %s", cfg)

    splits_path = getattr(getattr(cfg, "alfred", object()), "splits", "alfred/data/splits/oct21.json")
    args_dict = {
        "data": "alfred/data/json_2.1.0",
        "pframe": 300,
        "fast_epoch": False,
        "use_templated_goals": False,
        "dout": "exp/model",
        "pp_folder": "pp",
        "reward_config": "alfred/models/config/rewards.json",
        "max_steps": 100,
    }

    if not os.path.exists(splits_path):
        logger.error("Split file not found: %s", splits_path)
        return

    if not getattr(cfg.dataset, "collect_dir", None):
        cfg.dataset.collect_dir = os.path.join("output", "hmt_collect")
    os.makedirs(cfg.dataset.collect_dir, exist_ok=True)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    pprint.pprint({k: len(v) for k, v in splits.items()})
    logger.info("Loaded dataset splits from %s", splits_path)

    exp_type = getattr(cfg, "exp_type", "collect_llm")

    # ------------------------------------------------------------------ #
    # EVALUATE mode: loop over full valid_seen / valid_unseen split        #
    # ------------------------------------------------------------------ #
    if exp_type == "evaluate":
        eval_set = getattr(getattr(cfg, "dataset", object()), "eval_set", "valid_seen")
        if eval_set not in splits:
            logger.error("eval_set '%s' not found in splits. Available: %s", eval_set, list(splits.keys()))
            return

        files = list(splits[eval_set])

        # Optional subset for faster debug runs
        eval_portion = int(getattr(getattr(cfg, "alfred", object()), "eval_portion_in_percent", 100))
        if eval_portion < 100:
            seed = int(getattr(getattr(cfg, "alfred", object()), "random_seed_for_eval_subset", 1))
            random.seed(seed)
            n_sample = max(1, int(len(files) * eval_portion / 100))
            files = random.sample(files, n_sample)
            logger.info("Using %d/%d tasks (%d%% subset, seed=%d)", n_sample, len(splits[eval_set]), eval_portion, seed)

        # One-time preprocessing
        number_of_dirs = len(list(os.listdir(args_dict["data"]))) if os.path.exists(args_dict["data"]) else 0
        if number_of_dirs < 50:
            logger.info("Preprocessing dataset (one-time)...")
            dataset = Dataset(dotdict(args_dict), None)
            dataset.preprocess_splits(splits)

        env = ThorConnector(cfg=cfg, x_display=cfg.alfred.x_display)
        llm_agent = AlfredLlmAgent(cfg)
        planner = AlfredReactreeWithHMT(cfg, llm_agent, env, use_hmt=True)

        results = []
        start = time.time()
        save_path = cfg.out_dir
        total_tasks = len(files)

        # Per-task-type counters for the summary table
        from collections import defaultdict
        type_tried: dict = defaultdict(int)
        type_ok:    dict = defaultdict(int)

        def _print_summary_table(results_so_far, elapsed_sec):
            n_done  = len(results_so_far)
            n_ok    = sum(1 for r in results_so_far if r["success"])
            n_fail  = n_done - n_ok
            sr      = n_ok / n_done * 100 if n_done else 0.0
            elapsed = str(datetime.timedelta(seconds=int(elapsed_sec)))

            # Header
            sep  = "+" + "-"*8 + "+" + "-"*30 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*8 + "+"
            hdr  = "| {:^6} | {:^28} | {:^6} | {:^6} | {:^6} | {:^6} |".format(
                       "Done", "Task Type", "Tried", "OK", "Fail", "SR%")
            logger.info(sep)
            logger.info("| {:^74} |".format(
                f"HMT Progress  [{eval_set}]  {n_done}/{total_tasks}  |  SR {sr:.1f}%  |  {elapsed}"))
            logger.info(sep)
            logger.info(hdr)
            logger.info(sep)

            # Per-type rows (sorted by task type name)
            for ttype in sorted(type_tried.keys()):
                tried = type_tried[ttype]
                ok    = type_ok[ttype]
                fail  = tried - ok
                t_sr  = ok / tried * 100 if tried else 0.0
                logger.info("| {:^6} | {:<28} | {:^6} | {:^6} | {:^6} | {:^6.1f} |".format(
                    "", ttype[:28], tried, ok, fail, t_sr))

            # Total row
            logger.info(sep)
            logger.info("| {:^6} | {:<28} | {:^6} | {:^6} | {:^6} | {:^6.1f} |".format(
                n_done, "TOTAL", n_done, n_ok, n_fail, sr))
            logger.info(sep)

        for i, task_d in enumerate(tqdm(files)):
            terminate_info = planner.run(task_d, args_dict, logger)

            result = {
                "task": task_d["task"],
                "repeat_idx": task_d["repeat_idx"],
                "success": terminate_info["success"],
                "nl_inst": terminate_info["nl_inst"],
            }

            task_name = task_d["task"]
            task_type = task_name.split("/")[0]
            trial_num = task_name.split("/")[1]
            repeat_idx = task_d["repeat_idx"]
            vis_prefix = "success" if result["success"] else "failure"
            vis_log_name = f"{vis_prefix}_{task_type}_{trial_num}_ann_{repeat_idx}"
            save_vis_log(cfg, env.vis_log, vis_log_name, terminate_info["nl_inst"])
            results.append(result)

            # Update per-type counters
            type_tried[task_type] += 1
            if result["success"]:
                type_ok[task_type] += 1

            # Print summary table after every task
            _print_summary_table(results, time.time() - start)

        # Final summary
        n = len(results)
        n_success = sum(1 for r in results if r["success"])
        logger.info("=== HMT EVALUATION COMPLETE ===")
        _print_summary_table(results, time.time() - start)
        logger.info("Elapsed: %s", str(datetime.timedelta(seconds=int(time.time() - start))))
        logger.info("------------------------")
        try:
            from omegaconf import OmegaConf  # type: ignore
            logger.info(OmegaConf.to_yaml(cfg))
        except Exception:
            pass

        # Save per-task results JSON next to the hydra output dir
        results_path = os.path.join(save_path, f"hmt_results_{eval_set}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"eval_set": eval_set, "n": n, "n_success": n_success,
                       "success_rate": n_success / n if n else 0.0, "results": results}, f, indent=2)
        logger.info("Results saved to %s", results_path)

    # ------------------------------------------------------------------ #
    # COLLECT_LLM mode (original single-task smoke test)                  #
    # ------------------------------------------------------------------ #
    else:
        train_key = getattr(cfg.dataset, "train_set", "train")
        tasks = splits.get(train_key, [])
        if not tasks:
            logger.error("No tasks found for split key: %s", train_key)
            return

        task_d = tasks[1]
        logger.info("Running one-task HMT test | exp_type=%s", exp_type)
        logger.info("Selected task: %s", task_d)

        env = ThorConnector(cfg=cfg, x_display=cfg.alfred.x_display)
        llm_agent = AlfredLlmAgent(cfg)
        planner = AlfredReactreeWithHMT(cfg, llm_agent, env, use_hmt=True)

        result = planner.collect_llm(task_d, args_dict)
        logger.info("HMT result: %s", result)


if hydra is not None:
    @hydra.main(version_base=None, config_path="../conf", config_name="alfred_reactree")
    def main(cfg):
        try:
            _main_impl(cfg)
        except Exception:
            logger.exception("HMT run failed")
            raise
else:
    def main(cfg=None):
        logging.basicConfig(level=logging.INFO)
        logger.error("Hydra is not installed. Install with `pip install hydra-core` to run hmt main entrypoint.")


if __name__ == "__main__":
    main()