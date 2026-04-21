"""
Outer MCTS layer — searches over task decomposition candidates.

OuterMCTSPlanner drives the two-level search:
  1. Asks the LLM for Expand candidates (control_flow + subgoals).
  2. Simulates each candidate with inner MCTS to estimate reward.
  3. Selects the best candidate via UCT and commits it to the real environment.
"""

import copy
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import alfred.utils as alfred_utils

from .constants import (
    ALFRED_TERMINAL_ACTIONS,
    ALFRED_PRIMITIVE_ACTION_PATTERNS,
    Node,
    NodeType,
    _looks_like_compound_goal,
)
from .inner_mcts import ActionMCTSWrapper
from .types import DecompositionAction, DecompositionState

logger = logging.getLogger(__name__)


class OuterMCTSPlanner:
    """
    Outer MCTS over decomposition candidates.

    For each goal state the planner:
      1. Lazily generates LLM Expand candidates.
      2. Runs outer_budget MCTS iterations (select → expand → simulate → backup).
      3. Commits the best-scoring decomposition to the real environment.
      4. Recurses into each subgoal via execute_goal_with_mcts.
    """

    def __init__(
        self,
        cfg,
        llm_agent,
        env,
        outer_budget: int = 2,
        inner_budget: int = 5,
        decomp_candidate_count: int = 3,
    ):
        self.cfg = cfg
        self.llm_agent = llm_agent
        self.env = env
        self.outer_budget = outer_budget
        self.inner_budget = inner_budget
        self.decomp_candidate_count = decomp_candidate_count
        self.logger = logger
        self.action_mcts = ActionMCTSWrapper(cfg=cfg, llm_agent=llm_agent, env=env, budget=inner_budget)

        # Set by AlfredReactreeWithHMT.collect_llm_with_hmt before planning starts.
        self._current_traj_data: Optional[Dict] = None
        self._cur_step_id: int = 1
        self._cur_decision_id: int = 1
        # Scene restore info for simulation — set by collect_llm_with_hmt so that
        # outer_default_policy can reset the environment to a clean initial state
        # before each outer MCTS simulation, making reward signals comparable.
        self._sim_restore_info: Optional[Dict] = None

    # -----------------------------------------------------------------------
    # Node role helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _node_role(node: NodeType) -> str:
        return getattr(node, "node_role", "goal")

    def _get_decomposition_children(self, node: NodeType) -> List[NodeType]:
        return [c for c in node.get_children() if self._node_role(c) == "decomposition"]

    def _get_decomposition_paths(self, node: NodeType) -> List[List[str]]:
        all_paths: List[List[str]] = []

        def dfs(cur: NodeType) -> None:
            children = self._get_decomposition_children(cur)
            if not children:
                all_paths.append(list(cur.get_state().action_history))
                return
            for child in children:
                dfs(child)

        dfs(node)
        return all_paths

    # -----------------------------------------------------------------------
    # Node factory
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Small utilities
    # -----------------------------------------------------------------------

    def _record_terminal_result(self, node: NodeType, reward: float, success: bool) -> None:
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
        normalized = self._normalize_goal_text(goal)
        if not normalized:
            return False
        if normalized in ALFRED_TERMINAL_ACTIONS:
            return True
        if _looks_like_compound_goal(normalized):
            return False
        return any(p.match(normalized) for p in ALFRED_PRIMITIVE_ACTION_PATTERNS)

    def _split_goal_text(self, goal: str) -> List[str]:
        cleaned = (goal or "").strip()
        if not cleaned:
            return []
        parts = [
            p.strip()
            for p in re.split(
                r"\b(?:and then|then|and|after that|afterwards)\b|,",
                cleaned,
                flags=re.IGNORECASE,
            )
            if p.strip()
        ]
        return parts or [cleaned]

    def _has_decomposition_progress(
        self,
        goal: str,
        subgoals: List[str],
        ancestor_goals: Optional[List[str]] = None,
    ) -> bool:
        norm_goal = self._normalize_goal_text(goal)
        norm_subs = [self._normalize_goal_text(s) for s in subgoals if s.strip()]
        if not norm_subs:
            return False
        # Reject if every subgoal is identical to the parent — no progress.
        if all(s == norm_goal for s in norm_subs):
            return False
        # Reject if ANY subgoal equals the parent — that branch will recurse forever.
        if any(s == norm_goal for s in norm_subs):
            return False
        # Reject if any subgoal already appears in the ancestor chain — cycle.
        if ancestor_goals:
            norm_ancestors = {self._normalize_goal_text(a) for a in ancestor_goals}
            if any(s in norm_ancestors for s in norm_subs):
                return False
        return True

    def _is_state_terminal(self, state: DecompositionState) -> bool:
        return state.is_terminal or state.depth >= self._max_depth()

    @staticmethod
    def _candidate_key(action: DecompositionAction) -> Tuple[str, Tuple[str, ...]]:
        return (action.control_flow, tuple(action.subgoals))

    @staticmethod
    def _candidate_subgoal_key(action: DecompositionAction) -> Tuple[str, ...]:
        return tuple(action.subgoals)

    @staticmethod
    def _safe_obs_text(observation: str) -> str:
        return observation if observation else "No observation available."

    def _is_fully_expanded(self, node: NodeType) -> bool:
        state = node.get_state()
        if not state.candidate_generation_done:
            return False
        return len(self._get_decomposition_children(node)) >= len(state.get_available_actions())

    # -----------------------------------------------------------------------
    # LLM candidate generation
    # -----------------------------------------------------------------------

    def _llm_generate_expand_candidate(
        self,
        goal: str,
        obs_text: str,
        task_type: str,
        depth: int,
    ) -> Optional[DecompositionAction]:
        """Ask the LLM for one Expand decomposition. Returns None if the LLM
        chose Act/Error or failed to produce decomposition progress."""
        nl_inst_info = {"nl_inst": goal, "message": None, "task_type": task_type, "depth": depth}
        max_try = 6
        try:
            self.llm_agent.reset(nl_inst_info, obs_text)
            for attempt in range(max_try):
                next_step_info = self.llm_agent.plan_expand_only()
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
                    continue
                else:
                    self.logger.info(
                        "LLM chose '%s' instead of Expand for goal='%s'; no decomposition",
                        next_step_class, goal,
                    )
                    break
        except Exception as exc:
            self.logger.warning("_llm_generate_expand_candidate failed: %s", exc)
        return None

    def _generate_next_candidate(
        self, state: DecompositionState
    ) -> Optional[DecompositionAction]:
        """Lazily generate one new unique LLM decomposition candidate for *state*."""
        if state.candidate_generation_done:
            return None

        obs = state.env_snapshot.get("observation", "") if state.env_snapshot else ""
        task_type = state.env_snapshot.get("task_type", "unknown") if state.env_snapshot else "unknown"
        seen_subgoal_keys = {self._candidate_subgoal_key(c) for c in state.generated_candidates}

        candidate: Optional[DecompositionAction] = None
        for _ in range(4):
            sampled = self._llm_generate_expand_candidate(state.goal, obs, task_type, state.depth)
            if sampled is None:
                continue
            if self._candidate_subgoal_key(sampled) in seen_subgoal_keys:
                continue
            candidate = sampled
            break

        if candidate is None:
            state.candidate_generation_done = True
            self.logger.info(
                "No unique LLM decomposition for goal='%s'. Marking fully generated.", state.goal
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

    # def get_decomposition_candidates(
    #     self, goal: str, observation: str
    # ) -> List[DecompositionAction]:
    #     """Generate up to decomp_candidate_count unique LLM decompositions."""
    #     cleaned = goal.strip() if goal else "complete the task"
    #     max_candidates = max(1, int(self.decomp_candidate_count))
    #     final: List[DecompositionAction] = []
    #     seen: set = set()

    #     for _ in range(max_candidates * 4):
    #         if len(final) >= max_candidates:
    #             break
    #         candidate = self._llm_generate_expand_candidate(cleaned, observation, "unknown", 0)
    #         if candidate is None:
    #             continue
    #         key = self._candidate_subgoal_key(candidate)
    #         if key in seen:
    #             continue
    #         seen.add(key)
    #         final.append(candidate)

    #     self.logger.info("Generated %d LLM candidates for goal: %s", len(final), cleaned)
    #     return final

    # -----------------------------------------------------------------------
    # Reactree execution loop (used at execution time for primitive goals)
    # -----------------------------------------------------------------------

    def _execute_subgoal_with_reactree(
        self,
        goal: str,
        env_snapshot: Dict,
        depth: int,
    ) -> Tuple[bool, float, int, List[str]]:
        """Run the full ReAcTree Think/Act/Expand loop for a single goal.

        Falls back to inner MCTS directly if _current_traj_data is not set.
        """
        if self._current_traj_data is None:
            self.logger.warning("_current_traj_data not set, using inner MCTS directly")
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(
                goal, env_snapshot.get("observation", "")
            )
            return success, reward, steps, trajectory

        traj_data = self._current_traj_data
        task_type = env_snapshot.get("task_type", "unknown")
        nl_inst_info = {"nl_inst": goal, "message": None, "task_type": task_type, "depth": depth}

        try:
            obs = self.env.init_reset(traj_data)
            obs_text = obs["text"]
            self.llm_agent.reset(nl_inst_info, obs_text)
            skill_set = [s for s in self.llm_agent.update_skill_set(obs)
                         if not s.startswith("recall location of ")]
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
            "ReactTree START | goal='%s' | depth=%d | obs='%.80s'", goal, depth, obs_text
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

            self.logger.info("ReactTree step | goal='%s' | %s: %s", goal, next_step_class, next_step)

            if next_step_class == "Think":
                self._cur_decision_id += 1

            elif next_step_class == "Act":
                self._cur_decision_id += 1
                act_goal = str(next_step).strip()
                if act_goal == "done":
                    return True, 1.0, steps_taken, trajectory
                elif act_goal == "failure":
                    return False, -0.5, steps_taken, trajectory
                elif act_goal.startswith("recall location of "):
                    # Memory lookup — not a real env action; query working_memory and
                    # update observation so the LLM has location context on next step.
                    from alfred.utils import recall_working_memory
                    target_obj = act_goal.split("recall location of ", 1)[1]
                    current_obs_text = recall_working_memory(self.env.working_memory, target_obj)
                    self.llm_agent.add_obs(current_obs_text)
                    self.logger.info("Recall working memory | target='%s' | obs='%s'", target_obj, current_obs_text)
                    continue
                else:
                    act_success, act_reward, act_steps, act_trajectory = self.inner_mcts_solve_subgoal(
                        act_goal, current_obs_text
                    )
                    trajectory.extend(act_trajectory if act_trajectory else [act_goal])
                    steps_taken += act_steps
                    self._cur_step_id += act_steps
                    try:
                        scene_name = self.env.last_event.metadata.get("sceneName", "FloorPlan1")
                        obs_pack = self.env.llm_skill_interact(None, scene_name)
                        current_obs_text = obs_pack.get("message", current_obs_text)
                        self.llm_agent.add_obs(current_obs_text)
                        skill_set = [s for s in self.llm_agent.update_skill_set(obs_pack)
                                     if not s.startswith("recall location of ")]
                    except Exception as exc:
                        self.logger.warning("Failed to refresh obs after inner MCTS Act: %s", exc)
                    if not act_success:
                        return False, act_reward, steps_taken, trajectory

            elif next_step_class == "Expand":
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
                llm_action = DecompositionAction(control_flow, subgoals, prior_prob=0.80)
                nested_node.get_state().generated_candidates = [llm_action]
                result = self.execute_goal_with_mcts(nested_node)
                return result["success"], result["reward"], result["steps"], result.get("trajectory", [])

            elif next_step_class == "Error":
                self._cur_decision_id += 1
                try_count += 1
                if try_count >= max_try:
                    self.logger.warning("ReactTree max errors reached | goal='%s'", goal)
                    return False, -0.5, steps_taken, trajectory

            else:
                self.logger.warning(
                    "ReactTree unknown step class '%s' | goal='%s'", next_step_class, goal
                )
                break

        return False, -0.3, steps_taken, trajectory

    # -----------------------------------------------------------------------
    # Simulation helpers
    # -----------------------------------------------------------------------

    def _simulate_goal(
        self,
        goal: str,
        env_snapshot: Dict,
        depth: int,
        action_history: List[str],
    ) -> Dict[str, Any]:
        """Estimate the value of a goal via inner MCTS or recursive decomposition."""
        normalized = self._normalize_goal_text(goal)
        if not normalized:
            return {"success": False, "reward": -1.0, "steps": 1, "mode": "empty_goal",
                    "goal": goal, "trajectory": []}

        if depth >= self._max_depth() or self._is_primitive_goal(goal):
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(
                goal, env_snapshot.get("observation", "")
            )
            return {"success": success, "reward": reward, "steps": steps,
                    "mode": "primitive", "goal": goal, "trajectory": trajectory}

        subgoal_root = self._make_goal_node(
            goal=goal, env_snapshot=env_snapshot, depth=depth,
            action_history=action_history, executed_steps=[], node_role="goal",
        )
        best_subgoal_node, _ = self.outer_monte_carlo_tree_search(subgoal_root)
        best_action = getattr(best_subgoal_node, "decomposition_action", None)

        if best_action is None:
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(
                goal, env_snapshot.get("observation", "")
            )
            return {"success": success, "reward": reward, "steps": steps,
                    "mode": "primitive_fallback", "goal": goal, "trajectory": trajectory}

        if not self._has_decomposition_progress(goal, best_action.subgoals, ancestor_goals=action_history):
            success, reward, steps, trajectory = self.inner_mcts_solve_subgoal(
                goal, env_snapshot.get("observation", "")
            )
            return {"success": success, "reward": reward, "steps": steps,
                    "mode": "no_progress_fallback", "goal": goal, "trajectory": trajectory}

        nested = self._evaluate_decomposition_action(
            best_action,
            env_snapshot=env_snapshot,
            depth=depth + 1,
            action_history=action_history + [best_action.signature()],
        )
        nested["mode"] = "decomposed"
        nested["goal"] = goal
        nested["trajectory"] = list(nested.get("simulated_actions", []))
        nested["chosen_action"] = {
            "control_flow": best_action.control_flow,
            "subgoals": list(best_action.subgoals),
            "prior_prob": best_action.prior_prob,
        }
        return nested

    def _evaluate_decomposition_action(
        self,
        action: DecompositionAction,
        env_snapshot: Dict,
        depth: int,
        action_history: List[str],
    ) -> Dict[str, Any]:
        """Simulate all subgoals of a decomposition and return aggregate metrics."""
        total_reward = 0.0
        total_steps = 0
        success_count = 0
        fail_count = 0
        subgoal_results: List[Dict[str, Any]] = []
        simulated_actions: List[str] = []

        for subgoal in action.subgoals:
            result = self._simulate_goal(subgoal, env_snapshot=env_snapshot,
                                         depth=depth, action_history=action_history)
            subgoal_results.append(result)
            total_reward += float(result["reward"])
            total_steps += int(result["steps"])
            simulated_actions.extend(result.get("trajectory", []))
            for nested in result.get("subgoal_results", []):
                simulated_actions.extend(nested.get("trajectory", []))

            if result["success"]:
                success_count += 1
            else:
                fail_count += 1

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

    # -----------------------------------------------------------------------
    # Inner MCTS proxy (thin wrapper so callers don't reach into action_mcts)
    # -----------------------------------------------------------------------

    def inner_mcts_solve_subgoal(
        self, subgoal: str, current_obs: str
    ) -> Tuple[bool, float, int, List[str]]:
        success, reward, action_sequence = self.action_mcts.solve_subgoal(
            subgoal, current_obs, restore_info=self._sim_restore_info
        )
        steps_used = len(action_sequence) if action_sequence else 1
        self.logger.info(
            "Inner MCTS solve | subgoal='%s' | success=%s | reward=%.3f | steps=%d",
            subgoal, success, reward, steps_used,
        )
        return success, reward, steps_used, action_sequence

    # -----------------------------------------------------------------------
    # Outer MCTS: tree policy (selection + expansion)
    # -----------------------------------------------------------------------

    def outer_tree_policy(self, node: NodeType) -> NodeType:
        while not self._is_state_terminal(node.get_state()):
            role = self._node_role(node)
            state = node.get_state()
            if role == "decomposition":
                return node
            if self._is_fully_expanded(node):
                next_node = self.outer_best_child(node, is_exploration=True)
                if next_node is None or next_node is node:
                    return node
                node = next_node
            else:
                return self.outer_expand(node)
        return node

    def outer_expand(self, node: NodeType) -> NodeType:
        """Materialize one new decomposition candidate as a child node."""
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
            "Expanded | parent_depth=%d | child_depth=%d | children=%d | control_flow=%s | subgoals=%s",
            state.depth, child_state.depth, len(node.get_children()),
            selected.control_flow, selected.subgoals,
        )
        return child_node

    def outer_expand_action(
        self, state: DecompositionState, tried_paths: List[List[str]]
    ) -> Optional[DecompositionAction]:
        prefix = list(state.action_history)
        for candidate in state.get_available_actions():
            path = prefix + [candidate.signature()]
            if path not in [p[: len(path)] for p in tried_paths]:
                return candidate
        while not state.candidate_generation_done:
            candidate = self._generate_next_candidate(state)
            if candidate is None:
                break
            path = prefix + [candidate.signature()]
            if path not in [p[: len(path)] for p in tried_paths]:
                return candidate
        return None

    # -----------------------------------------------------------------------
    # Outer MCTS: simulation (default policy)
    # -----------------------------------------------------------------------

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
        """Simulation: evaluate a decomposition candidate via inner MCTS rollouts."""
        action = getattr(node, "decomposition_action", None)
        if action is None:
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
        success_count = evaluation["success_count"]
        fail_count = evaluation["fail_count"]

        state = node.get_state()
        state.succeed_subgoals += success_count
        state.failed_subgoals += fail_count
        state.is_terminal = True

        self.logger.info(
            "Outer default policy | control_flow=%s | success=%s | successes=%d | failures=%d | reward=%.3f",
            action.control_flow, evaluation["success"], success_count, fail_count, total_reward,
        )

        state.mcts_attempts.append({
            "decomposition": action.signature(),
            "reward": total_reward,
            "simulated_actions": evaluation.get("simulated_actions", []),
        })
        return total_reward, node

    # -----------------------------------------------------------------------
    # Outer MCTS: UCT child selection + backup
    # -----------------------------------------------------------------------

    def outer_best_child(self, node: NodeType, is_exploration: bool) -> NodeType:
        """UCT child selection with LLM prior weighting."""
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

        return best_node

    def outer_backup(self, node: NodeType, reward: float) -> None:
        """Propagate reward up the tree with discount."""
        gamma = float(getattr(getattr(self.cfg, "hmt", object()), "gamma", 0.95))
        discount = 1.0
        cur = node
        while cur is not None:
            cur.increment_visit_count()
            cur.update_quality_value(reward * discount)
            discount *= gamma
            cur = cur.get_parent()

    # -----------------------------------------------------------------------
    # Outer MCTS: full search loop
    # -----------------------------------------------------------------------

    def outer_monte_carlo_tree_search(
        self, node: NodeType
    ) -> Tuple[NodeType, List[NodeType]]:
        root_state = node.get_state()
        self.logger.info(
            "=== outer MCTS START | goal='%s' | depth=%d | budget=%d ===",
            root_state.goal, root_state.depth, self.outer_budget,
        )
        all_expand_nodes: List[NodeType] = []
        for i in range(self.outer_budget):
            self.logger.info(
                "--- Outer MCTS iteration %d/%d | goal='%s' | decomp_children=%d ---",
                i + 1, self.outer_budget, root_state.goal,
                len(self._get_decomposition_children(node)),
            )
            expand_node = self.outer_tree_policy(node)
            all_expand_nodes.append(expand_node)
            reward, leaf_node = self.outer_default_policy(expand_node)
            self.outer_backup(leaf_node, reward)

        best_node = self.outer_best_child(node, is_exploration=False)
        best_action = getattr(best_node, "decomposition_action", None)
        self.logger.info(
            "=== outer MCTS END | chosen: %s | subgoals=%s | q=%.3f | visits=%d ===",
            best_action.control_flow if best_action else "(none)",
            best_action.subgoals if best_action else [],
            best_node.quality_value,
            best_node.visit_count,
        )
        return best_node, all_expand_nodes

    # -----------------------------------------------------------------------
    # Execution: commit chosen decomposition to the real environment
    # -----------------------------------------------------------------------

    def execute_goal_with_mcts(self, goal_node: NodeType) -> Dict[str, Any]:
        """Top-level dispatcher: run MCTS then commit, or execute primitively."""
        state = goal_node.get_state()
        goal = state.goal
        role = self._node_role(goal_node)
        self.logger.info(
            ">> execute_goal_with_mcts | role=%s | goal='%s' | depth=%d", role, goal, state.depth
        )

        if not goal:
            self._record_terminal_result(goal_node, -1.0, False)
            return {"success": False, "reward": -1.0, "steps": 1, "mode": "empty_goal",
                    "goal": goal, "chosen_action": None, "subgoal_results": [], "trajectory": []}

        if state.depth >= self._max_depth() or self._is_primitive_goal(goal):
            reason = "max_depth" if state.depth >= self._max_depth() else "primitive"
            success, reward, steps, trajectory = self._execute_subgoal_with_reactree(
                goal, state.env_snapshot, state.depth
            )
            self._record_terminal_result(goal_node, reward, success)
            goal_node.get_state().trajectory = trajectory
            self.logger.info(
                "  -> %s | goal='%s' | success=%s | reward=%.3f | steps=%d",
                reason, goal, success, reward, steps,
            )
            return {"success": success, "reward": reward, "steps": steps, "mode": reason,
                    "goal": goal, "chosen_action": None, "subgoal_results": [], "trajectory": trajectory}

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
            return {"success": success, "reward": reward, "steps": steps,
                    "mode": "primitive_fallback", "goal": goal, "chosen_action": None,
                    "subgoal_results": [], "trajectory": trajectory}

        self.logger.info(
            "  -> committing decomposition | control_flow=%s | %d subgoals: %s",
            chosen_action.control_flow, len(chosen_action.subgoals), chosen_action.subgoals,
        )
        execution_result = self.execute_committed_decomposition(best_node)
        state.trajectory = list(execution_result.get("trajectory", []))
        state.final_success_trajectory = copy.deepcopy(execution_result.get("final_success_trajectory", []))
        state.succeed_subgoals += execution_result["success_count"]
        state.failed_subgoals += execution_result["fail_count"]
        state.is_terminal = execution_result["success"]
        execution_result.update({
            "mode": "mcts",
            "goal": goal,
            "chosen_action": self._format_action_metadata(chosen_action),
        })
        return execution_result

    def execute_committed_decomposition(self, decomp_node: NodeType) -> Dict[str, Any]:
        """Execute the chosen decomposition in the real environment subgoal by subgoal."""
        action = getattr(decomp_node, "decomposition_action", None)
        if action is None:
            return {"success": False, "reward": -1.0, "subgoal_results": [],
                    "steps_executed": 0, "success_count": 0, "fail_count": 0,
                    "reason": "no_decomposition_action"}

        total_reward = 0.0
        total_steps = 0
        success_count = 0
        fail_count = 0
        subgoal_results: List[Dict[str, Any]] = []
        final_success_trajectory: List[Dict[str, Any]] = []
        flat_trajectory: List[str] = []
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
            flat_trajectory.extend(primitive_actions)
            subgoal_results.append({"subgoal_index": index, "subgoal": subgoal, **step_result})
            final_success_trajectory.append({
                "subgoal_index": index,
                "subgoal": subgoal,
                "success": bool(step_result.get("success", False)),
                "mode": step_result.get("mode", "unknown"),
                "primitive_actions": primitive_actions,
            })
            total_reward += float(step_result["reward"])
            total_steps += int(step_result["steps"])

            if step_result["success"]:
                success_count += 1
            else:
                fail_count += 1

            self.logger.info(
                "Subgoal [%d/%d] | parent='%s' | subgoal='%s' | success=%s | reward=%.3f",
                index, len(action.subgoals), state.goal, subgoal,
                step_result["success"], step_result["reward"],
            )

            if action.control_flow == "sequence" and not step_result["success"]:
                self.logger.info(
                    "Sequence subgoal failed at index=%d, continuing for traceability", index
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
        state.trajectory = list(flat_trajectory)
        state.final_success_trajectory = copy.deepcopy(final_success_trajectory)

        return {
            "success": overall_success,
            "subgoal_results": subgoal_results,
            "reward": total_reward,
            "steps": total_steps,
            "steps_executed": total_steps,
            "success_count": success_count,
            "fail_count": fail_count,
            "control_flow": action.control_flow,
            "trajectory": flat_trajectory,
            "final_success_trajectory": final_success_trajectory,
        }
