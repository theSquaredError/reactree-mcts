"""
Inner MCTS layer — primitive action search for a single subgoal.

Classes
-------
_MCTSChatAdapter   : wraps AlfredLlmAgent to give MCTSAlgorithm a chat() interface
ActionMCTSWrapper  : full SELECT→EXPAND→SIMULATE→BACKUP MCTS for one primitive subgoal

The key difference from mcts/algorithm.py is that AI2-THOR cannot clone environments.
state.clone() is replaced by _restore_and_replay(restore_info, action_history) which
resets the env to the subgoal-start state and replays committed actions to reach any
node in the tree. Everything else — UCT selection, expansion, simulation, backup —
mirrors the original algorithm.
"""

import logging
import math
import types as _types
from typing import Any, Dict, List, Optional, Tuple

from .constants import ALFRED_PRIMITIVE_ACTION_PREFIXES, ALFRED_TERMINAL_ACTIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM adapter: gives MCTSAlgorithm the chat() interface it expects
# ---------------------------------------------------------------------------

class _MCTSChatAdapter:
    """Wraps AlfredLlmAgent to expose the chat() interface MCTSAlgorithm expects."""

    def __init__(self, llm_agent) -> None:
        self._agent = llm_agent
        self._base = llm_agent.llm

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Generate an action using free-text LLM generation then fuzzy-match to
        available_commands.  Mirrors mcts/algorithm.py llm_inference() but
        replaces guidance.select(100+ items) — which always returns 'done' on
        small 8B models — with guidance.gen() + best-overlap matching.

        Returns {matched_command: 1.0} or {} (llm_inference will retry / random).
        """
        import ast
        import re as _re

        # ---- 1. Extract available_commands from the last user message ----
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

        # ---- 2. Build prompt string from messages ----
        prompt_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(content)
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant: ")
        prompt_str = "\n".join(prompt_parts)

        # ---- 3. Free-text generation (reliable on 8B models, unlike select(100+)) ----
        try:
            import guidance
            lm = self._base + prompt_str
            lm += guidance.gen(stop="\n", name="action_text", max_tokens=60, temperature=0)
            generated: str = (lm["action_text"] or "").strip()
        except Exception as exc:
            logger.warning("_MCTSChatAdapter.chat gen failed: %s", exc)
            return {}

        if not generated:
            return {}

        # ---- 4. Exact match first ----
        gen_lower = generated.lower()
        for cmd in available_commands:
            if cmd.lower() == gen_lower:
                logger.debug("MCTS chat exact match: '%s'", cmd)
                return {cmd: 1.0}

        # ---- 5. Fuzzy word-overlap match (same idea as mcts/algorithm.py filter) ----
        gen_words = set(gen_lower.split())
        best_cmd: Optional[str] = None
        best_overlap = 0
        for cmd in available_commands:
            cmd_words = set(cmd.lower().split())
            overlap = len(gen_words & cmd_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cmd = cmd

        # Require at least one meaningful word overlap
        if best_cmd and best_overlap >= 1:
            logger.debug(
                "MCTS chat fuzzy match: generated='%s' → matched='%s' (overlap=%d)",
                generated, best_cmd, best_overlap,
            )
            return {best_cmd: 1.0}

        logger.debug("MCTS chat no match for generated='%s'", generated)
        return {}


# ---------------------------------------------------------------------------
# Inner MCTS wrapper
# ---------------------------------------------------------------------------

class ActionMCTSWrapper:
    """
    Full SELECT→EXPAND→SIMULATE→BACKUP MCTS for primitive action search.

    Mirrors mcts/algorithm.py but replaces state.clone()/env.clone() with
    _restore_and_replay(), which resets the AI2-THOR env to the subgoal-start
    state and replays the node's action_history to reach its position in the tree.

    Falls back to a direct LLM greedy rollout when restore_info is not provided
    (e.g. called without outer-planner context) or MCTSAlgorithm is unavailable.
    """

    def __init__(self, cfg, llm_agent, env, budget: int = 10):
        self.cfg = cfg
        self.llm_agent = llm_agent
        self.env = env
        self.budget = budget
        self.mcts_unavailable_reason = ""
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

            # Patch prompt_template to handle empty action_history safely.
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
                    msgs.append({
                        "role": "user",
                        "content": f"Init environment: {state.obs}\n The candidate actions are {available_commands}",
                    })
                else:
                    obs_list = list(getattr(state, "obs_list", [state.obs]))
                    action_history = list(getattr(state, "action_history", []))
                    for i, obs in enumerate(obs_list):
                        prefix = "Init environment: " if i == 0 else "Current observation: "
                        is_last = (i == len(obs_list) - 1)
                        if is_last:
                            msgs.append({
                                "role": "user",
                                "content": f"{prefix}{obs}\nThe candidate actions are {available_commands}",
                            })
                        else:
                            msgs.append({"role": "user", "content": f"{prefix}{obs}"})
                            if i < len(action_history):
                                msgs.append({"role": "assistant", "content": action_history[i]})
                return _SYS, msgs

            self.mcts.prompt_template = _types.MethodType(_prompt_template_fixed, self.mcts)

        except Exception as exc:
            self.mcts_unavailable_reason = str(exc)
            logger.warning(
                "ActionMCTSWrapper: MCTSAlgorithm unavailable, will use greedy fallback: %s", exc
            )

        self._can_use_mcts_backend = self.mcts is not None

    # -----------------------------------------------------------------------
    # _InnerNode — mirrors mcts/node.py Node for the primitive-action tree
    # -----------------------------------------------------------------------

    class _InnerNode:
        """
        One node in the inner MCTS tree.

        Each node represents an env state reached by executing action_history
        from the subgoal-start state. Mirrors the original Node class fields:
        visit_count, quality_value, children, parent — plus action_prob for
        the discounted UCT formula from algorithm.py backup().
        """

        def __init__(
            self,
            action_history: List[str],
            obs_list: List[str],
            available_commands: List[str],
            action_prob: float = 1.0,
            parent: Optional["ActionMCTSWrapper._InnerNode"] = None,
        ) -> None:
            self.action_history = list(action_history)
            self.obs_list = list(obs_list)
            self.available_commands = list(available_commands)
            self.action_prob = action_prob          # prob of action that led here
            self.parent = parent
            self.children: List["ActionMCTSWrapper._InnerNode"] = []
            self.tried_actions: List[str] = []      # actions already expanded from here
            self.visit_count: int = 0
            self.quality_value: float = 0.0

        @property
        def obs(self) -> str:
            return self.obs_list[-1] if self.obs_list else ""

        def is_fully_expanded(self) -> bool:
            return all(a in self.tried_actions for a in self.available_commands)

    # -----------------------------------------------------------------------
    # _RolloutState — thin wrapper so mcts.llm_inference() can read obs/history
    # -----------------------------------------------------------------------

    class _RolloutState:
        def __init__(self, obs: str, obs_list: List[str], action_history: List[str]):
            self.obs = obs
            self.obs_list = list(obs_list)
            self.action_history = list(action_history)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _current_skill_set(self) -> List[str]:
        try:
            scene_name = self.env.last_event.metadata.get("sceneName", "FloorPlan1")
            obs_pack = self.env.llm_skill_interact(None, scene_name)
            skill_set = self.llm_agent.update_skill_set(obs_pack)
            deduped: List[str] = []
            seen: set = set()
            for cmd in skill_set:
                if cmd == "look":
                    continue
                if cmd.startswith("recall location of "):
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

    # -----------------------------------------------------------------------
    # _restore_and_replay — the AI2-THOR equivalent of state.clone()
    #
    # Original algorithm.py line 52:  current_state = node.get_state().clone()
    # That forks the env so simulation doesn't touch the real state.
    # Here we reset the same env to the subgoal-start state, then replay
    # action_history to reach the target node's position in the tree.
    # -----------------------------------------------------------------------

    def _restore_and_replay(
        self, restore_info: Dict, action_history: List[str]
    ) -> Tuple[str, List[str]]:
        """
        Reset env to subgoal-start state and replay action_history.

        Returns (current_obs, available_commands) at the resulting state.
        This is called at the start of every EXPAND and every SIMULATE so each
        branch of the tree gets an independent env state — the same guarantee
        that clone() provides in the TextWorld version.
        """
        self.env.restore_scene(
            restore_info["object_poses"],
            restore_info["object_toggles"],
            restore_info["dirty_and_empty"],
        )
        self.env.step(restore_info["init_action"])
        self.env.set_task(restore_info["traj_data"], restore_info["model_args"], reward_type="dense")
        cfg_agent = getattr(self.cfg, "llm_agent", None)
        if cfg_agent and getattr(cfg_agent, "working_memory", False):
            self.env.reset_working_memory()

        cur_obs = restore_info.get("init_obs", "")
        for action in action_history:
            try:
                obs_ret = self.env.llm_skill_interact(action)
                cur_obs = obs_ret.get("message", cur_obs)
            except Exception as exc:
                logger.warning("_restore_and_replay replay failed at '%s': %s", action, exc)
                break

        available = self._current_skill_set()
        return cur_obs, available

    # -----------------------------------------------------------------------
    # MCTS primitives — mirrors algorithm.py methods
    # -----------------------------------------------------------------------

    def _inner_best_child(
        self, node: "_InnerNode", explore: bool
    ) -> Optional["_InnerNode"]:
        """
        UCT child selection — mirrors algorithm.py best_child().

        score = Q/N  +  C * sqrt(2*log(parent.N) / child.N) * action_prob
        C = 1/sqrt(2) during exploration, 0 during final exploitation.
        """
        if not node.children:
            return None
        best_score = -float("inf")
        best: Optional[ActionMCTSWrapper._InnerNode] = None
        C = 1.0 / math.sqrt(2.0) if explore else 0.0
        for child in node.children:
            if child.visit_count == 0:
                return child
            exploit = child.quality_value / child.visit_count
            explore_bonus = C * math.sqrt(
                2.0 * math.log(max(1, node.visit_count)) / child.visit_count
            )
            score = exploit + explore_bonus * child.action_prob
            if score > best_score:
                best_score = score
                best = child
        return best

    def _inner_expand(
        self, node: "_InnerNode", restore_info: Dict
    ) -> Optional["_InnerNode"]:
        """
        EXPAND — mirrors algorithm.py expand() + expand_action().

        1. Restore env to node's state via _restore_and_replay (= state.clone()).
        2. Ask LLM to pick an untried action from available_commands.
        3. Execute that action, observe result, build child node.
        """
        cur_obs, available = self._restore_and_replay(restore_info, node.action_history)

        untried = [a for a in available if a not in node.tried_actions]
        if not untried:
            return None

        state = self._RolloutState(
            obs=cur_obs, obs_list=node.obs_list, action_history=node.action_history
        )
        action, action_prob = self.mcts.llm_inference(state, untried)
        node.tried_actions.append(action)

        if action in ALFRED_TERMINAL_ACTIONS:
            # Terminal action chosen during expand — build a leaf node with no env step.
            child = self._InnerNode(
                action_history=node.action_history + [action],
                obs_list=node.obs_list,
                available_commands=[],
                action_prob=float(action_prob),
                parent=node,
            )
            node.children.append(child)
            return child

        try:
            obs_ret = self.env.llm_skill_interact(action)
            new_obs = obs_ret.get("message", cur_obs)
        except Exception as exc:
            logger.warning("_inner_expand: env step failed action='%s': %s", action, exc)
            return None

        child_available = self._current_skill_set()
        child = self._InnerNode(
            action_history=node.action_history + [action],
            obs_list=node.obs_list + [new_obs],
            available_commands=child_available,
            action_prob=float(action_prob),
            parent=node,
        )
        node.children.append(child)
        logger.debug(
            "EXPAND | depth=%d | action='%s' | prob=%.3f | untried_left=%d",
            len(child.action_history), action, action_prob, len(untried) - 1,
        )
        return child

    def _inner_tree_policy(
        self, node: "_InnerNode", restore_info: Dict
    ) -> "_InnerNode":
        """
        SELECT or EXPAND — mirrors algorithm.py tree_policy().

        Traverse using UCT until we reach a node that still has untried actions,
        then expand one of them. Returns the new leaf (or a terminal node).
        """
        while True:
            if not node.is_fully_expanded():
                child = self._inner_expand(node, restore_info)
                return child if child is not None else node
            if node.children:
                node = self._inner_best_child(node, explore=True)  # type: ignore[assignment]
            else:
                return node

    def _inner_default_policy(
        self, node: "_InnerNode", subgoal: str, restore_info: Dict
    ) -> float:
        """
        SIMULATE — mirrors algorithm.py default_policy().

        Restore env to node's state (= clone()), then run a greedy LLM rollout
        to terminal. Returns discounted reward: 1.0 on success, -1.0 on failure.
        The discount accumulates gamma * action_prob per step, same as original.
        """
        cur_obs, _ = self._restore_and_replay(restore_info, node.action_history)
        action_history = list(node.action_history)
        obs_list = list(node.obs_list)
        gamma = 0.95
        sim_discount = 1.0
        max_sim_steps = max(1, self.budget - len(action_history))

        failed_actions: List[str] = []
        for _ in range(max_sim_steps):
            available = [a for a in self._current_skill_set() if a not in failed_actions]
            if not available:
                return -1.0 * sim_discount

            state = self._RolloutState(
                obs=cur_obs, obs_list=obs_list, action_history=action_history
            )
            action, action_prob = self.mcts.llm_inference(state, available)
            sim_discount *= gamma * float(action_prob)

            if action == "done":
                return 1.0 * sim_discount
            if action == "failure":
                return -1.0

            try:
                obs_ret = self.env.llm_skill_interact(action)
                cur_obs = obs_ret.get("message", cur_obs)
                # If env rejected the action (action failed but didn't raise), exclude it
                if obs_ret.get("success") is False or "already" in cur_obs.lower() or "not close to you" in cur_obs.lower():
                    failed_actions.append(action)
            except Exception as exc:
                logger.warning("_inner_default_policy: env step failed: %s", exc)
                return -1.0

            action_history.append(action)
            obs_list.append(cur_obs)

            if self._is_subgoal_satisfied(subgoal, action, cur_obs):
                return 1.0 * sim_discount

        return -0.5 * sim_discount

    def _inner_backup(self, node: "_InnerNode", reward: float) -> None:
        """
        BACKPROPAGATE — mirrors algorithm.py backup().

        Walk to root, incrementing visit_count and accumulating discounted reward.
        discount *= action_prob * gamma at each level, same as original.
        """
        gamma = 0.95
        discount = 1.0
        while node is not None:
            node.visit_count += 1
            node.quality_value += reward * discount
            discount *= node.action_prob * gamma
            node = node.parent  # type: ignore[assignment]

    def _inner_monte_carlo_tree_search(
        self, root: "_InnerNode", subgoal: str, restore_info: Dict
    ) -> "_InnerNode":
        """
        Main MCTS loop — mirrors algorithm.py monte_carlo_tree_search().

        Runs `budget` iterations of SELECT→EXPAND→SIMULATE→BACKUP, then
        returns the best child of root via pure exploitation (C=0).
        """
        for i in range(self.budget):
            leaf = self._inner_tree_policy(root, restore_info)           # SELECT / EXPAND
            reward = self._inner_default_policy(leaf, subgoal, restore_info)  # SIMULATE
            self._inner_backup(leaf, reward)                             # BACKUP
            logger.debug(
                "Inner MCTS iter %d/%d | subgoal='%s' | leaf_depth=%d | reward=%.3f",
                i + 1, self.budget, subgoal, len(leaf.action_history), reward,
            )

        best = self._inner_best_child(root, explore=False)  # exploit only
        return best if best is not None else root

    # -----------------------------------------------------------------------
    # LLM fallback greedy rollout (no tree, no restore — used when
    # restore_info is unavailable or MCTSAlgorithm failed to import)
    # -----------------------------------------------------------------------

    def _solve_with_reactexpand_rollout(
        self, subgoal: str, obs: str
    ) -> Tuple[bool, float, List[str]]:
        """Greedy LLM loop with no MCTS tree."""
        logger.info(
            "Greedy fallback rollout | subgoal='%s' | obs='%s...'",
            subgoal, (obs or "")[:100],
        )

        task_type = "unknown"
        try:
            if getattr(self.env, "last_event", None) is not None:
                task_type = self.env.last_event.metadata.get("taskType", "unknown")
        except Exception:
            pass

        nl_inst_info = {"nl_inst": subgoal, "message": None, "task_type": task_type, "depth": 0}
        cur_obs = obs or "No observation available."
        action_history: List[str] = []
        action_prob_history: List[float] = []

        try:
            self.llm_agent.reset(nl_inst_info, cur_obs)
        except Exception as exc:
            logger.warning("Greedy fallback reset failed for '%s': %s", subgoal, exc)
            return False, -1.0, []

        max_steps = max(1, int(self.budget))
        think_budget = max_steps * 3
        think_count = 0

        for _ in range(max_steps):
            skill_set = self._current_skill_set()
            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class = next_step_info["next_step_class"]
                next_step = next_step_info["next_step"]
            except Exception as exc:
                logger.warning("Greedy fallback plan_next_step failed for '%s': %s", subgoal, exc)
                return False, -1.0, action_history

            if next_step_class == "Think":
                think_count += 1
                if think_count >= think_budget:
                    return False, -1.0, action_history
                continue

            if next_step_class == "Error":
                return False, -1.0, action_history

            if next_step_class == "Expand":
                raw_conditions = next_step.get("conditions", "")
                expanded = [s.strip() for s in raw_conditions.split(",") if s.strip()]
                if not expanded:
                    return False, -1.0, action_history
                next_step_class = "Act"
                next_step = expanded[0]

            if next_step_class != "Act":
                return False, -1.0, action_history

            action = str(next_step)
            action_history.append(action)
            action_prob_history.append(0.5)

            if action == "failure":
                return False, -1.0, action_history
            if action == "done":
                mean_prob = sum(action_prob_history) / max(1, len(action_prob_history))
                return True, mean_prob, action_history

            try:
                obs_ret = self.env.llm_skill_interact(action)
                cur_obs = obs_ret.get("message", "")
            except Exception as exc:
                logger.warning("Greedy fallback env step failed | action='%s': %s", action, exc)
                return False, -1.0, action_history

            self.llm_agent.add_obs(cur_obs)

            if self._is_subgoal_satisfied(subgoal, action, cur_obs):
                mean_prob = sum(action_prob_history) / max(1, len(action_prob_history))
                return True, mean_prob, action_history

        mean_prob = sum(action_prob_history) / max(1, len(action_prob_history)) if action_prob_history else 0.0
        return False, -mean_prob, action_history

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def solve_subgoal(
        self,
        subgoal: str,
        obs: str,
        restore_info: Optional[Dict] = None,
    ) -> Tuple[bool, float, List[str]]:
        """
        Solve one subgoal using inner MCTS.

        Parameters
        ----------
        subgoal      : natural-language subgoal text
        obs          : current env observation
        restore_info : scene restore dict from OuterMCTSPlanner._sim_restore_info.
                       When provided, enables full MCTS (SELECT→EXPAND→SIMULATE→BACKUP).
                       When None, falls back to greedy LLM rollout.

        Returns
        -------
        (success, reward, executed_action_sequence)
        """
        sg = (subgoal or "").strip()
        if not sg:
            return False, -1.0, []

        logger.info(
            "Inner MCTS ENTRY | subgoal='%s' | mcts=%s | restore=%s",
            sg, self._can_use_mcts_backend, restore_info is not None,
        )

        # Fall back to greedy if backend missing or no restore_info
        if not self._can_use_mcts_backend or restore_info is None:
            reason = "backend unavailable" if not self._can_use_mcts_backend else "no restore_info"
            logger.info("Inner MCTS using greedy fallback (%s) for '%s'", reason, sg)
            return self._solve_with_reactexpand_rollout(sg, obs)

        # Build root node at subgoal-start state
        available = self._current_skill_set()
        root = self._InnerNode(
            action_history=[],
            obs_list=[obs or "No observation available."],
            available_commands=available,
            action_prob=1.0,
            parent=None,
        )

        # Run full MCTS — explores the primitive action space
        best_node = self._inner_monte_carlo_tree_search(root, sg, restore_info)
        best_path = best_node.action_history

        logger.info(
            "Inner MCTS done | subgoal='%s' | best_path=%s | Q=%.3f | N=%d",
            sg, best_path,
            best_node.quality_value / max(1, best_node.visit_count),
            best_node.visit_count,
        )

        if not best_path:
            return False, -1.0, []

        # Execute the best found path for real.
        # Restore once more so we start from a clean subgoal-start state.
        cur_obs, _ = self._restore_and_replay(restore_info, [])
        executed: List[str] = []
        success = False

        for action in best_path:
            if action == "done":
                success = True
                break
            if action == "failure":
                break
            try:
                obs_ret = self.env.llm_skill_interact(action)
                cur_obs = obs_ret.get("message", cur_obs)
                executed.append(action)
            except Exception as exc:
                logger.warning("Inner MCTS commit failed action='%s': %s", action, exc)
                break
            if self._is_subgoal_satisfied(sg, action, cur_obs):
                success = True
                break

        mean_q = best_node.quality_value / max(1, best_node.visit_count)
        reward = mean_q if success else -abs(mean_q)
        logger.info(
            "Inner MCTS commit | subgoal='%s' | success=%s | reward=%.3f | executed=%s",
            sg, success, reward, executed,
        )
        return success, reward, executed
