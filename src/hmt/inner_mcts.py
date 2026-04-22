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

                # Prepend original task + current subgoal so the LLM always
                # knows both what the overall goal is and what step it is on.
                subgoal = getattr(state, "subgoal", "")
                nl_inst = getattr(state, "nl_inst", "")
                task_ctx = ""
                if nl_inst:
                    task_ctx += f"Overall task: {nl_inst}\n"
                if subgoal and subgoal != nl_inst:
                    task_ctx += f"Current subgoal: {subgoal}\n"
                sys_prompt = (task_ctx + "\n" + _SYS).strip() if task_ctx else _SYS

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
                return sys_prompt, msgs

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

        action_priors holds LLM-ranked per-action probabilities computed once
        on first expansion (via _llm_rank_candidates).  The reactree_action
        seed is pre-loaded with prob 2.0 so UCT explores it first.
        """

        def __init__(
            self,
            action_history: List[str],
            obs_list: List[str],
            available_commands: List[str],
            action_prob: float = 1.0,
            parent: Optional["ActionMCTSWrapper._InnerNode"] = None,
            action_priors: Optional[Dict[str, float]] = None,
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
            # Per-action priors populated lazily on first expand (or pre-seeded at root)
            self.action_priors: Dict[str, float] = action_priors or {}

        @property
        def obs(self) -> str:
            return self.obs_list[-1] if self.obs_list else ""

        def is_fully_expanded(self) -> bool:
            return all(a in self.tried_actions for a in self.available_commands)

    # -----------------------------------------------------------------------
    # _RolloutState — thin wrapper so mcts.llm_inference() can read obs/history
    # -----------------------------------------------------------------------

    class _RolloutState:
        def __init__(
            self,
            obs: str,
            obs_list: List[str],
            action_history: List[str],
            subgoal: str = "",
            nl_inst: str = "",
        ):
            self.obs = obs
            self.obs_list = list(obs_list)
            self.action_history = list(action_history)
            self.subgoal = subgoal
            self.nl_inst = nl_inst

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

    def _llm_rank_candidates(
        self,
        obs: str,
        available_commands: List[str],
        subgoal: str,
        reactree_action: Optional[str] = None,
        top_k: int = 8,
    ) -> Dict[str, float]:
        """
        Ask the LLM to reason about which actions are most likely to succeed
        given the current observation and subgoal, then return a prior-prob dict.

        Called once per node on its first expansion so MCTS explores in
        observation-informed order rather than picking candidates blindly.

        reactree_action (the outer planner's suggestion) is pre-boosted to
        prob=2.0 so UCT selects it first unless evidence suggests otherwise.
        """
        if not available_commands:
            priors: Dict[str, float] = {}
            if reactree_action:
                priors[reactree_action] = 2.0
            return priors

        k = min(top_k, len(available_commands))
        cmds_str = "\n".join(
            f"  {i + 1}. {cmd}" for i, cmd in enumerate(available_commands[:20])
        )
        prompt = (
            f"Subgoal: {subgoal}\n"
            f"Observation: {obs}\n\n"
            f"Available actions:\n{cmds_str}\n\n"
            f"List the {k} most helpful actions to achieve the subgoal, "
            f"one per line, most helpful first:"
        )

        ranked: List[str] = []
        try:
            import guidance
            llm_obj = getattr(self.llm_agent, "llm", None)
            if llm_obj is None:
                raise AttributeError("llm_agent.llm not found")
            lm = llm_obj + prompt
            lm += guidance.gen(stop="\n\n", name="ranking", max_tokens=250, temperature=0)
            ranking_text: str = (lm["ranking"] or "").strip()

            for line in ranking_text.split("\n"):
                line = line.strip().lstrip("0123456789.-) ").strip()
                if not line:
                    continue
                line_lower = line.lower()
                line_words = set(line_lower.split())
                best_cmd: Optional[str] = None
                best_overlap = 0
                for cmd in available_commands:
                    overlap = len(line_words & set(cmd.lower().split()))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_cmd = cmd
                if best_cmd and best_overlap >= 1 and best_cmd not in ranked:
                    ranked.append(best_cmd)
                if len(ranked) >= k:
                    break
        except Exception as exc:
            logger.warning("_llm_rank_candidates failed: %s", exc)

        # Assign probs: rank 1 → k/k, rank 2 → (k-1)/k, …
        priors = {}
        for i, cmd in enumerate(ranked):
            priors[cmd] = float(k - i) / k
        # Unranked actions get a small uniform prior so they can still be tried
        for cmd in available_commands:
            if cmd not in priors:
                priors[cmd] = 0.1
        # Pre-boost the reactree-provided action so it is explored first
        if reactree_action:
            priors[reactree_action] = max(priors.get(reactree_action, 0.1), 2.0)
            logger.debug(
                "_llm_rank_candidates: reactree_action='%s' boosted to 2.0", reactree_action
            )

        # For find/pick-up subgoals, boost "open" actions for containers
        # visible in the current observation. Cabinets and drawers hide objects
        # until opened — without this boost, MCTS ignores them and keeps
        # navigating to new locations instead of searching inside containers.
        sg_lower = subgoal.lower()
        is_find_subgoal = any(
            kw in sg_lower for kw in ("find", "pick up", "get", "fetch", "grab")
        )
        if is_find_subgoal and obs:
            obs_lower = obs.lower()
            import re as _re
            # Extract container types mentioned in obs (cabinet, drawer, fridge, etc.)
            container_types = _re.findall(
                r"\b(cabinet|drawer|fridge|refrigerator|safe|microwave|shelf)\b",
                obs_lower,
            )
            for cmd in available_commands:
                cmd_lower = cmd.lower()
                if cmd_lower.startswith("open ") and any(
                    ct in cmd_lower for ct in container_types
                ):
                    old_prior = priors.get(cmd, 0.1)
                    priors[cmd] = max(old_prior, 0.8)
                    logger.debug(
                        "_llm_rank_candidates: boosted open action '%s' to %.1f (find subgoal)",
                        cmd, priors[cmd],
                    )

        logger.debug(
            "_llm_rank_candidates | subgoal='%s' | top=%s | reactree='%s'",
            subgoal, ranked[:3], reactree_action,
        )
        return priors

    def select_best_action(
        self,
        obs: str,
        subgoal: str,
        reactree_action: Optional[str] = None,
    ) -> str:
        """
        Explore inner_budget primitive action candidates and return the single
        best one.  Uses _llm_rank_candidates (one LLM call) to rank all
        available actions with the reactree_action hint getting priority, then
        returns the top-ranked action.  No env steps are taken here — the
        caller executes the returned action and continues its own loop.
        """
        available = self._current_skill_set()
        if not available:
            return reactree_action or "done"

        priors = self._llm_rank_candidates(
            obs=obs,
            available_commands=available,
            subgoal=subgoal,
            reactree_action=reactree_action,
            top_k=max(self.budget, 5),
        )
        best = max(available, key=lambda a: priors.get(a, 0.1))
        logger.info(
            "select_best_action | subgoal='%s' | hint='%s' | best='%s' | prior=%.3f",
            subgoal, reactree_action, best, priors.get(best, 0.1),
        )
        return best

    @staticmethod
    def _extract_target_object(subgoal: str) -> Optional[str]:
        """
        Extract the main object name from a subgoal string.

        Examples:
          "find and pick up knife"    → "knife"
          "pick up Potato (1)"        → "potato"
          "go to CounterTop (1)"      → "countertop"
          "slice the potato"          → "potato"
        """
        import re as _re
        text = subgoal.lower().strip()
        # Strip leading verb phrases to reach the object
        text = _re.sub(
            r"^(find and pick up|pick up|go to|put|place|slice|cook|clean|"
            r"heat|cool|open|close|toggle|turn on|turn off|find)\s+",
            "", text,
        )
        # Drop trailing parenthesised instance numbers: "knife (1)" → "knife"
        text = _re.sub(r"\s*\(\d+(?:,\s*\d+)*\)", "", text).strip()
        return text if text else None

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
        2. On the node's first expansion, call _llm_rank_candidates() once to
           build observation-informed action priors (reactree_action pre-boosted).
        3. Pick the highest-prior untried action — so MCTS explores in LLM-reasoned
           order rather than blindly.
        4. Execute that action, observe result, build child node.
        """
        full_history = self._committed_history + node.action_history
        cur_obs, available = self._restore_and_replay(restore_info, full_history)

        untried = [a for a in available if a not in node.tried_actions]
        if not untried:
            return None

        # Lazily build ranked priors on first expansion of this node.
        # This is one LLM call per unique env state, not one per MCTS iteration.
        if not node.action_priors:
            node.action_priors = self._llm_rank_candidates(
                obs=cur_obs,
                available_commands=available,
                subgoal=getattr(self, "_current_subgoal", ""),
                reactree_action=getattr(self, "_reactree_action", None),
            )

        # Pick highest-prior untried action (greedy on LLM ranking).
        action = max(untried, key=lambda a: node.action_priors.get(a, 0.1))
        action_prob = node.action_priors.get(action, 1.0)
        node.tried_actions.append(action)

        logger.debug(
            "EXPAND | depth=%d | action='%s' (prior=%.3f) | reactree='%s' | untried_left=%d",
            len(node.action_history), action, action_prob,
            getattr(self, "_reactree_action", None), len(untried) - 1,
        )

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
        full_history = self._committed_history + node.action_history
        cur_obs, _ = self._restore_and_replay(restore_info, full_history)
        action_history = list(node.action_history)
        obs_list = list(node.obs_list)
        gamma = 0.95
        sim_discount = 1.0
        max_sim_steps = max(1, self.budget - len(action_history))

        logger.info(
            "  │  SIMULATE | start_depth=%d | max_sim_steps=%d",
            len(node.action_history), max_sim_steps,
        )
        target_object = self._extract_target_object(subgoal)
        logger.debug("  │  SIMULATE target_object='%s'", target_object)
        failed_actions: List[str] = []
        for _ in range(max_sim_steps):
            available = [a for a in self._current_skill_set() if a not in failed_actions]
            if not available:
                return -1.0 * sim_discount

            state = self._RolloutState(
                obs=cur_obs,
                obs_list=obs_list,
                action_history=action_history,
                subgoal=getattr(self, "_current_subgoal", ""),
                nl_inst=getattr(self, "_current_nl_inst", ""),
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
                logger.info(
                    "  │  sim step | action='%s' | obs='%s'", action, cur_obs,
                )
                # If env rejected the action, exclude it from future steps
                if obs_ret.get("success") is False or "already" in cur_obs.lower() or "not close to you" in cur_obs.lower():
                    failed_actions.append(action)
            except Exception as exc:
                logger.warning("_inner_default_policy: env step failed: %s", exc)
                return -1.0

            action_history.append(action)
            obs_list.append(cur_obs)

            # Partial reward signal: target object visible — log it but keep
            # simulating so MCTS can reach the actual pick-up action.
            if target_object and target_object in cur_obs.lower():
                logger.info(
                    "  │  sim object found | target='%s' | action='%s' | continuing sim",
                    target_object, action,
                )

            # Fast-path: primitive action text matches subgoal
            if self._is_subgoal_satisfied(subgoal, action, cur_obs):
                logger.info("  │  sim SUCCESS (text match) | action='%s'", action)
                return 1.0 * sim_discount

            # Real env signal: check transition reward and goal conditions met.
            # This catches composite goals (e.g. 'find and pick up knife') where
            # no single action text matches but the env knows conditions were met.
            try:
                transition_reward = self.env.get_transition_reward()
                conditions_met, conditions_total = self.env.get_goal_conditions_met()
                logger.info(
                    "  │  env signal | transition_reward=%.3f | conditions=%d/%d",
                    transition_reward, conditions_met, conditions_total,
                )
                if transition_reward > 0 and conditions_met > 0:
                    logger.info("  │  sim SUCCESS (env reward) | action='%s'", action)
                    return 1.0 * sim_discount
            except Exception as exc:
                logger.debug("env reward check failed: %s", exc)

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
            logger.info(
                "  ├─ inner iter %d/%d | subgoal='%s' | tree_nodes=%d",
                i + 1, self.budget, subgoal,
                sum(1 + len(c.children) for c in root.children) + 1,
            )
            leaf = self._inner_tree_policy(root, restore_info)           # SELECT / EXPAND
            logger.info(
                "  │  SELECT/EXPAND → depth=%d | path=%s",
                len(leaf.action_history), leaf.action_history,
            )
            reward = self._inner_default_policy(leaf, subgoal, restore_info)  # SIMULATE
            self._inner_backup(leaf, reward)                             # BACKUP
            logger.info(
                "  │  SIMULATE reward=%.3f | BACKUP done",
                reward,
            )

        best = self._inner_best_child(root, explore=False)  # exploit only
        best_node = best if best is not None else root
        logger.info(
            "  └─ inner MCTS best | path=%s | Q/N=%.3f",
            best_node.action_history,
            best_node.quality_value / max(1, best_node.visit_count),
        )
        return best_node

    # -----------------------------------------------------------------------
    # LLM fallback greedy rollout (no tree, no restore — used when
    # restore_info is unavailable or MCTSAlgorithm failed to import)
    # -----------------------------------------------------------------------

    def _solve_with_reactexpand_rollout(
        self, subgoal: str, obs: str, nl_inst: str = ""
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

        # Use original instruction if provided so the LLM knows both the
        # overall task and the current subgoal. Fall back to subgoal alone.
        if nl_inst and nl_inst != subgoal:
            message = f"Current subgoal: {subgoal}"
        else:
            message = None
        nl_inst_info = {"nl_inst": nl_inst if nl_inst else subgoal, "message": message, "task_type": task_type, "depth": 0}
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
        nl_inst: str = "",
        reactree_action: Optional[str] = None,
        committed_history: Optional[List[str]] = None,
    ) -> Tuple[bool, float, List[str]]:
        """
        Solve one subgoal using inner MCTS.

        Parameters
        ----------
        subgoal           : natural-language subgoal text
        obs               : current env observation (after committed_history actions)
        restore_info      : scene restore dict from OuterMCTSPlanner._sim_restore_info.
                            When provided, enables full MCTS (SELECT→EXPAND→SIMULATE→BACKUP).
                            When None, falls back to greedy LLM rollout.
        reactree_action   : optional primitive action suggested by the outer ReAcTree planner
                            (e.g. "go to counter top 1").  Pre-seeded with prior=2.0 so
                            UCT explores it first; inner MCTS then explores alternatives
                            ranked by LLM reasoning over the current observation.
        committed_history : actions already committed to the env before this MCTS call.
                            _restore_and_replay replays these first so the MCTS tree
                            starts from the correct current state rather than the
                            subgoal-start state.  Each successive call from
                            _execute_subgoal_with_reactree grows this list by one action.

        Returns
        -------
        (success, reward, executed_action_sequence)
        """
        sg = (subgoal or "").strip()
        if not sg:
            return False, -1.0, []

        logger.info(
            "Inner MCTS ENTRY | subgoal='%s' | mcts=%s | restore=%s | reactree_action='%s'",
            sg, self._can_use_mcts_backend, restore_info is not None, reactree_action,
        )

        # Store for use by _RolloutState in EXPAND and SIMULATE
        self._current_subgoal = sg
        self._current_nl_inst = nl_inst or sg
        self._reactree_action = (reactree_action or "").strip() or None
        # committed_history: actions already executed in the real env before this call.
        # _inner_expand and _inner_default_policy prepend these to node.action_history
        # before calling _restore_and_replay so each node is reached from the correct
        # current state rather than the subgoal-start state.
        self._committed_history: List[str] = list(committed_history or [])

        # Fall back to greedy if backend missing or no restore_info
        if not self._can_use_mcts_backend or restore_info is None:
            reason = "backend unavailable" if not self._can_use_mcts_backend else "no restore_info"
            logger.info("Inner MCTS using greedy fallback (%s) for '%s'", reason, sg)
            return self._solve_with_reactexpand_rollout(sg, obs, nl_inst=nl_inst)

        # Build root node at subgoal-start state.
        # Pre-seed root priors so the reactree_action is tried first and LLM reasoning
        # over the initial observation informs the entire exploration budget.
        available = self._current_skill_set()
        root_priors = self._llm_rank_candidates(
            obs=obs or "No observation available.",
            available_commands=available,
            subgoal=sg,
            reactree_action=self._reactree_action,
        )
        root = self._InnerNode(
            action_history=[],
            obs_list=[obs or "No observation available."],
            available_commands=available,
            action_prob=1.0,
            parent=None,
            action_priors=root_priors,
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
        # Restore to the committed state (not bare subgoal-start) so we continue
        # from where the caller left off, not from the beginning.
        cur_obs, _ = self._restore_and_replay(restore_info, self._committed_history)
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
