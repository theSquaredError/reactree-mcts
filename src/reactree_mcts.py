"""
reactree_mcts.py — ReAcTree + MCTS using ALFWorld.

ReAcTree Think / Act / Expand loop unchanged.
At every primitive Act step PrimitiveMCTS searches for the best action
sequence using ALFWorld's text environment as a cheap, resettable simulator.

State forking: env.reset() + replay replaces AI2-THOR's restore_scene().
Reward signal: info['goal_condition_success_rate'] — no custom reward shaping.
Action priors:  LLM generation over admissible_commands.
"""
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _extract_task(obs: str) -> str:
    """Last non-empty line of ALFWorld initial obs is the task description."""
    for line in reversed(obs.splitlines()):
        line = line.strip()
        if line:
            return line.split(":")[-1].strip() if ":" in line else line
    return obs.strip()


def _best_match(text: str, options: List[str]) -> str:
    """Return the option whose text best matches the generated string."""
    tl = text.lower()
    for opt in options:
        if opt.lower() == tl:
            return opt
    for opt in options:
        if opt.lower() in tl or tl in opt.lower():
            return opt
    return options[0]


# ─────────────────────────────────────────────────────────────────────────────
# MCTSNode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    action_history: List[str] = field(default_factory=list)
    obs_list: List[str] = field(default_factory=list)
    available_actions: List[str] = field(default_factory=list)
    action_prob: float = 1.0
    parent: Optional["MCTSNode"] = field(default=None, repr=False)
    children: List["MCTSNode"] = field(default_factory=list)
    tried_actions: List[str] = field(default_factory=list)
    visit_count: int = 0
    quality_value: float = 0.0
    action_priors: Dict[str, float] = field(default_factory=dict)

    @property
    def obs(self) -> str:
        return self.obs_list[-1] if self.obs_list else ""

    def is_fully_expanded(self) -> bool:
        return bool(self.available_actions) and all(
            a in self.tried_actions for a in self.available_actions
        )


# ─────────────────────────────────────────────────────────────────────────────
# PrimitiveMCTS
# ─────────────────────────────────────────────────────────────────────────────

class PrimitiveMCTS:
    """
    MCTS over primitive ALFWorld actions for one subgoal.
    SELECT → EXPAND → SIMULATE → BACKUP, repeated budget times.
    State forking: env.reset() + replay — cheap for ALFWorld text env.
    """

    def __init__(
        self, env, llm_agent, budget: int = 5, gamma: float = 0.95, max_depth: int = 4
    ):
        self.env = env
        self.llm_agent = llm_agent
        self.budget = budget
        self.gamma = gamma
        self.max_depth = max_depth

    # ── state restoration ─────────────────────────────────────────────────────

    def _restore(self, history: List[str]) -> Tuple[str, List[str], float, bool]:
        """Reset env to task start, replay history. Returns (obs, available, score, done)."""
        obs, info = self.env.reset()
        sc = info["goal_condition_success_rate"][0] if "goal_condition_success_rate" in info else 0.0
        done = False
        for action in history:
            obs, sc, done, info = self.env.step(action)
            obs, sc, done = obs[0], sc[0], done[0]
            if done:
                break
        available = [c for c in info["admissible_commands"][0] if c != "look"]
        return obs, available, sc, done

    # ── action ranking ────────────────────────────────────────────────────────

    def _rank_actions(
        self,
        obs: str,
        actions: List[str],
        subgoal: str,
        hint: Optional[str] = None,
        nl_inst: str = "",
    ) -> Dict[str, float]:
        """
        Ask the LLM to pick the best next action; give it prior=1.0.
        hint (ReAcTree Act suggestion) gets a boost to 0.8.
        All other actions get 0.1.
        """
        priors = {a: 0.1 for a in actions}
        if not actions:
            return priors

        task_line = f"Task: {nl_inst}\n" if nl_inst and nl_inst != subgoal else ""
        action_list = "\n".join(f"- {a}" for a in actions)
        user_content = (
            f"{task_line}Subgoal: {subgoal}\n"
            f"Observation: {obs}\n"
            f"Choose the single best next action from this list:\n{action_list}\n"
            f"Reply with only the exact action text."
        )
        chosen_raw = self.llm_agent._generate_from_messages(
            [{"role": "user", "content": user_content}], max_new_tokens=30
        )
        chosen = _best_match(chosen_raw.strip(), actions)
        logger.info("  _rank_actions | chosen='%s' | hint='%s'", chosen, hint)
        priors[chosen] = 1.0

        if hint and hint in priors:
            priors[hint] = max(priors[hint], 0.8)
        return priors

    # ── UCT ───────────────────────────────────────────────────────────────────

    def _uct(self, node: MCTSNode, explore: bool) -> float:
        if node.visit_count == 0:
            return float("inf")
        parent_n = node.parent.visit_count if node.parent else 1
        C = 1.0 / math.sqrt(2.0) if explore else 0.0
        return (
            node.quality_value / node.visit_count
            + C * math.sqrt(2.0 * math.log(max(1, parent_n)) / node.visit_count)
            * node.action_prob
        )

    # ── SELECT ────────────────────────────────────────────────────────────────

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.is_fully_expanded() and node.children:
            if len(node.action_history) >= self.max_depth:
                break
            node = max(node.children, key=lambda c: self._uct(c, explore=True))
        return node

    # ── EXPAND ────────────────────────────────────────────────────────────────

    def _expand(
        self,
        node: MCTSNode,
        committed: List[str],
        subgoal: str,
        hint: Optional[str],
        nl_inst: str,
    ) -> Optional[MCTSNode]:
        if len(node.action_history) >= self.max_depth:
            return None

        obs, available, _, _ = self._restore(committed + node.action_history)
        node.available_actions = available
        untried = [a for a in available if a not in node.tried_actions]
        if not untried:
            return None

        if not node.action_priors:
            node.action_priors = self._rank_actions(obs, available, subgoal, hint, nl_inst)

        action = max(untried, key=lambda a: node.action_priors.get(a, 0.1))
        prob = node.action_priors.get(action, 0.1)
        node.tried_actions.append(action)

        logger.info(
            "  EXPAND | depth=%d | action='%s' (prob=%.2f) | path=%s",
            len(node.action_history), action, prob, node.action_history,
        )

        obs, sc, done, info = self.env.step(action)
        child = MCTSNode(
            action_history=node.action_history + [action],
            obs_list=node.obs_list + [obs[0]],
            available_actions=[c for c in info["admissible_commands"][0] if c != "look"],
            action_prob=prob,
            parent=node,
        )
        node.children.append(child)
        return child

    # ── SIMULATE ──────────────────────────────────────────────────────────────

    def _simulate(
        self,
        node: MCTSNode,
        committed: List[str],
        subgoal: str,
        nl_inst: str,
    ) -> Tuple[float, bool]:
        """Greedy rollout from node state. Returns (discounted_score, success)."""
        obs, available, sc, done = self._restore(committed + node.action_history)
        discount = 1.0
        best_score = sc
        seen = set(node.action_history)

        for _ in range(20):
            if done:
                return best_score * discount, True
            candidates = [a for a in available if a not in seen] or available
            if not candidates:
                break
            priors = self._rank_actions(obs, candidates, subgoal, nl_inst=nl_inst)
            action = max(candidates, key=lambda a: priors.get(a, 0.1))
            seen.add(action)
            discount *= self.gamma
            obs, sc, done, info = self.env.step(action)
            obs, sc, done = obs[0], sc[0], done[0]
            available = [c for c in info["admissible_commands"][0] if c != "look"]
            best_score = max(best_score, sc)
            logger.info("  SIMULATE | '%s' | score=%.3f | done=%s", action, best_score, done)

        return best_score * discount, done

    # ── BACKUP ────────────────────────────────────────────────────────────────

    def _backup(self, node: MCTSNode, reward: float) -> None:
        cur: Optional[MCTSNode] = node
        discount = 1.0
        while cur is not None:
            cur.visit_count += 1
            cur.quality_value += reward * discount
            discount *= cur.action_prob * self.gamma
            cur = cur.parent

    # ── solve ─────────────────────────────────────────────────────────────────

    def solve(
        self,
        subgoal: str,
        obs: str,
        committed: List[str],
        hint: Optional[str] = None,
        nl_inst: str = "",
    ) -> Tuple[bool, List[str], str, List[str]]:
        """
        Run MCTS for budget iterations then execute the best found path.
        Returns (success, executed_actions, final_obs, final_available).
        After return the env is at the state after committed + executed_actions.
        """
        _, available, _, _ = self._restore(committed)
        root = MCTSNode(
            obs_list=[obs],
            available_actions=available,
            action_priors=self._rank_actions(obs, available, subgoal, hint, nl_inst),
        )

        logger.info(
            "MCTS start | subgoal='%s' | hint='%s' | budget=%d | max_depth=%d",
            subgoal, hint, self.budget, self.max_depth,
        )
        best_success: Optional[Tuple[float, List[str]]] = None

        for i in range(self.budget):
            logger.info("── iter %d/%d ──", i + 1, self.budget)
            node = self._select(root)
            logger.info(
                "  SELECT → depth=%d | path=%s | expanded=%s",
                len(node.action_history), node.action_history, node.is_fully_expanded(),
            )
            if not node.is_fully_expanded():
                expanded = self._expand(node, committed, subgoal, hint, nl_inst)
                if expanded is not None:
                    node = expanded

            reward, success = self._simulate(node, committed, subgoal, nl_inst)
            if success and (best_success is None or reward > best_success[0]):
                best_success = (reward, list(node.action_history))
                logger.info(
                    "  SUCCESS candidate | reward=%.3f | path=%s", reward, node.action_history,
                )
            self._backup(node, reward)
            logger.info(
                "  BACKUP | reward=%.3f | children: %s",
                reward,
                [(c.action_history[-1] if c.action_history else "?",
                  round(c.quality_value / c.visit_count, 3) if c.visit_count else 0)
                 for c in root.children],
            )

        if not root.children:
            logger.info("MCTS: no children expanded | subgoal='%s'", subgoal)
            return False, [], obs, available

        best_child = max(
            root.children,
            key=lambda c: c.quality_value / c.visit_count if c.visit_count else -float("inf"),
        )
        path = (best_success[1] if best_success else None) or best_child.action_history
        logger.info("MCTS done | best_path=%s", path)

        # Execute best path from committed state for real
        cur_obs, cur_available, _, _ = self._restore(committed)
        executed: List[str] = []
        final_score = 0.0
        final_done = False

        for action in path:
            obs, sc, done, info = self.env.step(action)
            cur_obs, final_score, final_done = obs[0], sc[0], done[0]
            executed.append(action)
            cur_available = [c for c in info["admissible_commands"][0] if c != "look"]
            if final_done:
                break

        return final_done or final_score >= 1.0, executed, cur_obs, cur_available


# ─────────────────────────────────────────────────────────────────────────────
# AlfWorldLlmAgent — ReAcTree planning agent for ALFWorld
# ─────────────────────────────────────────────────────────────────────────────

def _read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


class AlfWorldLlmAgent:
    """
    ReAcTree planning agent (Think / Act / Expand) for ALFWorld.
    Uses any HuggingFace causal-LM (default: Llama-3.1-8B-Instruct) via chat template.
    Maintains self.messages as a rolling chat history; no guidance dependency.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.llm_agent.model_name

        auth_token = getattr(cfg.llm_agent, "hf_auth_token", None)
        model_kwargs: dict = {"torch_dtype": torch.float16, "device_map": "auto"}
        if getattr(cfg.llm_agent, "load_in_8bit", False):
            model_kwargs["load_in_8bit"] = True
        if auth_token:
            model_kwargs["token"] = auth_token

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()
        self.messages: List[dict] = []

    def _generate_from_messages(self, messages: List[dict], max_new_tokens: int = 200) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

    def reset(self, nl_inst_info: dict, init_obs: str) -> None:
        prompt_path = os.path.join(
            self.cfg.prompt.sys_prompt_root_dir,
            f"{self.cfg.task_planner}.txt",
        )
        system_prompt = _read_file(prompt_path)
        nl_inst = nl_inst_info["nl_inst"]
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Your task is to: {nl_inst}\n{init_obs}"},
        ]

    def plan_next_step(self, available_actions: List[str]) -> dict:
        raw = self._generate_from_messages(self.messages, max_new_tokens=200)
        self.messages.append({"role": "assistant", "content": raw})

        if raw.startswith("Think:"):
            return {"next_step_class": "Think", "next_step": raw[6:].strip()}

        if raw.startswith("Act:"):
            action_text = raw[4:].strip().split("\n")[0].strip()
            skill_set = available_actions + ["done", "failure"]
            return {"next_step_class": "Act", "next_step": _best_match(action_text, skill_set)}

        if raw.startswith("Expand:"):
            control_flow = "sequence"
            conditions = ""
            for line in raw.splitlines()[1:]:
                line = line.strip()
                if "control flow:" in line:
                    for cf in ("sequence", "fallback", "parallel"):
                        if cf in line:
                            control_flow = cf
                            break
                elif "subgoals:" in line:
                    conditions = line.split("subgoals:", 1)[-1].strip()
            return {
                "next_step_class": "Expand",
                "next_step": {"control_flow": control_flow, "conditions": conditions},
            }

        # Default: treat unrecognised output as a Think step
        return {"next_step_class": "Think", "next_step": raw}

    def add_obs(self, obs_text: str) -> None:
        self.messages.append({"role": "user", "content": obs_text})


# ─────────────────────────────────────────────────────────────────────────────
# ReactreeMCTSLoop
# ─────────────────────────────────────────────────────────────────────────────

class ReactreeMCTSLoop:
    """
    ReAcTree Think / Act / Expand loop with MCTS at every primitive Act.
    """

    def __init__(
        self, cfg, llm_agent, env,
        mcts_budget: int = 5, mcts_max_depth: int = 4, gamma: float = 0.95,
    ):
        self.llm_agent = llm_agent
        self.mcts = PrimitiveMCTS(
            env, llm_agent, budget=mcts_budget, gamma=gamma, max_depth=mcts_max_depth
        )
        self.max_decisions = getattr(cfg.llm_agent, "max_decisions", 30)
        self.max_depth = getattr(cfg.llm_agent, "max_depth", 4)

    def run_goal(
        self,
        goal: str,
        obs: str,
        available: List[str],
        committed: List[str],
        depth: int = 0,
    ) -> dict:
        """
        Run ReAcTree for one goal.
        Returns {"success", "steps", "actions", "obs", "available"}.
        committed is updated in-place as MCTS executes actions.
        """
        self.llm_agent.reset(
            {"nl_inst": goal, "message": None, "task_type": "", "depth": depth}, obs
        )
        indent = "  " * depth
        logger.info("%s[goal d=%d] %s", indent, depth, goal)

        steps = 0
        local_actions: List[str] = []

        for decision_id in range(1, self.max_decisions + 1):
            next_step = self.llm_agent.plan_next_step(available)
            step_class = next_step["next_step_class"]
            step_value = next_step["next_step"]

            if step_class == "Think":
                logger.info("%s  [d%d #%d] Think: %s", indent, depth, decision_id, step_value)
                continue

            if step_class == "Error":
                logger.warning("%s  [d%d #%d] Error — stopping", indent, depth, decision_id)
                break

            if step_class == "Expand":
                if depth >= self.max_depth:
                    logger.info("%s  Expand skipped (max_depth=%d)", indent, self.max_depth)
                    continue
                control_flow = step_value.get("control_flow", "sequence")
                subgoals = [
                    s.strip() for s in step_value.get("conditions", "").split(",") if s.strip()
                ]
                if not subgoals:
                    continue
                logger.info(
                    "%s  [d%d #%d] Expand: %s → %s",
                    indent, depth, decision_id, control_flow, subgoals,
                )
                result = self._run_control_flow(
                    control_flow, subgoals, obs, available, committed, depth + 1,
                )
                steps += result["steps"]
                local_actions.extend(result["actions"])
                return {
                    "success": result["success"],
                    "steps": steps,
                    "actions": local_actions,
                    "obs": result["obs"],
                    "available": result["available"],
                }

            if step_class == "Act":
                action = str(step_value).strip()

                if action == "done":
                    logger.info("%s  [d%d #%d] Act: done → SUCCESS", indent, depth, decision_id)
                    return {"success": True, "steps": steps, "actions": local_actions, "obs": obs, "available": available}

                if action == "failure":
                    logger.info("%s  [d%d #%d] Act: failure → FAIL", indent, depth, decision_id)
                    return {"success": False, "steps": steps, "actions": local_actions, "obs": obs, "available": available}

                logger.info(
                    "%s  [d%d #%d] Act: '%s' → MCTS (committed=%d)",
                    indent, depth, decision_id, action, len(committed),
                )
                success, mcts_actions, obs, available = self.mcts.solve(
                    subgoal=goal,
                    obs=obs,
                    committed=list(committed),
                    hint=action,
                    nl_inst=goal,
                )
                steps += len(mcts_actions)
                local_actions.extend(mcts_actions)
                committed.extend(mcts_actions)
                logger.info("%s    MCTS → success=%s | path=%s", indent, success, mcts_actions)

                if success:
                    return {"success": True, "steps": steps, "actions": local_actions, "obs": obs, "available": available}

                self.llm_agent.add_obs(
                    f"MCTS tried {mcts_actions} but did not complete the goal. "
                    f"Current observation: {obs}"
                )
                continue

        logger.info("%s  [d=%d] loop exhausted → FAIL", indent, depth)
        return {"success": False, "steps": steps, "actions": local_actions, "obs": obs, "available": available}

    def _run_control_flow(
        self,
        control_flow: str,
        subgoals: List[str],
        obs: str,
        available: List[str],
        committed: List[str],
        depth: int,
    ) -> dict:
        indent = "  " * (depth - 1)
        logger.info("%s[%s] %d subgoals: %s", indent, control_flow.upper(), len(subgoals), subgoals)
        total_steps = 0
        all_actions: List[str] = []
        results: List[bool] = []

        for i, subgoal in enumerate(subgoals):
            logger.info("%s  subgoal %d/%d: %s", indent, i + 1, len(subgoals), subgoal)
            result = self.run_goal(subgoal, obs, available, committed, depth)
            total_steps += result["steps"]
            all_actions.extend(result["actions"])
            results.append(result["success"])
            obs = result["obs"]
            available = result["available"]

            if control_flow == "sequence" and not result["success"]:
                logger.info("%s  sequence broken at subgoal %d", indent, i + 1)
                return {"success": False, "steps": total_steps, "actions": all_actions, "obs": obs, "available": available}
            if control_flow == "fallback" and result["success"]:
                logger.info("%s  fallback succeeded at subgoal %d", indent, i + 1)
                return {"success": True, "steps": total_steps, "actions": all_actions, "obs": obs, "available": available}

        success = all(results) if control_flow in ("sequence", "parallel") else any(results)
        logger.info("%s[%s] done | success=%s", indent, control_flow.upper(), success)
        return {"success": success, "steps": total_steps, "actions": all_actions, "obs": obs, "available": available}


# ─────────────────────────────────────────────────────────────────────────────
# run_task
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    cfg,
    llm_agent: AlfWorldLlmAgent,
    env,
    task_file: str,
    mcts_budget: int = 5,
    mcts_max_depth: int = 4,
) -> dict:
    """
    Run one ALFWorld task with ReactreeMCTS.
    env must be an AlfWorldEnv already initialised for task_file.
    Returns {"success", "steps", "actions", "nl_inst"}.
    """
    obs, info = env.reset()
    available = [c for c in info["admissible_commands"][0] if c != "look"]
    task = _extract_task(obs)
    logger.info("ReactreeMCTS | task=%s | task_file=%s", task, task_file)

    mcts_cfg = getattr(cfg, "reactree_mcts", None)
    gamma = float(getattr(mcts_cfg, "gamma", 0.95))

    loop = ReactreeMCTSLoop(
        cfg, llm_agent, env,
        mcts_budget=mcts_budget, mcts_max_depth=mcts_max_depth, gamma=gamma,
    )
    committed: List[str] = []
    result = loop.run_goal(task, obs, available, committed, depth=0)

    return {
        "success": result["success"],
        "steps": result["steps"],
        "actions": result["actions"],
        "nl_inst": task,
    }
