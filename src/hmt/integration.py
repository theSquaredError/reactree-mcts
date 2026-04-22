"""
AlfredReactreeWithHMT — plugs the two-level HMT search into AlfredReactree.

Overrides collect_llm() to use OuterMCTSPlanner instead of the original
one-shot Expand logic, while keeping full compatibility with the evaluator.
"""

import logging
from typing import Any, Dict

from alfred.alfred_reactree import AlfredReactree
from alfred.utils import dotdict, load_task_json

from .constants import Node
from .outer_mcts import OuterMCTSPlanner
from .tree_utils import export_outer_tree_artifacts
from .types import DecompositionState

logger = logging.getLogger(__name__)


class AlfredReactreeWithHMT(AlfredReactree):
    """
    Extended AlfredReactree with hierarchical MCTS decomposition search.

    When use_hmt=True (default), collect_llm() is replaced by a two-level loop:
      - Outer MCTS explores candidate task decompositions.
      - Inner MCTS (inside OuterMCTSPlanner) searches primitive action sequences.

    When use_hmt=False, falls back to the original single-shot ReAcTree.
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
                outer_budget, inner_budget, decomp_candidate_count,
            )
            self.outer_mcts = OuterMCTSPlanner(
                cfg=cfg,
                llm_agent=llm_agent,
                env=env,
                outer_budget=outer_budget,
                inner_budget=inner_budget,
                decomp_candidate_count=decomp_candidate_count,
            )

    # -----------------------------------------------------------------------
    # Entry points
    # -----------------------------------------------------------------------

    def collect_llm(self, task_d: Dict, args_dict: Dict) -> Dict:
        if self.use_hmt:
            return self.collect_llm_with_hmt(task_d, args_dict)
        return super().collect_llm(task_d, args_dict)

    def collect_llm_with_hmt(self, task_d: Dict, args_dict: Dict) -> Dict:
        """Hierarchical collection: outer decomposition search + inner action search."""
        self.logger.info(
            "Starting HMT collection | task=%s | repeat_idx=%s",
            task_d.get("task"), task_d.get("repeat_idx"),
        )
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

        # Get initial observation
        try:
            init_obs = self.env.init_reset(traj_data)
            init_obs_text = init_obs["text"]
            init_skill_set = self.llm_agent.update_skill_set(init_obs)
        except Exception:
            self.logger.exception("env.init_reset failed; using instruction as fallback obs")
            init_obs_text = nl_inst
            init_skill_set = ["done", "failure"]

        # ---- Wire runtime state into the OuterMCTSPlanner ----
        self.outer_mcts._current_traj_data = traj_data
        self.outer_mcts._cur_step_id = 1
        self.outer_mcts._cur_decision_id = 1
        self.outer_mcts._nl_inst = nl_inst  # forward to inner MCTS prompts
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

        # Build root decomposition state
        root_state = DecompositionState(
            goal=nl_inst,
            env_snapshot={
                "observation": init_obs_text,
                "task_type": traj_data.get("task_type", "unknown"),
                "skill_set": init_skill_set,
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
            self.cfg, root_node, task_d.get("task", nl_inst)
        )

        if chosen is not None:
            self.logger.info(
                "Chosen HMT decomposition | control_flow=%s | subgoals=%s | q=%.3f | visits=%d",
                chosen["control_flow"],
                chosen["subgoals"],
                getattr(getattr(root_node, "selected_child", root_node), "quality_value", root_node.quality_value),
                getattr(getattr(root_node, "selected_child", root_node), "visit_count", root_node.visit_count),
            )

        terminate_info: Dict[str, Any] = {
            "success": execution_result["success"],
            "terminate": execution_result.get("mode", "hmt"),
            "steps": execution_result["steps"],
            "nl_inst": nl_inst,
        }
        self.logger.info("HMT terminate_info: %s", terminate_info)

        hmt_outer: Dict[str, Any] = {
            "control_flow": chosen["control_flow"] if chosen else None,
            "subgoals": chosen["subgoals"] if chosen else [],
            "prior_prob": chosen["prior_prob"] if chosen else 0.0,
            "value": getattr(
                getattr(root_node, "selected_child", root_node), "quality_value", root_node.quality_value
            ),
            "visits": getattr(
                getattr(root_node, "selected_child", root_node), "visit_count", root_node.visit_count
            ),
            "reward": execution_result["reward"],
            "subgoal_results": execution_result["subgoal_results"],
            "tree_artifacts": tree_artifacts,
        }
        terminate_info["hmt_outer"] = hmt_outer
        return terminate_info

    def run(self, task_d: Dict, args_dict: Dict, log) -> Dict:
        """Evaluator-compatible entry point (mirrors AlfredReactree.run)."""
        from PIL import Image as _Image
        log.info(task_d)

        try:
            self.env.vis_log = [
                {"action": "init", "images": _Image.fromarray(self.env.last_event.frame)}
            ]
        except Exception:
            self.env.vis_log = []

        terminate_info = self.collect_llm_with_hmt(task_d, args_dict)
        env_success = self.env.get_goal_satisfied()
        self.logger.info(
            "Success check | planner=%s | env=%s | task=%s",
            terminate_info.get("success"), env_success, task_d.get("task"),
        )
        terminate_info["success"] = env_success
        return terminate_info
