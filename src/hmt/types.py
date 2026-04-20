"""
Core data structures for the HMT two-level search.

DecompositionState  — state node at the outer (decomposition) MCTS level
DecompositionAction — an Expand candidate: control_flow + subgoal list
"""

import copy
from typing import Dict, List, Optional


class DecompositionState:
    """
    State at the decomposition level.

    Attributes
    ----------
    env_snapshot        : frozen copy of environment info (observation, task_type)
    goal                : current task goal string
    depth               : nesting depth in the decomposition tree
    executed_steps      : human-readable log of committed steps
    action_history      : machine signatures of actions taken
    succeed_subgoals    : number of subgoals completed successfully
    failed_subgoals     : number of subgoals that failed
    generated_candidates: DecompositionAction candidates produced so far
    candidate_generation_done: True once the LLM has been exhausted
    trajectory          : primitive actions executed to reach this state
    mcts_attempts       : per-iteration simulation records
    final_success_trajectory: structured record of executed subgoals + their actions
    """

    def __init__(
        self,
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
        final_success_trajectory: Optional[List[Dict]] = None,
    ):
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
        self.trajectory = list(trajectory or [])
        self.mcts_attempts = list(mcts_attempts or [])
        self.final_success_trajectory = list(final_success_trajectory or [])

    def clone(self) -> "DecompositionState":
        """Deep copy for branching exploration."""
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
    An Expand action at the decomposition level.

    Attributes
    ----------
    control_flow : 'sequence' | 'fallback' | 'parallel'
    subgoals     : ordered list of subgoal strings
    prior_prob   : LLM confidence score (0–1)
    """

    def __init__(self, control_flow: str, subgoals: List[str], prior_prob: float = 0.5):
        self.control_flow = control_flow
        self.subgoals = subgoals
        self.prior_prob = prior_prob

    def __repr__(self) -> str:
        return f"DecompAction({self.control_flow}: {self.subgoals[:2]}...)"

    def signature(self) -> str:
        return f"{self.control_flow}::{'||'.join(self.subgoals)}"
