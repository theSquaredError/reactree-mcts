The goal is to create a hierarchical+mcts task planning solution:
1. Given root node, the using mcts we can explore multiple sequence or subgoals path,
2. in each sequence for each subgoal also we use mcts for each of the subgoal's subgoal they might be primitive actions like provided by environment or other llm designed actions.
3. we use mcts over above loop to get a final trajectory


suppose a root node has generate a sequence of subgoal with control flow as: sequence, now with the mcts the each subgoal should be explored with taking subgoals next child as one candidate


Questions:
when a subgoal is explored under a root node how it gets reward,
how multiple primitive actions are simulated under one subgoals


OuterMCTSPlanner
│
├── Configuration
│   ├── outer_budget   — how many MCTS iterations per goal
│   ├── inner_budget   — budget passed to ActionMCTSWrapper for each subgoal  
│   └── decomp_candidate_count — max decomposition candidates to generate
│
├── Two sub-systems
│   ├── LLM (via llm_agent)         — generates Expand candidates
│   └── ActionMCTSWrapper           — solves primitive subgoals (inner loop)
│
└── Standard MCTS 4-phase loop (outer_monte_carlo_tree_search)
    ├── Selection   → outer_tree_policy
    ├── Expansion   → outer_expand / outer_expand_action
    ├── Simulation  → outer_default_policy → _evaluate_decomposition_action
    └── Backup      → outer_backup



4-phase loop per iterations

outer_monte_carlo_tree_search (outer_budget iterations)
  │
  ├─ outer_tree_policy(root)
  │    if role=="decomposition" → return it (already a leaf candidate)
  │    if fully_expanded        → UCT-select best existing child
  │    else                     → outer_expand (ask LLM, create new decomp child)
  │
  ├─ outer_default_policy(expand_node)
  │    reads node.decomposition_action (e.g. sequence::["pick knife","put in fridge"])
  │    calls _evaluate_decomposition_action
  │      → for each subgoal: _simulate_goal → inner_mcts_solve_subgoal
  │    returns (reward, leaf_node)
  │
  └─ outer_backup(leaf_node, reward)      ← reward propagates here



The propagation path for concrete example:

root (goal node, depth=0)
  └─ decomp_child_1 (depth=1, sequence::["pick knife","put in fridge"])
         ↑ reward * 0.95^1 applied here
         
  decomp_child_1 simulated → reward R earned
  
  Backup starts at decomp_child_1:
    decomp_child_1.Q += R * 1.0    (discount=1.0)
    decomp_child_1.N += 1
    
  Move up: cur = decomp_child_1.get_parent() = root
    root.Q += R * 0.95             (discount=0.95)
    root.N += 1
    
  Move up: cur = root.get_parent() = None → loop ends


  DecompositionState:

  Think of it as the snapshot the outer MCTS carries at each node:

Task context: goal text, environment snapshot, depth.
Search bookkeeping: generated_candidates, candidate_generation_done, action_history, executed_steps.
Outcome bookkeeping: succeed_subgoals, failed_subgoals, is_terminal.
Trace/log payloads: trajectory, mcts_attempts, final_success_trajectory.
Main fields and why they exist

goal, env_snapshot, depth: define what subproblem this node represents.
action_history: path of decomposition signatures taken to reach this node.
executed_steps: human-readable execution/decomposition trace.
generated_candidates: candidate decompositions already generated for this goal node.
candidate_generation_done: stops further candidate generation once exhausted or max count reached.
succeed_subgoals, failed_subgoals: cumulative success/failure counters for scoring/reporting.
is_terminal: tells tree policy to stop descending/expanding this node.
trajectory: flat primitive action trace for this node.
mcts_attempts: per-simulation records used for exported diagnostics.
final_success_trajectory: structured subgoal-level primitive traces after committed execution.



collect_llm_with_hmt