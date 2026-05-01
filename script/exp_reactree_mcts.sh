#!/usr/bin/env bash
# Run ReactreeMCTS on ALFWorld.
#
# ReAcTree Think / Act / Expand loop unchanged.
# Every primitive Act step hands off to MCTS which searches for the best
# sequence of admissible actions to complete the subgoal.
# Reward signal: info['goal_condition_success_rate'] from ALFWorld.
#
# Usage:
#   bash script/exp_reactree_mcts.sh              # evaluate valid_seen
#   bash script/exp_reactree_mcts.sh collect_llm  # single-task smoke test
#   bash script/exp_reactree_mcts.sh evaluate valid_unseen

set -e
cd "$(dirname "$0")/.."

EXP_TYPE="${1:-evaluate}"
EVAL_SET="${2:-valid_seen}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
MCTS_BUDGET="${MCTS_BUDGET:-10}"
MCTS_MAX_DEPTH="${MCTS_MAX_DEPTH:-12}"
EVAL_PORTION="${EVAL_PORTION:-100}"

echo "=== ReactreeMCTS (ALFWorld) ==="
echo "  exp_type       : $EXP_TYPE"
echo "  eval_set       : $EVAL_SET"
echo "  model          : $MODEL"
echo "  mcts_budget    : $MCTS_BUDGET"
echo "  mcts_max_depth : $MCTS_MAX_DEPTH"
echo "  eval_portion   : $EVAL_PORTION%"
echo ""

python src/reactree_mcts_runner.py \
  --config-name=alfworld_reactree_mcts \
  exp_type="$EXP_TYPE" \
  dataset.eval_set="$EVAL_SET" \
  llm_agent.model_name="$MODEL" \
  llm_agent.max_decisions=100 \
  prompt.sys_prompt_root_dir=resource/alfworld/sys_prompt \
  reactree_mcts.mcts_budget="$MCTS_BUDGET" \
  reactree_mcts.mcts_max_depth="$MCTS_MAX_DEPTH" \
  alfworld.eval_portion_in_percent="$EVAL_PORTION"
