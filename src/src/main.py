import argparse
import glob
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from os.path import join as pjoin

from src.environment.alfworld_env import AlfWorldEnv
from mcts.algorithm import MCTSAlgorithm
from mcts.state import State
from mcts.node import Node
from src.visualization.mcts_visualizer import app, socketio, update_visualizer_node, update_visualizer_tree, update_visualizer_task
# from src.llm.llm import LlaMaChatModel, GPT4Engine, HuggingfaceChatModel
from src.llm.llm import HuggingfaceChatModel
import logging
import sys
from datetime import datetime
from threading import Thread
import threading
from alfworld.info import ALFWORLD_DATA


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mcts_log_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Play the abstract text version of an ALFRED environment."
    )
    parser.add_argument(
        "problem",
        nargs="?",
        default=None,
        help="Path to a folder containing PDDL and traj_data files. Default: pick one at random found in {ALFWORLD_DATA}",
    )
    parser.add_argument(
        "--controller",
        default="oracle",
        choices=["oracle", "oracle_astar", "mrcnn", "mrcnn_astar"],
    )
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--load_receps", action="store_true")
    parser.add_argument(
        "--reward_config",
        type=str,
        default=pjoin("./alfworld/agents/config", "rewards.json"),
    )
    parser.add_argument(
        "--computation_budget", # 每个节点的子节点个数
        type=int,
        default=3,
        help="The maximum number of computations allowed for MCTS.",
    )
    parser.add_argument(
        "--max_sim_round",
        type=int,
        default=15,
        help="The maximum number of simulation rounds allowed for MCTS.",
    )
    parser.add_argument(
        "--play_round", # 树的深度
        type=int,
        default=15,
        help="The number of play rounds allowed for MCTS.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Path to save output files.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/base_config.yaml",
        help="path to env config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--llm_template",
        type=str,
        default="llama-3.1",
        choices=["gpt-4", "llama-3.1"],
        help="LLM model to use.",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=2,
        help="Number of problems to run. Default: 100",
    )
    return parser.parse_args()

def select_problems(num_problems):
    all_problems = glob.glob(
        pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True
    )
    valid_problems = [p for p in all_problems if "movable_recep" not in p]
    random.shuffle(valid_problems)
    valid_problems = [item for item in valid_problems if "pick_clean" in item]
    return valid_problems[34:min(num_problems, len(valid_problems))]

def select_problems_fix(num_problems):
    # 固定的train的sample
    train_path = '/home/azureuser/vikas/Embodied-Agent-Planning/mcts_datagen/data/json_2.1.1/train'
    first_level_dirs = next(os.walk(train_path))[1]

    # 然后对于每个子文件夹，获取它的第一个子文件夹
    results = []
    for dir_name in first_level_dirs:
        base_path = pjoin(train_path, dir_name)
        # 获取这个目录下的所有子文件夹
        sub_dirs = next(os.walk(base_path))[1]
        if sub_dirs:  # 如果有子文件夹
            # 获取第一个子文件夹的完整路径
            first_sub_dir = pjoin(base_path, sub_dirs[0])
            results.append(first_sub_dir)

    # 如果您还需要在这些第一个子文件夹中查找 initial_state.pddl
    valid_problems = []
    for dir_path in results:
        pddl_files = glob.glob(pjoin(dir_path, "initial_state.pddl"))
        if pddl_files and "movable_recep" not in pddl_files[0]:
            valid_problems.extend(pddl_files)
    # valid_problems = [item for item in valid_problems if "pick_two_obj_and_place" in item]
    return valid_problems[200:]

def run_mcts(problem, args, logger, task_index, LLM):
    try:
        setup_seed(args.seed)
        
        args.problem = os.path.dirname(problem)

        logger.info(f"Selected task {task_index + 1}/{args.num_problems}: {args.problem}")

        task_name = args.problem.split("/")[-2]
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tree_data_path = os.path.join(args.output_path, task_name, current_time)
        os.makedirs(tree_data_path, exist_ok=True)
        image_dir = os.path.join(tree_data_path, "images")
        os.makedirs(image_dir, exist_ok=True)
        
        alf_env = AlfWorldEnv(args.config_path, task_file=args.problem)

        mcts = MCTSAlgorithm(
            computation_budget=args.computation_budget,
            max_sim_round_number=args.max_sim_round,
            play_round=args.play_round,
            LLM=LLM,
            seed=args.seed
        )

        init_state = State(alf_env, max_sim_round_number=args.max_sim_round, image_dir=image_dir)
        init_node = Node()
        init_node.set_state(init_state)
        init_state.env.save_frame(init_state.image_path)
        
        mcts.execute_mcts(init_node, tree_data_path)

        logger.info(f"MCTS execution completed for task {task_index + 1}: {args.problem}")
        
    except Exception as e:
        logger.error(f"An error occurred in task {task_index + 1}: {e}", exc_info=True)

def main():
    args = parse_arguments()
    setup_seed(args.seed)
    
    if args.llm_template == "gpt-4":
        LLM = GPT4Engine()
    elif args.llm_template == "llama-3.1":
        # LLM = LlaMaChatModel()
        LLM= HuggingfaceChatModel()
    else:
        raise ValueError(f"Unknown LLM template: {args.llm_template}")
    
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
    logger = setup_logging(log_dir)

    initial_data = {"message": "MCTS visualization starting"}
    update_visualizer_tree(initial_data, 0, 0)

    problems = select_problems_fix(args.num_problems)
    logger.info(problems[:10])
    
    for i, problem in tqdm(enumerate(problems), total=len(problems), desc="task number"):
        mcts_thread = threading.Thread(target=run_mcts, args=(problem, args, logger, i, LLM))
        mcts_thread.start()
        mcts_thread.join()  # Wait for the current task to finish before starting the next one

    logger.info(f"Completed {len(problems)} tasks.")

    logger.info("Starting Flask application")
    socketio.run(app, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
    