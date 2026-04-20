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
from src.llm.llm import LlaMaChatModel, GPT4Engine
import logging
import sys
from datetime import datetime
from threading import Thread
import threading
from alfworld.info import ALFWORLD_DATA

def main():
    LLM = GPT4Engine()
    mcts = MCTSAlgorithm(
    computation_budget=1,
    max_sim_round_number=1,
    play_round=1,
    LLM=LLM,
    seed=1)
    mcts.visualize_tree("/mnt/tangyehui/dataset/mcts_datasets/20241017_1032_v2/origin/output/look_at_obj_in_light-Newspaper-None-DeskLamp-225/20241016_215535/mcts_tree.json")



if __name__ == "__main__":
    train_path = '/mnt/tangyehui/code/mcts_datagen/data/json_2.1.1/train'
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
    print(valid_problems[:10])
    valid_problems = [item for item in valid_problems if "pick_clean" in item]
    print(valid_problems[:10])


    