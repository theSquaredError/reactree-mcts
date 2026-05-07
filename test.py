import os, importlib, logging
from datetime import datetime
from dotenv import load_dotenv
import argparse
import glob
import random
from tqdm import tqdm
from os.path import join as pjoin
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,                                # DEBUG shows LLM input/output
    format='%(asctime)s [%(levelname)s] %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
    ]
)
# Silence noisy third-party loggers
for _lib in ('transformers', 'torch', 'urllib3', 'filelock', 'huggingface_hub'):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from src.mcts_src.environment.alfworld_env import AlfWorldEnv
from src.rm.rm_llm_agent import LlmAgent

# llm = LlmAgent(hf_token=os.getenv('hf_token'))

# select a task
def select_problems_fix(num_problems):
   
    train_path = '/home/azureuser/vikas/Embodied-Agent-Planning/mcts_datagen/data/json_2.1.1/valid_unseen'
    first_level_dirs = next(os.walk(train_path))[1]

    results = []
    for dir_name in first_level_dirs:
        base_path = pjoin(train_path, dir_name)
        # 获取这个目录下的所有子文件夹
        sub_dirs = next(os.walk(base_path))[1]
        if sub_dirs:  # 如果有子文件夹
            # 获取第一个子文件夹的完整路径
            first_sub_dir = pjoin(base_path, sub_dirs[0])
            results.append(first_sub_dir)
    
    valid_problems = []
    for dir_path in results:
        pddl_files = glob.glob(pjoin(dir_path, "initial_state.pddl"))
        if pddl_files and "movable_recep" not in pddl_files[0]:
            valid_problems.extend(pddl_files)
    # valid_problems = [item for item in valid_problems if "pick_two_obj_and_place" in item]
    return valid_problems[0]

task = select_problems_fix(1)
alf_env = AlfWorldEnv(config_path = '/home/azureuser/vikas/ReAcTree/conf/base_config.yaml', task_file=os.path.dirname(task))

llm = LlmAgent(hf_token=os.getenv('hf_token'))

import re

# Pull obs and info stored on the env object after reset
init_obs = alf_env.init_obs
init_info = alf_env.init_info  # returned by wait_and_get_info()[2]

# Extract "Your task is to: ..." from the initial observation
task_match = re.search(r'Your task is to: (.+)', init_obs)
task_desc = task_match.group(1).rstrip('.').strip() if task_match else "complete the task"

# admissible_commands is batched: info['admissible_commands'][batch_idx]
try:
    admissible_commands = init_info['admissible_commands'][0]
except (KeyError, TypeError, IndexError):
    admissible_commands = []

logger.info("Task          : %s", task_desc)
logger.info("Init obs      : %s", init_obs[:300])
logger.info("Available cmds: %s ...", admissible_commands[:5])

from src.rm.alfworld_reactree import AlfWorldAgentNode

task_type = task.split('/')[-3].split('-')[0]  # e.g. 'pick_and_place_simple'

# Root node represents the full task at depth 0
root = AlfWorldAgentNode(
    {'nl_inst': task_desc, 'task_type': task_type},
    depth=0,
    max_steps=20,
    max_decisions=20,
    max_depth=4,
    llm_agent=llm,
    env=alf_env,
)

# Calls the planner loop, expanding into ControlFlowNode + child ALFWorld nodes as needed
root.run()


# max_steps: only increments for env action: Act:
# max_decisions: for total llm decision: increments for act:, think:, expand
# depth = 