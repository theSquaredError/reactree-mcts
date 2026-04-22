try:
    from src.visualization.mcts_visualizer import update_visualizer_node, update_visualizer_task
except Exception:
    def update_visualizer_node(*args, **kwargs):
        return None

    def update_visualizer_task(*args, **kwargs):
        return None
import os
import sys
import math
import json
from collections import deque
from graphviz import Digraph
from .state import State
from .node import Node
import logging
import random
import numpy as np
from src.llm.prompt import SYSTEM_PROMPT, JUDGE_PROMPT

class MCTSAlgorithm:
    def __init__(self, computation_budget, max_sim_round_number, LLM, play_round=6, seed=42, gamma = 0.95):
        self.computation_budget = computation_budget
        self.max_sim_round_number = max_sim_round_number
        self.play_round = play_round
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.LLM = LLM
        self.current_round = 0
        self.seed = seed
        self.setup_seed(seed)

        self.gamma = gamma # disocunt factor
        
    def setup_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def tree_policy(self, node):
        while not node.get_state().is_terminal():
            if node.is_fully_expanded():
                node = self.best_child(node, True)
                update_visualizer_node(node, "select", self.current_round)
            else:
                expanded_node = self.expand(node)
                update_visualizer_node(expanded_node, "expand", self.current_round)
                return expanded_node
        return node

    def default_policy(self, node):  # simulate
        current_state = node.get_state().clone()
        env = current_state.env

        simulated_nodes = []
        parent_node = node

        sim_count = 1
        while not current_state.is_terminal():
            action, action_prob = self.llm_inference(current_state, current_state.admissible_commands)
            sim_count = sim_count*self.gamma*action_prob
            obs, reward, done, info = env.step(action)
            
            # 更新状态信息
            current_state.obs = obs[0]
            current_state.score = info['goal_condition_success_rate'][0] if 'goal_condition_success_rate' in info else 0.0
            current_state.done = done[0]
            current_state.current_round_index += 1
            current_state.admissible_commands = info["admissible_commands"][0]
            current_state.action_history.append(action)
            current_state.action_prob_history.append(action_prob)
            current_state.obs_list.append(current_state.obs)
            current_state.admissible_commands_list.append(current_state.admissible_commands)

            # 创建模拟节点并构建父子关系
            simulated_node = Node()
            simulated_node.set_state(current_state)
            simulated_node.set_parent(parent_node)
            parent_node.add_child(simulated_node)
            try:
                update_visualizer_node(simulated_node, "simulate", self.current_round)
            except:
                self.logger.info(f"visualizer error")
            parent_node = simulated_node
            save_simulated_node = Node()
            save_simulated_node.set_state(current_state.clone_no_step())
            # 暂存图片
            simulated_nodes.append(save_simulated_node)
            env.save_frame(current_state.image_path)
            current_state.image_path = current_state.get_image_path(current_state.image_dir)

        current_state.env.env.close()
        final_state_reward = current_state.score

        original_parent = node
        for simulated_node in simulated_nodes:
            parent = simulated_node.get_parent()
            if parent:
                parent.remove_child(simulated_node)
            simulated_node.set_parent(None)
            update_visualizer_node(simulated_node, "delete", self.current_round)

        # 清除原始父节点的所有子节点
        original_parent.children = []

        # # 如果任务执行成功，保留；否则删除
        if not parent_node.get_state().done:
            return final_state_reward*sim_count, original_parent
        else:
            # 遍历save_simulated_node
            simulated_nodes[0].set_parent(node)
            node.add_child(simulated_nodes[0])
            update_visualizer_node(simulated_nodes[0], "expand", self.current_round)

            for i in range(len(simulated_nodes) - 1):
                parent = simulated_nodes[i]
                child = simulated_nodes[i + 1]
                
                # 设置父子关系
                parent.add_child(child)
                child.set_parent(parent)  
                update_visualizer_node(child, "expand", self.current_round)

            # 否则不discount，在back_up里discount
            return final_state_reward, simulated_nodes[-1]
    
    def expand(self, node):
        tried_paths = node.get_all_child()

        action,action_prob = self.expand_action(node.get_state(), tried_paths)
        
        new_state = node.get_state().clone()
        obs, reward, done, info = new_state.env.step(action)
        
        # 保存当前帧
        new_state.image_path = new_state.get_image_path(new_state.image_dir)
        new_state.env.save_frame(new_state.image_path)
        
        # 更新新状态的信息
        new_state.obs = obs[0]
        new_state.score = info['goal_condition_success_rate'][0] if 'goal_condition_success_rate' in info else 0.0
        new_state.done = done[0]
        new_state.current_round_index += 1
        new_state.admissible_commands = info["admissible_commands"][0]
        new_state.action_history.append(action)
        new_state.action_prob_history.append(action_prob)
        new_state.obs_list.append(new_state.obs)
        new_state.admissible_commands_list.append(new_state.admissible_commands)

        sub_node = Node()
        sub_node.set_state(new_state)
        
        node.add_child(sub_node)
        return sub_node

    def expand_action(self, state, tried_paths):
        available_commands = list(state.admissible_commands)
        while True:
            action, action_prob = self.llm_inference(state, available_commands)
            # action = random.choice(available_commands)
            if state.action_history + [action] not in [item[:len(state.action_history)+1] for item in tried_paths]:
                break
            if action in available_commands:
                available_commands.remove(action)
        return action,action_prob

    def prompt_template(self, state, available_commands):
        # 构建提示信息
        system = SYSTEM_PROMPT
        messages = []
        if "Welcome to TextWorld, ALFRED!" in state.obs:
            messages.append({"role": "user", "content": f"Init environment: {state.obs}\n The candidate actions are {available_commands}"})
        else:
            for i in range(len(state.obs_list)):
                if i == 0:
                    messages.append({"role": "user", "content": f"Init environment: {state.obs_list[i]}\n"})
                    messages.append({"role": "assistant", "content": f"{state.action_history[i]}"})
                    continue
                if i == len(state.obs_list) - 1:
                    messages.append({"role": "user", "content": f"Current observation: {state.obs_list[i]}\nThe candidate actions are {available_commands}"})
                else:
                    messages.append({"role": "user", "content": f"Current observation: {state.obs_list[i]}"})
                    messages.append({"role": "assistant", "content": f"{state.action_history[i]}"})
        return system, messages
    
    def llm_inference(self, state, available_commands):
        action = ""
        action_candidates = {}
        llm_count = 0
        while len(action_candidates)==0:
            system, messages = self.prompt_template(state, available_commands)
            messages = [{"role": "system", "content": system}] + messages
            self.logger.info(f'[LLM] messages passed to model: {str(messages)[:200]}\n')
            action_candidates = self.LLM.chat(messages)
            
            self.logger.info(f"[PRE-FILTER] action_candidates: {action_candidates}")
            self.logger.debug(f"[AVAILABLE] available_commands: {available_commands}")
            self.logger.info(f"action_candidates: {action_candidates}, llm_count: {llm_count} ")
            action_candidates = {key: value for key, value in action_candidates.items() if key in available_commands}
            # --- DIAGNOSTIC LOGGING ---
            self.logger.info(f"[POST-FILTER] action_candidates: {action_candidates}")
            # -------------------------

            llm_count += 1
            if llm_count >=5:
                action = random.choice(available_commands)
                self.logger.info(f"random select {action}")
                action_prob = 0.9
                return action,action_prob
        action = list(action_candidates.keys())[0]
        action_prob = action_candidates[action]
        return action, action_prob

    def llm_as_judge(self, node, available_commands, state):
        # llm判断目前推理链是否错误，若错误则终止simulation
        messages = [{"role": "system", "content": JUDGE_PROMPT}]
        user_content = ''
        for i in range(len(node.get_state().obs_list)):
            if i == 0:
                user_content += f"Init environment: {node.get_state().obs_list[i]}\n"
                user_content += f"Action {i} is {node.get_state().action_history[i]}. "
                continue
            if i == len(node.get_state().obs_list) - 1:
                user_content += f"Action {i-1}'s  observation: {node.get_state().obs_list[i]}\n Decide whether the action chain is correct."
            else:
                user_content += f"Action {i-1}'s  observation: {state.obs_list[i]}\n Decide the next action, The candidate actions are {available_commands}. Your response must be exactly one action name chosen strictly from provide candidate actions.Do not provide any explanations."
        messages.append({"role": "user", "content": user_content})
        judge = self.LLM.chat(messages)
        self.logger.info(f"judge is :{judge}")

        if 'True' in judge and judge['True']>0.5:
            return True # 无错，不终止
        else:
            return False # 有错，终止
    
    def get_root(self, node):
        while node.parent:
            node = node.parent
            return node

    def best_child(self, node, is_exploration):
        best_score = -sys.maxsize
        best_sub_node = None
        node_candidates = []
        if is_exploration:
            node_candidates += node.get_children()
            for sub_node in node_candidates:
                if self.llm_as_judge(sub_node):
                    return sub_node
        else:
            node_candidates += node.get_children()
            # 获取同一层级的children
            root = self.get_root(node)
            queue = deque([root])  # (node, level)
            target_level = len(node.state.action_history)

            while queue:
                t_node = queue.popleft()
                
                # 如果已经超过目标层级，可以提前结束
                if t_node is None:
                    break

                if len(t_node.state.action_history) > target_level:
                    break

                # 如果当前正在遍历目标层级
                if len(t_node.state.action_history) == target_level:
                    if t_node != node:
                        node_candidates.extend(t_node.get_children())

                # 将子节点加入队列
                for child in t_node.children:
                    queue.append((child))


        for sub_node in node_candidates:
            left = sub_node.quality_value / sub_node.visit_count
            right = 2.0 * math.log(node.visit_count) / sub_node.visit_count
            action_prob = sub_node.state.action_prob_history[-1]
            C = 1.0 / math.sqrt(2.0) if is_exploration else 0.0
            score = left  +  C * math.sqrt(right)*action_prob

            if score > best_score:
                best_sub_node = sub_node
                best_score = score

        return best_sub_node

    def backup(self, node, reward):
        discount_factor = 1
        # updata parent 
        while node is not None:
            node.increment_visit_count()
            node.update_quality_value(reward*discount_factor)
            if node.get_parent() is not None:
                discount_factor = discount_factor*node.state.action_prob_history[-1]*self.gamma
            node = node.get_parent()
            
    def monte_carlo_tree_search(self, node):
        all_expand_nodes = []
        for i in range(self.computation_budget):
            self.current_round += 1
            expand_node = self.tree_policy(node)
            all_expand_nodes.append(expand_node)
            reward, leaf_node = self.default_policy(expand_node)
            self.backup(leaf_node, reward)

        # best next node从该层中所有的节点中选，而不是只包含选中的node的子node
        best_next_node = self.best_child(node, False)
        try:
            update_visualizer_node(best_next_node, "best", self.current_round)
        except:
            self.logger.info(f"visualizer error")
        return best_next_node, all_expand_nodes

    def execute_mcts(self, init_node, output_path):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        current_node = init_node
        
        # 在这里添加任务信息的传递
        task = current_node.get_state().task
        update_visualizer_task(task)

        for i in range(self.play_round):
            self.current_round = 0
            self.logger.info(f"Play round: {i + 1}")
            update_visualizer_node(current_node, "start_round", self.current_round)

            best_node, all_expand_nodes = self.monte_carlo_tree_search(current_node)
            action = best_node.get_state().action_history[-1]
            self.logger.info(f"Play round: {i + 1}, Action: {action}")
            
            done = best_node.get_state().done
            
            if done:
                self.logger.info("Goal achieved!")
                update_visualizer_node(best_node, "best", self.current_round)
                break
            else:
                # 只关闭环境，不删除节点
                for node in all_expand_nodes:
                    if node.get_state() != best_node.get_state():
                        if not node.get_state().env.env.envs[0].env.response_queue.full():
                            # 不能能重复关闭
                            node.get_state().env.env.close()
                
                # 关闭当前节点的环境
                if not current_node.get_state().env.env.envs[0].env.response_queue.full():
                    current_node.get_state().env.env.close()

            current_node = best_node  # 更新当前节点为最佳节点
            update_visualizer_node(current_node, "best", self.current_round)
            update_visualizer_node(current_node, "end_round", self.current_round)

            if i == self.play_round - 1:
                self.logger.info("Max rounds reached without achieving goal.")

        # 在这里，我们可以收集所有的action chains
        action_chains = self.collect_action_chains(init_node)
        
        # 关闭最后一个节点的环境
        if not current_node.get_state().env.env.envs[0].env.response_queue.full():
            current_node.get_state().env.env.close()
        
        # 保存和可视化MCTS树
        mcts_tree_path = output_path + "/mcts_tree.json"
        self.save_tree(init_node, mcts_tree_path)
        self.visualize_tree(mcts_tree_path, output_filename=output_path + "/mcts_tree.pdf")
        self.logger.info(f"Saved MCTS tree to {mcts_tree_path}")

        # 删除未被引用的图片
        image_paths = set()
        queue = deque([init_node])
        while queue:
            node = queue.popleft()
            image_paths.add(node.state.image_path)
            
            if node.children:  
                queue.extend(node.children)
        image_dir = node.state.image_dir
        for filename in os.listdir(image_dir):
            if filename.lower().endswith('.png'):
                file_path = os.path.join(image_dir, filename)
                if file_path not in image_paths:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
        return best_node

    def save_tree(self, root_node, filename):
        tree_dict = self.convert_np_bools(self.node_to_dict(root_node))
        with open(filename, "w") as f:
            json.dump(tree_dict, f, indent=4)

    def convert_np_bools(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_np_bools(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_np_bools(i) for i in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def node_to_dict(self, node):
        return {
            "task": node.state.task,
            "quality_value": node.quality_value,
            "visit_count": node.visit_count,
            "state": {
                "current_round_index": node.state.current_round_index,
                "action_history": node.state.action_history,
                "action_prob_history": node.state.action_prob_history,
                "score": node.state.score,
                "done": node.state.done,
                "obs": node.state.obs,
                "admissible_commands": node.state.admissible_commands,
                "image_path": node.state.image_path,
            },
            "children": [self.node_to_dict(child) for child in node.get_children()],
        }

    def add_node(self, g, node, parent_id=None):
        node_id = str(id(node))
        state = node.get('state', {})
        label = (
            f"task: {node.get('task', 'None')}\n"
            f"Q/N={node.get('quality_value', 0.0):.2f}/{node.get('visit_count', 0)}\n"
            f"State: Round {state.get('current_round_index', 0)}, "
            f"Actions: {state['action_history'][-1] if state['action_history'] else 'None'}\n"
            f"Score: {state.get('score', 0.0):.2f}, Done: {state.get('done', False)}\n"
            f"Obs: {state.get('obs', 'None')}\n"
            f"Admissible Commands: {state.get('admissible_commands', [])}\n"
            f"Image Path: {state.get('image_path', 'None')}"
        )
        g.node(node_id, label)
        if parent_id:
            g.edge(parent_id, node_id)
        for child in node.get("children", []):
            self.add_node(g, child, node_id)

    def add_node_simple(self, g, node, parent_id=None):
        node_id = str(id(node))
        action_history = node.get('state', {}).get('action_history', [])
        last_action = action_history[-1] if action_history else None

        label = (
            f"action: {last_action}"
        )
        g.node(node_id, label)
        if parent_id:
            g.edge(parent_id, node_id)
        for child in node.get("children", []):
            self.add_node_simple(g, child, node_id)

    def visualize_tree(self, tree_json_path, output_filename="tree_simple.pdf"):
        with open(tree_json_path, "r") as f:
            tree_json = json.load(f)

        g = Digraph(format="pdf")
        g.attr(rankdir='TB', nodesep='0.5', ranksep='5.0')
        self.add_node(g, tree_json)
        base_filename = output_filename.rsplit('.', 1)[0]
        g.render(base_filename, format='pdf')
        g.render(base_filename, format='png')

        # 简单的图
        g = Digraph(format="pdf")
        g.attr(rankdir='TB', nodesep='0.5', ranksep='5.0')
        self.add_node_simple(g, tree_json)
        base_filename = output_filename.rsplit('.', 1)[0] + "_simple"
        g.render(base_filename, format='pdf')
        g.render(base_filename, format='png')

    def collect_action_chains(self, root_node):
        action_chains = []
        
        def dfs(node, current_chain):
            if not node.children:
                action_chains.append({
                    'actions': current_chain + [node.get_state().action_history[-1]],
                    'score': node.quality_value,
                    'done': node.get_state().done
                })
            else:
                for child in node.children:
                    dfs(child, current_chain + [node.get_state().action_history[-1] if node.get_state().action_history else []])
        
        dfs(root_node, [])
        return action_chains
