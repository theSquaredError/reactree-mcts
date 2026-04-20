import json
from graphviz import Digraph


def convert_label_to_str(label):
    action = label["action"]
    args = label.get("args", [])

    if action == "GotoLocation":
        return f"go to {args[0]}"
    elif action == "PickupObject":
        return f"take {args[0]}"
    elif action == "PutObject":
        return f"put {args[0]}"
    elif action == "OpenObject":
        return f"open {args[0]}"
    elif action == "CloseObject":
        return f"close {args[0]}"
    elif action == "ToggleObject":
        return f"use {args[0]}"
    elif action == "HeatObject":
        return f"heat {args[0]}"
    elif action == "CoolObject":
        return f"cool {args[0]}"
    elif action == "CleanObject":
        return f"clean {args[0]}"
    elif action == "SliceObject":
        return f"slice {args[0]}"
    elif action == "ShowInventory":
        return "inventory"
    elif action == "ExamineObject":
        return f"examine {args[0]}"
    elif action == "LookAround":
        return "look"
    else:
        return "pass"


def find_longest_ordered_subsequence(label_action, action_history):
    label_index = 0
    history_index = 0
    label_length = len(label_action)
    history_length = len(action_history)

    current_subsequence = []

    while history_index < history_length:
        if label_index >= label_length:
            break

        if label_action[label_index] in action_history[history_index]:
            current_subsequence.append(action_history[history_index])
            label_index += 1

        history_index += 1

    if len(current_subsequence) == len(label_action):
        # If the current subsequence is the same length as the label, return a high score
        # This is a temporary reward plan.
        return 100
    else:
        return len(current_subsequence)


def save_tree(root_node, filename):
    tree_dict = node_to_dict(root_node)
    with open(filename, "w") as f:
        json.dump(tree_dict, f, indent=4)


def node_to_dict(node):
    return {
        "quality_value": node.quality_value,
        "visit_times": node.visit_times,
        "state": {
            "current_round_index": node.state.current_round_index,
            "action_history": node.state.action_history,
        },
        "children": [node_to_dict(child) for child in node.get_children()],
    }


def add_node(g, node, parent_id=None):
    node_id = str(id(node))
    label = (
        f"Q/N={node['quality_value']:.2f}/{node['visit_times']}\n"
        f"State: Round {node['state']['current_round_index']}, "
        f"Actions: {node['state']['action_history'][-1] if node['state']['action_history'] else 'None'}"
    )
    g.node(node_id, label)
    if parent_id:
        g.edge(parent_id, node_id)
    for child in node["children"]:
        add_node(g, child, node_id)


def visualize_tree(tree_json, output_filename="tree"):
    g = Digraph(format="pdf")
    g.attr(rankdir='TB', nodesep='0.5', ranksep='5.0')  # 设置布局为从上到下（TB: top to bottom）  
    add_node(g, tree_json)
    g.render(output_filename)

path = "/home/zy/code/alfworld/output_test"
import os
with open("output_test/mcts_tree.json", "r") as f:
    tree_json = json.load(f)
visualize_tree(tree_json, output_filename=os.path.join(path, "mcts_tree"))  
