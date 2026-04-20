from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json
import numpy as np
import logging

app = Flask(__name__)
socketio = SocketIO(app)

global_tree_data = None
global_current_round = 0
global_task = None
#TODO: 服务器跑数据需要对本地做端口映射
@app.route('/')
def index():
    return render_template('mcts_visualizer.html')

@socketio.on('connect')
def handle_connect():
    socketio.emit('connection_established', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    pass

def update_visualizer_task(task):
    global global_task
    global_task = task
    socketio.emit('task_update', json.dumps({"task": task}))
def convert_np_bools(obj):
    if isinstance(obj, dict):
        return {k: convert_np_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_bools(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
def update_visualizer_node(node, action, current_round):
    global global_tree_data, global_current_round, global_task
    
    global_current_round = current_round
    
    if action == "delete":
        remove_node_from_tree(global_tree_data, str(id(node)))
    else:
        node_dict = convert_np_bools(node_to_dict(node))
        node_dict['class'] = action
        
        if node.get_parent() is None:
            global_tree_data = node_dict
            global_task = node.state.task  # 保存任务信息
        else:
            parent_dict = find_parent_in_tree(global_tree_data, node.get_parent())
            if parent_dict:
                if 'children' not in parent_dict:
                    parent_dict['children'] = []
                existing_node = next((child for child in parent_dict['children'] if child['id'] == node_dict['id']), None)
                if existing_node:
                    existing_node.update(node_dict)
                else:
                    parent_dict['children'].append(node_dict)
    
    socketio.emit('tree_update', json.dumps({
        "tree": global_tree_data,
        "current_round": global_current_round,
        "task": global_task  # 在每次更新时都发送任务信息
    }))

def remove_node_from_tree(tree, node_id):
    if tree['id'] == node_id:
        return None
    if 'children' in tree:
        tree['children'] = [child for child in tree['children'] if child['id'] != node_id]
        for child in tree['children']:
            remove_node_from_tree(child, node_id)
    return tree

def find_parent_in_tree(tree, parent_node):
    if tree['id'] == str(id(parent_node)):
        return tree
    for child in tree.get('children', []):
        result = find_parent_in_tree(child, parent_node)
        if result:
            return result
    return None

def node_to_dict(node):
    return {
        "id": str(id(node)),
        "quality_value": node.quality_value,
        "visit_count": node.visit_count,
        "state": {
            "current_round_index": node.state.current_round_index,
            "action_history": node.state.action_history,
            "score": node.state.score,
            "done": node.state.done,
            "obs": node.state.obs,
            "admissible_commands": node.state.admissible_commands
        },
        "children": [],
        "class": ""
    }

def update_visualizer_tree(tree_data, current_round, play_round):
    global global_tree_data, global_current_round
    
    global_tree_data = tree_data
    global_current_round = current_round
    
    socketio.emit('tree_update', json.dumps({
        "tree": global_tree_data,
        "current_round": global_current_round,
        "play_round": play_round
    }))

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
