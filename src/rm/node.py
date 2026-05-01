class Node:
    def __init__(self):
        self.parent = None
        self.children = []
        self.visit_count = 0
        self.quality_value = 0.0
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def increment_visit_count(self):
        self.visit_count += 1

    def update_quality_value(self, value):
        self.quality_value += value

    def is_fully_expanded(self):
        return len(self.children) >= len(self.state.get_available_actions())

    def add_child(self, sub_node):
        sub_node.parent = self
        self.children.append(sub_node)

    def get_all_child(self):
        all_paths = []

        def dfs(node: Node):
            if not node.children:  # Leaf node
                all_paths.append(node.get_state().action_history)
            else:
                for child in node.children:
                    dfs(child)

        dfs(self)
        return all_paths

    def set_parent(self, parent):
        self.parent = parent

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)

    def __repr__(self):
        return f"Node(Q/N={self.quality_value}/{self.visit_count}, state={self.state})"
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.state == other.state and 
                self.visit_count == other.visit_count and 
                self.quality_value == other.quality_value)
        return False
    

