import copy
import os

class State:
    def __init__(self, env, max_sim_round_number=20, image_dir=None):
        self.env = env
        self.current_round_index = 0
        self.action_history = []
        self.action_prob_history = []
        self.max_sim_round_number = max_sim_round_number
        self.obs, self.info = env.init_obs, env.init_info
        self.obs_list = [self.obs]
        self.score = 0
        self.done = False
        self.task = self.extract_task(self.obs)
        self.admissible_commands = [item for item in self.info["admissible_commands"][0] if item != "look"]
        self.admissible_commands_list = [self.admissible_commands]
        self.image_dir = image_dir
        self.image_path = self.get_image_path(image_dir)

    @staticmethod
    def extract_task(obs):
        return obs.split("\n")[-1].split(":")[-1].strip()
    
    def get_image_path(self, image_dir):
        existing_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        frame_count = len(existing_files) + 1
        frame_filename = f"{frame_count:04d}.png"
        frame_path = os.path.join(image_dir, frame_filename)
        return frame_path

    def clone(self):
        new_env = self.env.clone()
        new_state = State(new_env, self.max_sim_round_number, os.path.dirname(self.image_path))
        new_state.current_round_index = self.current_round_index
        new_state.action_history = copy.deepcopy(self.action_history)
        new_state.action_prob_history = copy.deepcopy(self.action_prob_history)
        new_state.obs = self.obs
        new_state.obs_list = copy.deepcopy(self.obs_list)
        new_state.score = self.score
        new_state.done = self.done
        new_state.task = self.task
        new_state.admissible_commands = copy.deepcopy(self.admissible_commands)
        new_state.admissible_commands_list = copy.deepcopy(self.admissible_commands_list)
        new_state.env.step("look")
        
        for action in new_state.action_history:
            new_state.env.step(action)
        
        return new_state
    
    def clone_no_step(self):
        # 没有env的step的clone
        new_state = State(self.env, self.max_sim_round_number, os.path.dirname(self.image_path))
        new_state.current_round_index = self.current_round_index
        new_state.action_history = copy.deepcopy(self.action_history)
        new_state.action_prob_history = copy.deepcopy(self.action_prob_history)
        new_state.obs = self.obs
        new_state.info = self.info
        new_state.obs_list = copy.deepcopy(self.obs_list)
        new_state.score = self.score
        new_state.done = self.done
        new_state.task = self.task
        new_state.admissible_commands = copy.deepcopy(self.admissible_commands)
        new_state.admissible_commands_list = copy.deepcopy(self.admissible_commands_list)

        return new_state


    def is_terminal(self):
        return self.current_round_index >= self.max_sim_round_number or self.done

    def get_available_actions(self):
        return [item for item in self.admissible_commands if item != "look"]

    def __repr__(self):
        return f"State(round={self.current_round_index}, actions={self.action_history}, score={self.score}, done={self.done})"

    def __eq__(self, other):
        if isinstance(other, State):
            return (
                self.current_round_index == other.current_round_index
                and self.action_history == other.action_history
            )
        return False

    def __hash__(self):
        return hash((self.current_round_index, tuple(self.action_history)))
