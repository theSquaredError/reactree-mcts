import os
import yaml
from alfworld.agents import environment
import cv2
import os
from PIL import Image

class AlfWorldEnv:
    def __init__(self, config_path, task_file=None):
        self.config_path = config_path
        with open(config_path) as reader:
            self.config = yaml.safe_load(reader)
        
        split = "eval_out_of_distribution"

        self.task_file = task_file
        self.traj_data_file = os.path.join(self.task_file, 'traj_data.json')

        env_class = getattr(environment, self.config["env"]["type"])
        self.env = env_class(self.config, train_eval=split)
        self.env = self.env.init_env(batch_size=1)
        self.action_history = []
        self.last_obs  = None
        self.last_info = None

        self.init_obs, self.init_info = self.reset()
        self.last_obs  = self.init_obs
        self.last_info = self.init_info

    def set_task(self):
        if os.path.exists(self.traj_data_file):
            self.env.envs[0].set_task(self.traj_data_file)
        else:
            raise FileNotFoundError(f"traj_data.json not found in {self.task_file}")

    def reset(self):
        obs = self.env.envs[0].reset(self.traj_data_file)
        self.action_history = []
        info = self.get_info()
        return obs, info

    def step(self, action):
        self.action_history.append(action)
        obs, scores, dones, infos = self.env.step([action])
        self.last_obs  = obs[0] if isinstance(obs, list) else obs
        self.last_info = infos
        return obs, scores, dones, infos

    def clone(self):
        new_env = AlfWorldEnv(self.config_path, self.task_file)
        # if self.task_file:
        #     new_env.set_task()
        # new_env.reset()
        
        # 重放动作以达到当前状态
        # for action in self.action_history:
        #     new_env.step([action])
        
        # new_env.action_history = copy.deepcopy(self.action_history)
        return new_env

    def get_info(self):
        return self.env.wait_and_get_info()[2]

    def save_frame(self, image_path):
        if image_path:
            images = self.env.get_frames()
            
            # pil_image = Image.fromarray(images[0])
            # pil_image.save(frame_path)
            cv2_image = cv2.cvtColor(images[0], cv2.COLOR_RGB2RGBA)
            cv2.imwrite(image_path, cv2_image)
            return image_path
        return None
