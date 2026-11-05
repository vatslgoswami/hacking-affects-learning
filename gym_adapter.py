import gymnasium as gym
from typing import Literal
from environment import Environment
import numpy as np

EnvType = Literal["hacky-variable", "genuine-fixed", "genuine-variable", "pickup-and-deliver"]
VALID_ACTIONS = ["north", "east", "south", "west", "claim", "pickup"]


class GridWorld(gym.Env):
    def __init__(self, board_size: int, start_pos: list[int], 
                     fixed_goal_pos: list[int], max_actions: int, 
                     env_type: EnvType):
        super().__init__()
        self.env = Environment(board_size, start_pos, fixed_goal_pos, max_actions, env_type, self.np_random)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(-1, 1, (10,))

    def reset(self, seed=None, options=None):
        #this is a standard method in gymnasium that basically 'starts a new episode'
        super().reset(seed=seed)
        self.env.reset(self.np_random) 
        obs = self.env.agent_observation()
        obs_np = self._map_obs_to_np(obs)
        info = {}
        return obs_np, info

    def step(self, n_action:int):
        '''
        takes an action and then returns the 
        - observation, 
        - immed_reward, 
        - terminated: whether the episode has ended (max limit/task completed)
        - truncated: if we cutoff the agent's episode due to external reasons but we want it to use the value estimates in learning
        - info
        '''
        action = self._map_int_to_action(n_action)
        act_completed, reward, obs, terminated, truncated, info = self.env.agent_action(action)
        obs = self._map_obs_to_np(obs)
        return obs, reward, terminated, truncated, info

    def _map_obs_to_np(self, obs):
        arr = []
        fields =  ["agent_x", "agent_y", "goal_x", "goal_y", "package_x", "package_y", 
                   "carrying_package", "package_active", "task_mode", "remaining_time"]
        for field in fields:
            arr.append(obs[field])
        np_arr = np.array(arr, np.float32)
        return np_arr

    def _map_int_to_action(self, n):
        if n not in range(0, 6):
            raise ValueError("Invalid action. 'n' must be between 0-5.")
        int_to_act_map = {0: "north", 1: "east", 2:"south", 
                          3: "west", 4: "claim", 5: "pickup"}
        return int_to_act_map[n]



#----------------------------------
def main():
    grid =  GridWorld(7, [3, 3], [3, 0], 30, "genuine-fixed")
    env = grid.env
    obs = env.agent_observation()
    print(f"Observation 1: {obs}\n-------")
    print(grid._map_obs_to_np(obs))
    print(grid.action_space)
    action = grid.action_space.sample()
    print(action)
    print(grid._map_int_to_action(action))

if __name__=="__main__":
    main()
