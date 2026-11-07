import gymnasium as gym
from typing import Literal
from environment import Environment
from goal_split import GoalSplit
import numpy as np

EnvType = Literal["hacky-variable", "genuine-fixed", "genuine-variable", "pickup-and-deliver"]
VALID_ACTIONS = ["north", "east", "south", "west", "claim", "pickup"]


class GridWorld(gym.Env):
    def __init__(self, board_size: int, start_pos: list[int], 
                     fixed_goal_pos: list[int], max_actions: int, 
                     env_type: EnvType, goal_split: GoalSplit | None = None):
        super().__init__()
        self.env = Environment(board_size, start_pos, fixed_goal_pos, max_actions, env_type, self.np_random, goal_split)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(-1, 1, (10,))

    @property
    def env_type(self) -> EnvType:
        return self.env.env_type

    @property
    def goal_split(self) -> GoalSplit | None:
        return self.env.goal_split

    def reset(self, seed=None, options=None):
        #this is a standard method in gymnasium that basically 'starts a new episode'
        super().reset(seed=seed)
        self.env.reset(self.np_random, goal_pos=(options or {}).get('goal_pos'))
        obs = self.env.agent_observation()
        obs_np, _ = self._map_obs_to_np(obs)
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
        obs, _ = self._map_obs_to_np(obs)
        return obs, reward, terminated, truncated, info

    def agent_observation(self):
        obs_raw = self.env.agent_observation()
        obs, _ = self._map_obs_to_np(obs_raw)
        return obs

    def get_env_settings(self):
         env = self.env
         obs = env.agent_observation()
         settings = {
                'env_type': env.env_type,
                'task_mode': self._decode_task_mode(obs['task_mode']) ,
                'board_size': env.board_size,
                'start_pos': env.start_pos.copy(),
                'fixed_goal_pos': env.fixed_goal_pos.copy(),
                'max_actions': env.max_actions,
                'policy_observation_fields': self._map_obs_to_np(obs)[1],
            }
         if env.goal_split is not None:
             settings['goal_split'] = {'seen': env.goal_split.seen, 'withheld': env.goal_split.withheld}
         return settings

    def _map_obs_to_np(self, obs):
        arr = []
        fields =  ["agent_x", "agent_y", "goal_x", "goal_y", "package_x", "package_y", 
                   "carrying_package", "package_active", "task_mode", "remaining_time"]
        for field in fields:
            arr.append(obs[field])
        np_arr = np.array(arr, np.float32)
        return np_arr, fields

    def _map_int_to_action(self, n):
        if n not in range(0, 6):
            raise ValueError("Invalid action. 'n' must be between 0-5.")
        int_to_act_map = {0: "north", 1: "east", 2:"south", 
                          3: "west", 4: "claim", 5: "pickup"}
        return int_to_act_map[n]

    def _decode_task_mode(self, task_mode):
            decoding = {
                0 : "beacon",
                1 : "pickup-and-deliver" 
            }
            return decoding[task_mode]



#----------------------------------
def main():
    grid =  GridWorld(7, [3, 3], [3, 0], 30, "genuine-fixed")
    env = grid.env
    obs = env.agent_observation()
    print(f"Observation 1: {obs}\n-------")
    print(grid._map_obs_to_np(obs)[0])
    print(grid.action_space)
    action = grid.action_space.sample()
    print(action)
    print(grid._map_int_to_action(action))

if __name__=="__main__":
    main()
