import gymnasium as gym
from typing import Literal
from rl_envs.environment import Environment
from rl_envs.task_splits import DeliverySplit, GoalSplit
import numpy as np

EnvType = Literal["hacky-variable", "genuine-fixed", "genuine-variable", "pickup-and-deliver"]
VALID_ACTIONS = ["north", "east", "south", "west", "claim", "pickup"]

class GridWorld(gym.Env):
    def __init__(self, board_size: int, start_pos: list[int], 
                     fixed_goal_pos: list[int], max_actions: int, 
                     env_type: EnvType, goal_split: GoalSplit | None = None,
                     delivery_split: DeliverySplit | None = None):
        super().__init__()
        self.env = Environment(board_size, start_pos, fixed_goal_pos, max_actions, env_type, self.np_random, goal_split, delivery_split)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(-1, 1, (10,))
        self.assisted_start_probability = 0.0
        self.assisted_start_max_distance = 0

    @property
    def env_type(self) -> EnvType:
        return self.env.env_type

    @property
    def goal_split(self) -> GoalSplit | None:
        return self.env.goal_split

    @property
    def delivery_split(self) -> DeliverySplit | None:
        return self.env.delivery_split

    def set_delivery_assistance(self, probability: float, max_distance: int):
        """Set the chance and maximum destination distance for assisted resets."""
        if self.env_type != 'pickup-and-deliver':
            raise ValueError('delivery assistance requires a delivery environment')
        if not 0 <= probability <= 1:
            raise ValueError('assistance probability must be between 0 and 1')
        if not isinstance(max_distance, int) or isinstance(max_distance, bool) or max_distance < 0:
            raise ValueError('assistance distance must be a nonnegative integer')
        self.assisted_start_probability = probability
        self.assisted_start_max_distance = max_distance

    def _choose_assisted_start(self):
        goal_x, goal_y = self.env.goal_pos
        positions = [(x, y) for x in range(self.env.board_size) for y in range(self.env.board_size)
                     if abs(x - goal_x) + abs(y - goal_y) <= self.assisted_start_max_distance]
        return positions[int(self.np_random.integers(len(positions)))]

    def reset(self, seed=None, options=None):
        # Start a new episode using Gymnasium's random generator and any selected task instance.
        super().reset(seed=seed)
        self.env.reset(self.np_random, goal_pos=(options or {}).get('goal_pos'),
                       delivery_pair=(options or {}).get('delivery_pair'))
        # During delivery training, optionally start near the destination while carrying the package.
        assisted = (self.assisted_start_probability > 0
                    and self.np_random.random() < self.assisted_start_probability)
        if assisted:
            self.env._set_assisted_delivery_start(self._choose_assisted_start())
        # Convert the environment's observation to the fixed array expected by the policy.
        obs = self.env.agent_observation()
        obs_np, _ = self._map_obs_to_np(obs)
        info = {'assisted_start': assisted} if self.env_type == 'pickup-and-deliver' else {}
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
        # Translate the policy's action and return the result in Gymnasium's format.
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
         if env.delivery_split is not None:
             settings['delivery_split'] = {'seen': env.delivery_split.seen, 'withheld': env.delivery_split.withheld}
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
