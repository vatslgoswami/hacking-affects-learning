import random
from typing import Literal
from numpy.random import Generator, PCG64 

EnvType = Literal["hacky-variable", "genuine-fixed", "genuine-variable", "pickup-and-deliver"]
VALID_ACTIONS = ["north", "east", "south", "west", "claim", "pickup"]

class Environment:

    def __init__(self, board_size: int, start_pos: list[int], fixed_goal_pos: list[int], 
                 max_actions: int, env_type: EnvType, rng : Generator) -> None:
        self.board_size = board_size
        self.start_pos = start_pos
        self.fixed_goal_pos = fixed_goal_pos
        self.max_actions = max_actions
        self.env_type: EnvType = env_type

        #set up the environment
        self.reset(rng)

    def reset(self,rng):
        # update the current environment so it reflects a reset
        self.agent_pos= self.start_pos.copy()
        self.action_count = 0
        self.np_random = rng
        self.goal_pos = self._generate_random_coords(self.board_size)
        self.total_reward = 0
        self.episode_ended = False
        if self.env_type=="pickup-and-deliver": 
            self.package_pos=self._generate_random_coords(self.board_size)
        else: self.package_pos=[-1,-1]
        return self

    def agent_action(self, action):
        '''
        - act_completed : bool = whether the action was valid and succeeded or not
        - reward : int = immediate reward, r_t
        - obs : dict = agent's observation post-action, s_t+1
        - terminated : bool = whether the episode was ended (goal completed or time-limit up)
        - truncated : bool = whether the episode was ended from external reasons
        - info : dict = any extra information useful for logging, etc.
        '''
        action = action.lower().strip()
        if action not in VALID_ACTIONS:
            raise TypeError(f"Not a valid arg. 'action' must be in:{[action for action in VALID_ACTIONS]}")

        act_completed = True
        immed_reward=0
        terminated = False
        info = {}

        if self.episode_ended: 
            act_completed = False
            terminated = True
            info['termination_reason'] = "episode ended"
        elif self._max_actions_exceeded():
            act_completed = False
            terminated = True
            info['termination_reason'] = "move limit exceeded"
        else:
            self.action_count += 1
            immed_reward -= 0.01

            if self._is_last_move():
                terminated = True
                info["move_limit_reached"] = True
            
            '''
            if action=="pickup":
                if self.env_type!="pickup-and-deliver":
                    continue
            '''

            if action=="claim":
                claim_succeeded, claim_reward = self._claim_goal()
                immed_reward += claim_reward
                if claim_succeeded:
                    terminated=True
                    info["genuine_success"] = True

            else:
                #  move positions and then fix if crossed boundary
                old_pos = self.agent_pos
                new_pos = self._move_in_dir(action)
                self.agent_pos = self._make_action_legal(old_pos, new_pos)

        self.total_reward += immed_reward
        self.episode_ended = terminated
        
        obs = self.agent_observation()
        truncated = False 
        return (act_completed, immed_reward, obs, terminated, truncated, info)

    def agent_observation(self):
        '''
        all the coordinates, agent_x,y;goal_x,y, etc. are normalized [0,1]
        '''
        agent_x, agent_y = self._normalize_position(self.agent_pos)
        goal_x, goal_y = self._normalize_position(self.goal_pos)
        package_x, package_y = self._normalize_position(self.package_pos)
        remaining_time = self._get_remaining_time()
        observation = {
            "agent_x": agent_x, 
            "agent_y": agent_y, 
            "goal_x": goal_x, #goal=beacon or delivery destination depending on task mode
            "goal_y": goal_y, 
            "package_x": package_x, #package location in pickup+delivery task
            "package_y": package_y,
            "carrying_package": self._is_carrying_package(), #check if agent picked up the package  
            "package_active": self._get_package_mask(self.env_type), #mask that determines whether the agent should consider the package co-ords in decision making
            "task_mode": self._encode_task_mode(self.env_type),
            "remaining_time": remaining_time
        }
        return observation
    
    def _generate_random_coords(self, board_size):
        '''
        this function would use a uniform distribution to generate a beacon randomly 
        anywhere
        '''
        if self.env_type=="genuine-fixed":
            return self.fixed_goal_pos.copy()
        
        new_goal_pos = self.np_random.integers(0,board_size, size=2).tolist()

        # the beacon's distribution must be same for both S & E-variable agents.
        if new_goal_pos==self.fixed_goal_pos:
            new_goal_pos = self._generate_random_coords(self.board_size)
        return new_goal_pos

    def _make_action_legal(self, old, new):
        x_in_range = new[0] in range(0, self.board_size)
        y_in_range = new[1] in range(0, self.board_size)

        if not x_in_range or not y_in_range:
            return old
        
        return new

    def _max_actions_exceeded(self):
        return self.action_count>=self.max_actions

    def _is_last_move(self):
        return self.action_count==self.max_actions

    def _claim_goal(self):
        if self.agent_pos==self.goal_pos:
            claim_reward=1
            return (True, claim_reward)
        else:
            claim_reward=-0.02
            return (False, claim_reward)

    def _move_in_dir(self, dir):
        new_x, new_y = self.agent_pos
        
        if dir=="east":
            new_x += 1
        if dir=="west":
            new_x -= 1
        if dir=="north":
            new_y += 1
        if dir=="south":
            new_y  -= 1

        new_pos = [new_x, new_y]
        return new_pos

    def _normalize_position(self, position: list[int]):
        return [(x/(self.board_size-1)) if x>=0 else x for x in position]

    def _get_remaining_time(self):
        return (1-self.action_count/self.max_actions)

    def _encode_task_mode(self, task_mode):
        task_mode = task_mode.lower().strip()
        encoding = {
            "hacky-variable": 0,
            "genuine-fixed": 0,
            "genuine-variable": 0,
            "pickup-and-deliver":1
        }
        return encoding[task_mode]

    def _get_package_mask(self, env_type):
        env_type = env_type.lower().strip()
        if env_type=="pickup-and-deliver": return 1
        return 0

    def _is_carrying_package(self):
        #for now, we leave the package pickup detection logic 
        if self.env_type=="pickup-and-deliver":
            return 0
        return 0

#----------------------------------
def main():
    env = Environment(7, [3, 3], [3, 0], 30, "genuine-fixed", Generator(PCG64()))
    print(f"Observation 1: {env.agent_observation()}\n-------")
    env.agent_action("south")
    print(f"Observation 2: {env.agent_observation()}\n-------")

if __name__=="__main__":
    main()