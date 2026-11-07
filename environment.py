import random
from typing import Literal
from numpy.random import Generator, PCG64 
from task_splits import DeliverySplit, GoalSplit, development_delivery_split

EnvType = Literal["hacky-variable", "genuine-fixed", "genuine-variable", "pickup-and-deliver"]
VALID_ACTIONS = ["north", "east", "south", "west", "claim", "pickup"]

class Environment:

    def __init__(self, board_size: int, start_pos: list[int], fixed_goal_pos: list[int], 
                 max_actions: int, env_type: EnvType, rng : Generator,
                 goal_split: GoalSplit | None = None, delivery_split: DeliverySplit | None = None) -> None:
        self.board_size = board_size
        self.start_pos = start_pos
        self.fixed_goal_pos = fixed_goal_pos
        self.max_actions = max_actions
        self.env_type: EnvType = env_type
        self.goal_split = self._set_goal_split(goal_split)
        self.delivery_split = self._set_delivery_split(delivery_split)

        #set up the environment
        self.reset(rng)

    def reset(self,rng, goal_pos=None, delivery_pair=None, assisted_agent_pos=None):
        # update the current environment so it reflects a reset
        self.agent_pos= self.start_pos.copy()
        self.action_count = 0
        self.np_random = rng
        self._place_task_objects(goal_pos, delivery_pair)
        self.carrying_package = False
        if assisted_agent_pos is not None:
            self._set_assisted_delivery_start(assisted_agent_pos)
        self.total_reward = 0
        self.episode_ended = False
        return self

    def _set_assisted_delivery_start(self, position):
        """Start a delivery episode carrying the package, without granting pickup reward."""
        if self.env_type != 'pickup-and-deliver':
            raise ValueError('assisted delivery starts require a delivery environment')
        if (len(position) != 2 or any(not isinstance(coord, int) or isinstance(coord, bool)
                                      or not 0 <= coord < self.board_size for coord in position)):
            raise ValueError('assisted agent position must be an in-bounds integer coordinate')
        self.agent_pos = list(position)
        self.carrying_package = True

    def _set_goal_split(self, goal_split):
        if goal_split is None:
            return None
        if self.env_type not in ('genuine-variable', 'hacky-variable'):
            raise ValueError('a goal split requires a variable-beacon environment')
        for x, y in goal_split.seen + goal_split.withheld:
            if not (0 <= x < self.board_size and 0 <= y < self.board_size) or (x, y) == tuple(self.fixed_goal_pos):
                raise ValueError('goal split contains an invalid or excluded location')
        return goal_split

    def _set_delivery_split(self, delivery_split):
        if self.env_type == 'pickup-and-deliver' and delivery_split is None:
            delivery_split = development_delivery_split()
        if delivery_split is None:
            return None
        if self.env_type != 'pickup-and-deliver':
            raise ValueError('a delivery split requires the pickup-and-deliver environment')
        for package, destination in delivery_split.seen + delivery_split.withheld:
            if (package == destination or any(
                len(pos) != 2 or not all(0 <= coord < self.board_size for coord in pos)
                or pos in (tuple(self.start_pos), tuple(self.fixed_goal_pos))
                for pos in (package, destination)
            )):
                raise ValueError('delivery split contains an invalid or excluded pair')
        return delivery_split

    def _place_task_objects(self, goal_pos, delivery_pair):
        if delivery_pair is not None and self.delivery_split is None:
            raise ValueError('delivery_pair requires a delivery environment')
        if self.delivery_split is not None:
            if goal_pos is not None:
                raise ValueError('use delivery_pair for a delivery environment')
            self.package_pos, self.goal_pos = self._select_delivery_pair(delivery_pair)
            return

        self.goal_pos = self._generate_random_coords(self.board_size)
        if goal_pos is not None:
            if (self.env_type != 'genuine-variable' or self.goal_split is None
                    or tuple(goal_pos) not in self.goal_split.seen + self.goal_split.withheld):
                raise ValueError('goal_pos must be a location in the variable-goal split')
            self.goal_pos = list(goal_pos)
        self.package_pos = [-1, -1]

    def _select_delivery_pair(self, delivery_pair):
        split = self.delivery_split
        assert split is not None  # Only called for a delivery environment.
        if delivery_pair is None:
            pair = split.seen[int(self.np_random.integers(len(split.seen)))]
        else:
            pair = tuple(tuple(pos) for pos in delivery_pair)
            if pair not in split.seen and pair not in split.withheld:
                raise ValueError('delivery_pair must be in the delivery split')
        return [list(pos) for pos in pair]

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
            
            if action=="claim":
                claim_succeeded, claim_reward = self._claim_goal()
                immed_reward += claim_reward
                if claim_succeeded:
                    terminated=True
                    info["genuine_success"] = True
                    if self.env_type == "pickup-and-deliver":
                        info["genuine_delivery"] = True

            elif action == "pickup":
                pickup_succeeded = self._pickup_package()
                if self.env_type == "pickup-and-deliver":
                    info["pickup_succeeded"] = pickup_succeeded
                    if pickup_succeeded:
                        immed_reward += 0.25

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

        if self.goal_split is not None and self.env_type in ('genuine-variable', 'hacky-variable'):
            return list(self.goal_split.seen[int(self.np_random.integers(len(self.goal_split.seen)))])
        
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
        if (self.agent_pos == self.goal_pos
                and (self.env_type != "pickup-and-deliver" or self.carrying_package)):
            claim_reward = 0.75 if self.env_type == "pickup-and-deliver" else 1
            return (True, claim_reward)
        else:
            claim_reward=-0.02
            return (False, claim_reward)

    def _pickup_package(self):
        if (self.env_type == "pickup-and-deliver" and not self.carrying_package
                and self.agent_pos == self.package_pos):
            self.carrying_package = True
            return True
        return False

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
        return int(self.carrying_package)

#----------------------------------
def main():
    env = Environment(7, [3, 3], [3, 0], 30, "genuine-fixed", Generator(PCG64()))
    print(f"Observation 1: {env.agent_observation()}\n-------")
    env.agent_action("south")
    print(f"Observation 2: {env.agent_observation()}\n-------")

if __name__=="__main__":
    main()
