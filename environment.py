import random
from typing import Literal

EnvType = Literal["hacky-variable", "genuine-fixed", "genuine-variable", "pickup-and-deliver"]
VALID_ACTIONS = ["north", "east", "south", "west", "claim", "pickup"]

class Environment:

    def __init__(self, board_size: int, start_pos: list[int], 
                 fixed_beacon_pos: list[int], max_actions: int, env_type: EnvType) -> None:
        self.board_size = board_size
        self.start_pos = start_pos
        self.fixed_beacon_pos = fixed_beacon_pos
        self.max_actions = max_actions
        self.env_type: EnvType = env_type

        #set up the environment
        self.reset()

    def _reset_agent_pos(self):
        return self.start_pos.copy()

    def spawn_beacon(self, board_size):
        '''
        this function would use a uniform distribution to generate a beacon randomly 
        anywhere
        '''
        if self.env_type=="genuine-fixed":
            return self.fixed_beacon_pos
        
        x_dim = random.randint(0,board_size-1)
        y_dim = random.randint (0,board_size-1)
        new_beacon_pos = [x_dim, y_dim]

        # the beacon's distribution must be same for both S & E-variable agents.
        if new_beacon_pos==[3,0]:
            new_beacon_pos = self.spawn_beacon(self.board_size)
        return new_beacon_pos
    
    def reset(self):
        # update the current environment so it reflects a reset
        self.agent_pos: list[int] = self._reset_agent_pos()
        self.action_count = 0
        self.beacon_pos = self.spawn_beacon(self.board_size)
        self.curr_reward = 0
        self.episode_ended = False
        return self

    def _make_legal(self, old, new):
        x_in_range = new[0] in range(0, self.board_size)
        y_in_range = new[1] in range(0, self.board_size)

        if not x_in_range or not y_in_range:
            return old
        
        return new

    def _max_actions_exceeded(self):
        return self.action_count>=self.max_actions

    def _is_last_move(self):
        return self.action_count==self.max_actions

    def _claim_beacon(self):
        if self.agent_pos==self.beacon_pos:
            self.curr_reward+=1
            return True
        else:
            self.curr_reward-=0.02
            return False
        
    def agent_action(self, action):
        if self.episode_ended: return False
        action = action.lower().strip()
        if action not in VALID_ACTIONS:
            raise TypeError(f"Not a valid arg. 'action' must be in:{[action for action in VALID_ACTIONS]}")
        if self._max_actions_exceeded():
            return False
        
        self.action_count+=1
        self.curr_reward -= 0.01
        if self._is_last_move():
            self.episode_ended = True

        if action=="pickup":
            if self.env_type!="pickup-and-deliver":
                return True

        if action=="claim":
            if self._claim_beacon():
                self.episode_ended=True

        else:
            #  move positions and then fix if crossed boundary
                    old_pos = self.agent_pos
                    new_x, new_y = self.agent_pos
                    
                    if action=="east":
                        new_x += 1
                    if action=="west":
                        new_x -= 1
                    if action=="north":
                        new_y += 1
                    if action=="south":
                        new_y  -= 1
            
                    new_pos = [new_x, new_y]
                    self.agent_pos = self._make_legal(old_pos, new_pos)
            
        return True

def main():
    agent = Environment(7, [3,3], [3,0], 30, "genuine-fixed")
    for i in range(0,3):
        agent.agent_action("South")
        print(f"Step {i+1}: {agent.curr_reward}")

    agent.agent_action("claim")
    print(f"final pos: {agent.agent_pos}")
    print(f"claiming reward...")
    print(agent.curr_reward)
    return 1

if __name__=="__main__":
    main()