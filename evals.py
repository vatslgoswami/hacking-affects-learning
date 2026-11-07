from environment import EnvType
from stable_baselines3.common.base_class import BaseAlgorithm
from gym_adapter import GridWorld
import torch

def evaluate_model(model:BaseAlgorithm, test_env:GridWorld, env_type:EnvType, num_episodes=25, 
                   eval_seed=100, is_eval_greedy:bool=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(eval_seed)
        num_success, num_actions_taken, total_return = 0,0,0 
        for _ in range(num_episodes):
            # start a new episode
            obs, _ = test_env.reset()
            # play out the game till termination
            while True:
                action, _ = model.predict(obs, deterministic=is_eval_greedy)
                obs, reward, terminated, truncated, info = test_env.step(int(action))
                num_actions_taken +=1
                total_return+=reward
                if terminated:
                    #measure successes
                    if info.get("genuine_success")==True:
                        num_success+=1
                    break
                if truncated: break

        success_rate = num_success/num_episodes
        episode_length = num_actions_taken/num_episodes
        avg_reward = total_return/num_episodes
    return success_rate, avg_reward, episode_length
