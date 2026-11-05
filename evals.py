from environment import EnvType
from stable_baselines3 import PPO
from gym_adapter import GridWorld
import torch

def evaluate_model(model:PPO, test_env:GridWorld, env_type:EnvType, 
                   num_episodes=25, eval_act_seed=100):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(eval_act_seed)
        num_success, num_actions_taken, total_return = 0,0,0 
        for _ in range(num_episodes):
            # start a new episode
            obs, _ = test_env.reset()
            # play out the game till termination
            while True:
                action, _ = model.predict(obs, deterministic=False)
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