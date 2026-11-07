from environment import EnvType
from stable_baselines3.common.base_class import BaseAlgorithm
from gym_adapter import GridWorld
import torch

def _board_state(test_env: GridWorld):
    env = test_env.env
    return {
        'agent_pos': list(env.agent_pos),
        'goal_pos': list(env.goal_pos),
        'package_pos': list(env.package_pos),
        'package_active': bool(env._get_package_mask(env.env_type)),
        'carrying_package': bool(env._is_carrying_package()),
    }


def evaluate_model(model:BaseAlgorithm, test_env:GridWorld, env_type:EnvType, num_episodes=25,
                   eval_seed=100, is_eval_greedy:bool=False, record_episodes=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(eval_seed)
        num_success, num_actions_taken, total_return = 0,0,0
        episodes = []
        for episode_index in range(num_episodes):
            # start a new episode
            obs, _ = test_env.reset()
            if record_episodes:
                episode = {
                    'episode_index': episode_index,
                    'board_size': test_env.env.board_size,
                    'initial_state': _board_state(test_env),
                    'steps': [],
                }
                episode_return = 0.0
            # play out the game till termination
            while True:
                action, _ = model.predict(obs, deterministic=is_eval_greedy)
                action_id = int(action)
                obs, reward, terminated, truncated, info = test_env.step(action_id)
                num_actions_taken +=1
                total_return+=reward
                if record_episodes:
                    episode_return += reward
                    episode['steps'].append({
                        'action': action_id,
                        'action_name': test_env._map_int_to_action(action_id),
                        'state': _board_state(test_env),
                        'reward': float(reward),
                        'terminated': bool(terminated),
                        'truncated': bool(truncated),
                        'info': dict(info),
                    })
                if terminated or truncated:
                    success = bool(info.get('genuine_success', False))
                    if success:
                        num_success += 1
                    if record_episodes:
                        episode['total_return'] = episode_return
                        episode['action_count'] = len(episode['steps'])
                        episode['genuine_success'] = success
                        episode['terminated'] = bool(terminated)
                        episode['truncated'] = bool(truncated)
                        episodes.append(episode)
                    break

        success_rate = num_success/num_episodes
        episode_length = num_actions_taken/num_episodes
        avg_reward = total_return/num_episodes
    metrics = (success_rate, avg_reward, episode_length)
    return metrics, episodes
