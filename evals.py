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

def _shortest_successful_length(initial_state, env_type: EnvType):
    def distance(start, end):
        return sum(abs(a - b) for a, b in zip(start, end))

    start = initial_state['agent_pos']
    goal = initial_state['goal_pos']
    if env_type == 'pickup-and-deliver' and not initial_state['carrying_package']:
        package = initial_state['package_pos']
        return distance(start, package) + distance(package, goal) + 2
    return distance(start, goal) + 1

def _evaluate_episode(model, test_env, env_type, episode_index, is_eval_greedy, record_episodes, reset_seed=None, goal=None, delivery_pair=None):
    if delivery_pair is not None:
        obs, _ = test_env.reset(seed=reset_seed, options={'delivery_pair': delivery_pair})
    elif goal is None:
        obs, _ = test_env.reset(seed=reset_seed)
    else:
        obs, _ = test_env.reset(seed=reset_seed, options={'goal_pos': goal})
    initial_state = _board_state(test_env)
    shortest_length = _shortest_successful_length(initial_state, env_type)
    episode_length = 0
    episode_return = 0.0
    picked_up = False
    reached_destination_with_package = False
    episode = None

    if record_episodes:
        episode = {
            'episode_index': episode_index,
            'board_size': test_env.env.board_size,
            'initial_state': initial_state,
            'shortest_successful_length': shortest_length,
            'steps': [],
        }

    while True:
        action, _ = model.predict(obs, deterministic=is_eval_greedy)
        action_id = int(action)
        obs, reward, terminated, truncated, info = test_env.step(action_id)
        episode_length += 1
        episode_return += reward

        if env_type == 'pickup-and-deliver':
            picked_up = picked_up or bool(info.get('pickup_succeeded', False))
            reached_destination_with_package = reached_destination_with_package or (
                test_env.env.carrying_package and test_env.env.agent_pos == test_env.env.goal_pos
            )

        if episode is not None:
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
            break

    outcome = {
        'genuine_success': success,
        'total_return': episode_return,
        'action_count': episode_length,
        'shortest_successful_length': shortest_length,
    }
    if env_type == 'pickup-and-deliver':
        outcome.update(picked_up=picked_up,
                       reached_destination_with_package=reached_destination_with_package)
    if episode is not None:
        episode.update(outcome)
        episode['extra_actions'] = episode_length - shortest_length if success else None
        episode['terminated'] = bool(terminated)
        episode['truncated'] = bool(truncated)
    return outcome, episode

def _summarize_episodes(outcomes):
    num_episodes = len(outcomes)
    successes = [outcome for outcome in outcomes if outcome['genuine_success']]
    num_success = len(successes)
    successful_lengths = [outcome['action_count'] for outcome in successes]
    successful_shortest_lengths = [outcome['shortest_successful_length'] for outcome in successes]
    summary_metrics = {
        'success_rate': num_success / num_episodes,
        'average_return': sum(outcome['total_return'] for outcome in outcomes) / num_episodes,
        'average_episode_length': sum(outcome['action_count'] for outcome in outcomes) / num_episodes,
        'num_successful_episodes': num_success,
        'num_failed_episodes': num_episodes - num_success,
        'average_successful_episode_length': sum(successful_lengths) / num_success if num_success else None,
        'average_shortest_successful_episode_length': sum(successful_shortest_lengths) / num_success if num_success else None,
        'average_extra_actions': (sum(successful_lengths) - sum(successful_shortest_lengths)) / num_success if num_success else None,
    }
    if 'picked_up' in outcomes[0]:
        summary_metrics.update({
            'num_episodes_with_pickup': sum(outcome['picked_up'] for outcome in outcomes),
            'num_episodes_reaching_destination_with_package': sum(
                outcome['reached_destination_with_package'] for outcome in outcomes),
            'num_genuine_deliveries': num_success,
        })
    return summary_metrics

def evaluate_model(model:BaseAlgorithm, test_env:GridWorld, env_type:EnvType, num_episodes=25,
                   eval_seed=100, is_eval_greedy:bool=False, record_episodes=False, eval_env_seed=10):
    if num_episodes <= 0:
        raise ValueError('num_episodes must be positive')
    if env_type != test_env.env_type:
        raise ValueError('env_type must match the evaluation environment')

    outcomes = []
    episodes = []
    split = test_env.env.goal_split if env_type == 'genuine-variable' else None
    if split is not None and num_episodes != 25:
        raise ValueError('the development goal panel requires 25 evaluation episodes')
    if split is not None and eval_env_seed is None:
        raise ValueError('the development goal panel requires an evaluation environment seed')
    panel = split.evaluation_panel(eval_env_seed) if split is not None else None
    delivery_split = test_env.delivery_split if env_type == 'pickup-and-deliver' else None
    if delivery_split is not None and (num_episodes != 25 or eval_env_seed is None):
        raise ValueError('the development delivery panel requires 25 episodes and an environment seed')
    delivery_panel = delivery_split.evaluation_panel(eval_env_seed) if delivery_split is not None else None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(eval_seed)
        for episode_index in range(num_episodes):
            outcome, episode = _evaluate_episode(
                model, test_env, env_type, episode_index, is_eval_greedy, record_episodes,
                reset_seed=eval_env_seed if episode_index == 0 else None,
                goal=panel[episode_index][0] if panel is not None else None,
                delivery_pair=delivery_panel[episode_index][0] if delivery_panel is not None else None,
            )
            if panel is not None:
                outcome['goal_group'] = panel[episode_index][1]
                if episode is not None:
                    episode['goal_group'] = panel[episode_index][1]
            if delivery_panel is not None:
                outcome['pair_group'] = delivery_panel[episode_index][1]
                if episode is not None:
                    episode['pair_group'] = delivery_panel[episode_index][1]
            outcomes.append(outcome)
            if episode is not None:
                episodes.append(episode)

    metrics = _summarize_episodes(outcomes)
    if panel is not None:
        for group in ('seen', 'withheld'):
            group_metrics = _summarize_episodes([outcome for outcome in outcomes if outcome['goal_group'] == group])
            metrics[f'{group}_success_rate'] = group_metrics['success_rate']
            metrics[f'{group}_average_extra_actions'] = group_metrics['average_extra_actions']
            metrics[f'{group}_episode_count'] = len([outcome for outcome in outcomes if outcome['goal_group'] == group])
        metrics['goal_panel'] = [{'goal': list(goal), 'group': group} for goal, group in panel]
    if delivery_panel is not None:
        for group in ('seen', 'withheld'):
            group_outcomes = [outcome for outcome in outcomes if outcome['pair_group'] == group]
            group_metrics = _summarize_episodes(group_outcomes)
            metrics[f'{group}_success_rate'] = group_metrics['success_rate']
            metrics[f'{group}_average_extra_actions'] = group_metrics['average_extra_actions']
            metrics[f'{group}_episode_count'] = len(group_outcomes)
            for name in ('num_episodes_with_pickup', 'num_episodes_reaching_destination_with_package',
                         'num_genuine_deliveries'):
                metrics[f'{group}_{name}'] = group_metrics[name]
        metrics['delivery_panel'] = [{'package': list(pair[0]), 'destination': list(pair[1]), 'group': group}
                                     for pair, group in delivery_panel]
    return metrics, episodes
