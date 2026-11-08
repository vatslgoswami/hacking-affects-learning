import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from training.evals import evaluate_model
from rl_envs.gym_adapter import GridWorld
from rl_envs.task_splits import development_delivery_split, development_goal_split
from training.mlp_architecture import reset_action_head

STARTS = ((3, 3), (3, 0), (3, 1), (2, 0), (4, 0),
          (0, 0), (0, 6), (6, 0), (6, 6))
ACTION_NAMES = ('north', 'east', 'south', 'west', 'claim', 'pickup')

def optimal_actions(position, goal):
    # A shortest move reduces the distance along either coordinate.
    x, y = position
    goal_x, goal_y = goal
    if (x, y) == (goal_x, goal_y):
        return [4]
    actions = []
    if x < goal_x:
        actions.append(1)
    elif x > goal_x:
        actions.append(3)
    if y < goal_y:
        actions.append(0)
    elif y > goal_y:
        actions.append(2)
    return actions

def make_demonstrations():
    # Set up a variable-goal environment and separate training and held-out examples.
    split = development_goal_split()
    env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable', goal_split=split)
    data = {'train': [], 'heldout': []}
    try:
        # Keep entire goals out of training and build complete shortest paths.
        for group, goals in (('train', split.seen), ('heldout', split.withheld)):
            for goal in goals:
                for start in (*STARTS, goal):
                    env.reset(seed=0, options={'goal_pos': goal})
                    env.env.agent_pos = list(start)
                    # Follow the expert until it claims the goal or runs out of moves.
                    for _ in range(env.env.max_actions):
                        position = tuple(env.env.agent_pos)
                        valid = optimal_actions(position, goal)
                        action = valid[0]
                        data[group].append({
                            'start': list(start), 'goal': list(goal), 'position': list(position),
                            'observation': env.agent_observation().tolist(),
                            'action': action, 'optimal_actions': valid,
                        })
                        _, _, terminated, _, info = env.step(action)
                        if terminated:
                            if not info.get('genuine_success'):
                                raise RuntimeError('expert path did not reach the goal')
                            break
                    else:
                        raise RuntimeError('expert path exceeded the action limit')
    finally:
        env.close()
    return data

def make_delivery_demonstrations(num_episodes=100, seed=2026):
    # Sample delivery pairs reproducibly from the seen and withheld splits.
    split = development_delivery_split()
    if num_episodes < 1 or num_episodes > min(len(split.seen), len(split.withheld)):
        raise ValueError('demonstration count must fit in both delivery splits')
    rng = np.random.default_rng(seed)
    env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver', delivery_split=split)
    data = {'train': [], 'heldout': []}
    try:
        # Reserve complete delivery pairs for held-out evaluation.
        for group, pairs in (('train', split.seen), ('heldout', split.withheld)):
            indexes = rng.choice(len(pairs), size=num_episodes, replace=False)
            for episode, index in enumerate(indexes):
                package, destination = pairs[int(index)]
                env.reset(seed=0, options={'delivery_pair': (package, destination)})
                # Start immediately after the shortest path to pickup.
                env.env._set_assisted_delivery_start(list(package))
                env.env.action_count = sum(abs(a - b) for a, b in zip((3, 3), package)) + 1
                # Record the shortest delivery path after the package has been picked up.
                for _ in range(env.env.max_actions):
                    position = tuple(env.env.agent_pos)
                    valid = optimal_actions(position, destination)
                    action = valid[0]
                    data[group].append({
                        'episode': episode, 'package': list(package), 'goal': list(destination),
                        'position': list(position), 'observation': env.agent_observation().tolist(),
                        'action': action, 'optimal_actions': valid,
                    })
                    _, _, terminated, _, info = env.step(action)
                    if terminated:
                        if not info.get('genuine_delivery'):
                            raise RuntimeError('expert path did not complete the delivery')
                        break
                else:
                    raise RuntimeError('expert path exceeded the action limit')
    finally:
        env.close()
    return data

def save_demonstrations(data, path):
    # Save the exact examples used by every model.
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
    Path(path).write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()

def prepare_model(checkpoint, head_condition, seed):
    model = PPO.load(checkpoint, device='cpu')
    if head_condition == 'fresh':
        # Give every history in the seed block the same new action head.
        reset_action_head(model, seed)
    elif head_condition != 'intact':
        raise ValueError('head condition must be intact or fresh')
    return model

def tensors(examples):
    observations = torch.tensor([item['observation'] for item in examples], dtype=torch.float32)
    actions = torch.tensor([item['action'] for item in examples], dtype=torch.long)
    mask = torch.zeros((len(examples), len(ACTION_NAMES)), dtype=torch.bool)
    for index, item in enumerate(examples):
        mask[index, item['optimal_actions']] = True
    return observations, actions, mask

def prediction_metrics(model, heldout):
    observations, actions, mask = heldout
    # Score held-out states with all equally short actions counted as valid.
    with torch.no_grad():
        distribution = model.policy.get_distribution(observations).distribution
        logits = distribution.logits
        probabilities = distribution.probs
        loss = F.cross_entropy(logits, actions).item()
        optimal_probability = (probabilities * mask).sum(dim=1).mean().item()
        correct = mask.gather(1, probabilities.argmax(dim=1, keepdim=True)).float().mean().item()
        claim = actions == 4
        claim_probability = probabilities[claim, 4].mean().item()
    return {'prediction_loss': loss, 'optimal_action_probability': optimal_probability,
            'optimal_action_accuracy': correct, 'claim_probability_at_goal': claim_probability}

def run_imitation(checkpoint, data, output_dir, *, seed, eval_seed, metadata,
                  head_condition='intact', updates=1000, batch_size=64,
                  learning_rate=0.0003, eval_interval=100,
                  target_task: Literal['genuine-variable', 'pickup-and-deliver']='genuine-variable'):
    # Load the source policy and prepare the shared training and held-out examples.
    if updates < 1 or batch_size < 1 or eval_interval < 1:
        raise ValueError('updates, batch size and evaluation interval must be positive')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model = prepare_model(checkpoint, head_condition, seed)
    training = tensors(data['train'])
    heldout = tensors(data['heldout'])
    # Only policy parameters receive gradients from the imitation loss.
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate, eps=1e-5)
    model.policy.optimizer = optimizer
    rng = np.random.default_rng(seed)
    # Evaluate the trained policy on full episodes of the selected task.
    if target_task == 'genuine-variable':
        eval_env = GridWorld(7, [3, 3], [3, 0], 30, 'genuine-variable',
                             goal_split=development_goal_split())
    elif target_task == 'pickup-and-deliver':
        eval_env = GridWorld(7, [3, 3], [3, 0], 30, 'pickup-and-deliver',
                             delivery_split=development_delivery_split())
    else:
        raise ValueError('unsupported imitation task')
    history = []
    try:
        # Alternate supervised updates with held-out and rollout evaluations.
        for update in range(updates + 1):
            if update % eval_interval == 0 or update == updates:
                # Evaluate without training on the held-out goals.
                prediction = prediction_metrics(model, heldout)
                rollout, episodes = evaluate_model(
                    model, eval_env, target_task, num_episodes=25, eval_seed=eval_seed,
                    is_eval_greedy=False, record_episodes=update in (0, updates), eval_env_seed=10,
                )
                history.append({'updates': update, **prediction, 'rollout': rollout})
                if episodes:
                    (output_dir / f'episodes_{update}.json').write_text(json.dumps(episodes))
                print(f"  update {update}: withheld success {rollout['withheld_success_rate']:.2f}, "
                      f"held-out loss {prediction['prediction_loss']:.3f}", flush=True)
            if update == updates:
                break
            # Use the same sampled examples for every history in this seed block.
            indexes = torch.as_tensor(rng.integers(len(training[1]), size=batch_size), dtype=torch.long)
            distribution = model.policy.get_distribution(training[0][indexes]).distribution
            loss = F.cross_entropy(distribution.logits, training[1][indexes])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Save the trained policy, learning metrics and the source details needed to reproduce it.
        model.save(output_dir / 'final_model.zip')
        config = {
            **metadata, 'seed': seed, 'eval_seed': eval_seed, 'eval_env_seed': 10,
            'head_condition': head_condition, 'updates': updates, 'batch_size': batch_size,
            'learning_rate': learning_rate, 'eval_interval': eval_interval,
            'checkpoint': str(checkpoint),
            'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'train_examples': len(data['train']), 'heldout_examples': len(data['heldout']),
            'target_task': target_task,
            'optimizer': 'Adam', 'optimizer_epsilon': 1e-5,
            'rollout_panel': ('15 seen and 10 held-out delivery pairs' if target_task == 'pickup-and-deliver'
                              else '15 seen episodes and 10 held-out goal episodes'),
            'saved_model_note': 'Imitation model for evaluation; reset optimizer before PPO training.',
        }
        (output_dir / 'config.json').write_text(json.dumps(config, indent=2))
        (output_dir / 'metrics.json').write_text(json.dumps(history, indent=2))
    finally:
        eval_env.close()
    return history
