import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from stable_baselines3 import PPO

from rl_envs.gym_adapter import GridWorld
from plots import save_evaluation_plot, save_training_plots
from rl_envs.task_splits import GoalSplit, development_delivery_split, development_goal_split
from experiments.common import find_checkpoint, load_source_config
from experiments.settings import CORRECTION_BUDGET, DELIVERY_BUDGET, DELIVERY_CURRICULUM, ENTROPY_SCHEDULE
from training.mlp_architecture import reset_action_head as reset_policy_action_head
from training.train import get_run_settings, train_and_eval_loop
from utils import configure_training_logger, create_run_folder, save_run_config

def _load_source_settings(source_run, checkpoint):
    # Check that the selected checkpoint belongs to a supported source task.
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = load_source_config(source_run)
    if config['env_type'] not in ('genuine-fixed', 'genuine-variable', 'hacky-variable'):
        raise ValueError('choose a beacon source run')
    return config

def _create_follow_up_environments(source_config,
                                   target_task: Literal['genuine-variable', 'pickup-and-deliver']):
    # Keep the source goal split consistent with the navigation task.
    goal_split: GoalSplit | None = development_goal_split()
    expected_split = {'seen': [list(pos) for pos in goal_split.seen],
                      'withheld': [list(pos) for pos in goal_split.withheld]}
    if (source_config['env_type'] in ('genuine-variable', 'hacky-variable')
            and source_config.get('goal_split') != expected_split):
        raise ValueError('source goal split does not match the correction goal split')

    # Select the task's split and any assisted training curriculum.
    delivery_split = None
    delivery_curriculum = None
    if target_task == 'pickup-and-deliver':
        if (source_config['board_size'], source_config['start_pos'], source_config['fixed_goal_pos'],
                source_config['max_actions']) != (7, [3, 3], [3, 0], 30):
            raise ValueError('delivery transfer requires the board settings used by the fresh control')
        goal_split = None
        delivery_split = development_delivery_split()
        delivery_curriculum = DELIVERY_CURRICULUM.copy()
    env_args = (source_config['board_size'], source_config['start_pos'],
                source_config['fixed_goal_pos'], source_config['max_actions'], target_task)
    train_env = GridWorld(*env_args, goal_split=goal_split, delivery_split=delivery_split)
    test_env = GridWorld(*env_args, goal_split=goal_split, delivery_split=delivery_split)
    return train_env, test_env, delivery_curriculum

def prepare_correction_model(checkpoint, train_env, seed, reset_action_head=False):
    # Restore the source policy and attach the repaired environment.
    model = PPO.load(checkpoint, env=train_env, device='cpu')
    model.set_random_seed(seed)

    # Give every history in a seed block the same newly initialized value head.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.nn.init.orthogonal_(model.policy.value_net.weight, gain=1.0)
        torch.nn.init.zeros_(model.policy.value_net.bias)

    if reset_action_head:
        # Replace the action preferences while keeping the policy's earlier layers.
        reset_policy_action_head(model, seed)

    # Start optimization and PPO updates afresh while keeping the policy weights.
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=model.lr_schedule(1), **model.policy.optimizer_kwargs
    )
    model._n_updates = 0
    model._last_obs = None
    return model

def _save_follow_up_results(run_dir, run_id, model, train_env, loop_settings, source_details):
    # Save the source checkpoint details with the training settings.
    settings = get_run_settings(model, train_env, loop_settings)
    settings['source_checkpoint'] = source_details
    save_run_config(run_dir, settings, run_id)

    # Save evaluation and PPO diagnostic plots.
    print(f'Saved evaluation plot: {save_evaluation_plot(run_dir)}')
    for plot_path in save_training_plots(run_dir):
        print(f'Saved training plot: {plot_path}')

def run_correction_from_checkpoint(checkpoint, source_run, *, history, checkpoint_kind,
                                   output_dir='runs', training_budget=None,
                                   condition='source', lineage=None, reset_action_head=False,
                                   target_task: Literal['genuine-variable', 'pickup-and-deliver']='genuine-variable'):
    # Load the source settings and select the follow-up budget.
    source_run = Path(source_run)
    checkpoint = Path(checkpoint)
    source_config = _load_source_settings(source_run, checkpoint)
    if target_task not in ('genuine-variable', 'pickup-and-deliver'):
        raise ValueError('target task must be genuine-variable or pickup-and-deliver')
    if training_budget is None:
        training_budget = DELIVERY_BUDGET if target_task == 'pickup-and-deliver' else CORRECTION_BUDGET
    if training_budget <= 0:
        raise ValueError('training budget must be positive')
    seed = source_config['PPO_settings']['seed']
    eval_settings = source_config['training_eval_loop_settings']

    # Create separate environments and restart training from the source policy.
    train_env, test_env, delivery_curriculum = _create_follow_up_environments(source_config, target_task)
    model = prepare_correction_model(checkpoint, train_env, seed, reset_action_head)
    source_timesteps = model.num_timesteps

    # Create a separate output folder and logger for this follow-up run.
    branch = 'ppo_transfer' if target_task == 'pickup-and-deliver' else 'ppo_correction'
    run_id = f'{branch}-{history}-{checkpoint_kind}-seed-{seed}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    run_dir, checkpoints_dir, evals_dir, _ = create_run_folder(output_dir, run_id)
    logger = configure_training_logger(model, run_dir)
    try:
        # Evaluate at step zero, then train and evaluate throughout correction.
        model, loop_settings = train_and_eval_loop(
            model, test_env, training_budget, eval_settings['eval_seed'],
            checkpoints_dir, evals_dir,
            num_eval_episodes=eval_settings['eval_episode_count'],
            is_eval_greedy=eval_settings['eval_action_sampler_choice'] == 'greedy',
            eval_env_seed=eval_settings['eval_env_seed'],
            confirmation_seed=eval_settings['confirmation_seed'],
            confirmation_env_seed=eval_settings['confirmation_env_seed'],
            entropy_schedule=ENTROPY_SCHEDULE, delivery_curriculum=delivery_curriculum,
            reset_num_timesteps=True,
        )
        # Record which source model and reset choices produced these results.
        source_details: dict[str, object] = {
            'history': history,
            'source_run': str(source_run),
            'checkpoint': str(checkpoint),
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'checkpoint_kind': checkpoint_kind,
            'source_timesteps': source_timesteps,
            'condition': condition,
            'retained': ('policy layers except action head and shared feature extractor'
                         if reset_action_head else 'policy and shared feature extractor'),
            'reset': ('action head, value head, optimizer, rollout buffer and training counters'
                      if reset_action_head else 'value head, optimizer, rollout buffer and training counters'),
            'action_head': 'fresh' if reset_action_head else 'intact',
        }
        if lineage is not None:
            source_details['demonstration_lineage'] = lineage
        # Save the settings and plots after the training loop finishes.
        _save_follow_up_results(run_dir, run_id, model, train_env, loop_settings, source_details)
    finally:
        # Release the logger and both environments even if training fails.
        logger.close()
        train_env.close()
        test_env.close()
    return run_dir

def run_correction(source_run, fresh=False, output_dir='runs', checkpoint_kind='final',
                   target_task: Literal['genuine-variable', 'pickup-and-deliver']='genuine-variable'):
    # Find the requested source checkpoint before starting the follow-up run.
    source_run = Path(source_run)
    source_type = load_source_config(source_run)['env_type']
    if checkpoint_kind not in ('reliable', 'final'):
        raise ValueError('checkpoint kind must be reliable or final')
    checkpoint = find_checkpoint(source_run, 'initial' if fresh else checkpoint_kind)
    return run_correction_from_checkpoint(
        checkpoint, source_run, history='fresh' if fresh else source_type,
        checkpoint_kind='initial' if fresh else checkpoint_kind, output_dir=output_dir,
        target_task=target_task,
    )

def main():
    # Turn the command line choices into one correction or delivery transfer run.
    parser = argparse.ArgumentParser(description='Train a source policy on a follow-up task.')
    parser.add_argument('source_run', type=Path, help='Folder containing a beacon source run')
    parser.add_argument('--fresh', action='store_true', help='Start from this run’s initial checkpoint')
    parser.add_argument('--checkpoint', choices=('reliable', 'final'), default='final',
                        help='Source checkpoint to use (default: final)')
    parser.add_argument('--output-dir', default='runs', help='Folder for the correction run')
    parser.add_argument('--task', choices=('genuine-variable', 'pickup-and-deliver'),
                        default='genuine-variable', help='Follow-up task')
    args = parser.parse_args()
    print(f'Follow-up run saved to: {run_correction(args.source_run, args.fresh, args.output_dir, args.checkpoint, args.task)}')

if __name__ == '__main__':
    main()
