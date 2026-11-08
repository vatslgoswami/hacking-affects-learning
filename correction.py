import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO

from gym_adapter import GridWorld
from plots import save_evaluation_plot, save_training_plots
from task_splits import development_goal_split
from train import get_run_settings, train_and_eval_loop
from utils import configure_training_logger, create_run_folder, save_run_config

CORRECTION_BUDGET = 200_000
ENTROPY_SCHEDULE = {'initial_coef': 0.1, 'final_coef': 0.01, 'decay_fraction': 0.35}

def prepare_correction_model(checkpoint, train_env, seed):
    # Restore the source policy and attach the repaired environment.
    model = PPO.load(checkpoint, env=train_env, device='cpu')
    model.set_random_seed(seed)

    # Give every history in a seed block the same newly initialized value head.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.nn.init.orthogonal_(model.policy.value_net.weight, gain=1.0)
        torch.nn.init.zeros_(model.policy.value_net.bias)

    # Start optimization and PPO updates afresh while keeping the policy weights.
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=model.lr_schedule(1), **model.policy.optimizer_kwargs
    )
    model._n_updates = 0
    model._last_obs = None
    return model

def run_correction_from_checkpoint(checkpoint, source_run, *, history, checkpoint_kind,
                                   output_dir='runs', training_budget=None,
                                   condition='source', lineage=None):
    # Read the source settings needed to recreate the corrected environment.
    source_run = Path(source_run)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if training_budget is None:
        training_budget = CORRECTION_BUDGET
    if training_budget <= 0:
        raise ValueError('training budget must be positive')
    with (source_run / 'config.json').open() as file:
        source_config = json.load(file)
    source_type = source_config['env_type']
    if source_type not in ('genuine-fixed', 'genuine-variable', 'hacky-variable'):
        raise ValueError('choose a beacon source run')

    seed = source_config['PPO_settings']['seed']
    eval_settings = source_config['training_eval_loop_settings']
    goal_split = development_goal_split()
    saved_split = source_config.get('goal_split')
    expected_split = {'seen': [list(pos) for pos in goal_split.seen],
                      'withheld': [list(pos) for pos in goal_split.withheld]}
    if source_type in ('genuine-variable', 'hacky-variable') and saved_split != expected_split:
        raise ValueError('source goal split does not match the correction goal split')

    # Build separate training and evaluation environments with the repaired checker.
    env_args = (source_config['board_size'], source_config['start_pos'],
                source_config['fixed_goal_pos'], source_config['max_actions'], 'genuine-variable')
    train_env = GridWorld(*env_args, goal_split=goal_split)
    test_env = GridWorld(*env_args, goal_split=goal_split)
    model = prepare_correction_model(checkpoint, train_env, seed)
    source_timesteps = model.num_timesteps

    run_id = f'ppo_correction-{history}-{checkpoint_kind}-seed-{seed}-{datetime.now():%Y%m%d-%H%M%S-%f}'
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
            entropy_schedule=ENTROPY_SCHEDULE, reset_num_timesteps=True,
        )
        settings = get_run_settings(model, train_env, loop_settings)
        settings['source_checkpoint'] = {
            'history': history,
            'source_run': str(source_run),
            'checkpoint': str(checkpoint),
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'checkpoint_kind': checkpoint_kind,
            'source_timesteps': source_timesteps,
            'condition': condition,
            'retained': 'policy and shared feature extractor',
            'reset': 'value head, optimizer, rollout buffer and training counters',
        }
        if lineage is not None:
            settings['source_checkpoint']['demonstration_lineage'] = lineage
        save_run_config(run_dir, settings, run_id)
        print(f'Saved evaluation plot: {save_evaluation_plot(run_dir)}')
        for plot_path in save_training_plots(run_dir):
            print(f'Saved training plot: {plot_path}')
    finally:
        logger.close()
        train_env.close()
        test_env.close()
    return run_dir

def run_correction(source_run, fresh=False, output_dir='runs', checkpoint_kind='final'):
    source_run = Path(source_run)
    with (source_run / 'config.json').open() as file:
        source_type = json.load(file)['env_type']
    if checkpoint_kind not in ('reliable', 'final'):
        raise ValueError('checkpoint kind must be reliable or final')
    checkpoints = source_run / 'checkpoints'
    matches = ([checkpoints / 'step_0.zip'] if fresh
               else list(checkpoints.glob(f'step_*_{checkpoint_kind}*.zip')))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f'expected exactly one source checkpoint in {checkpoints}')
    return run_correction_from_checkpoint(
        matches[0], source_run, history='fresh' if fresh else source_type,
        checkpoint_kind='initial' if fresh else checkpoint_kind, output_dir=output_dir,
    )

def main():
    parser = argparse.ArgumentParser(description='Train a source policy on corrected variable navigation.')
    parser.add_argument('source_run', type=Path, help='Folder containing a beacon source run')
    parser.add_argument('--fresh', action='store_true', help='Start from this run’s initial checkpoint')
    parser.add_argument('--checkpoint', choices=('reliable', 'final'), default='final',
                        help='Source checkpoint to use (default: final)')
    parser.add_argument('--output-dir', default='runs', help='Folder for the correction run')
    args = parser.parse_args()
    print(f'Correction run saved to: {run_correction(args.source_run, args.fresh, args.output_dir, args.checkpoint)}')

if __name__ == '__main__':
    main()
