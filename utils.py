import time
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

def create_run_folder(BASE_DIR, run_id):
    base_dir = Path(BASE_DIR)
    run_dir = base_dir / f'run-{run_id}'
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = run_dir / 'checkpoints'
    evals_dir = run_dir / 'evaluations'
    plots_dir = run_dir / 'plots'
    for path in [checkpoints_dir, evals_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return run_dir, checkpoints_dir, evals_dir, plots_dir

def save_checkpoint(model, checkpoints_dir):
    checkpoint_path = Path(checkpoints_dir) / f'step_{model.num_timesteps}.zip'
    with checkpoint_path.open('xb') as checkpoint_file:
        model.save(checkpoint_file)
    return checkpoint_path

def load_checkpoint(checkpoint_path, env=None):
    return PPO.load(checkpoint_path, env=env, device='cpu')

def configure_training_logger(model, run_dir):
    logger = configure(str(run_dir), ['stdout', 'csv'])
    model.set_logger(logger)
    return logger

def save_evaluation_summary(evals_dir, n_timesteps, metrics, eval_episode_count, eval_seed, is_eval_greedy):
    success_rate, average_return, average_episode_length = metrics
    summary = {
        'n_timesteps': n_timesteps,
        'success_rate': success_rate,
        'average_return': average_return,
        'average_episode_length': average_episode_length,
        'eval_episode_count': eval_episode_count,
        'eval_seed': eval_seed,
        'eval_action_sampler_choice': 'greedy' if is_eval_greedy else 'stochastic',
    }
    path = Path(evals_dir) / f'step_{n_timesteps}.json'
    with path.open('x') as file:
        json.dump(summary, file, indent=2)
    return path

def save_evaluation_episodes(evals_dir, n_timesteps, episodes, eval_seed, is_eval_greedy):
    record = {
        'n_timesteps': n_timesteps,
        'eval_episode_count': len(episodes),
        'eval_seed': eval_seed,
        'eval_action_sampler_choice': 'greedy' if is_eval_greedy else 'stochastic',
        'episodes': episodes,
    }
    path = Path(evals_dir) / f'episodes_{n_timesteps}.json'
    with path.open('x') as file:
        json.dump(record, file, indent=2)
    return path

def save_run_config(RUN_DIR, settings, run_id):
    config = {
        'run_id': run_id,
        'env_type' : settings["env_type"],
        'task_mode': settings['task_mode'],
        'board_size': settings['board_size'],
        'start_pos': settings['start_pos'],
        'fixed_goal_pos': settings['fixed_goal_pos'],
        'max_actions': settings['max_actions'],
        'policy_observation_fields': settings['policy_observation_fields'],
        'architecture': settings['architecture'],
        'PPO_settings': settings['PPO_settings'],
        'training_eval_loop_settings': settings['training_eval_loop_settings'],
        'versions': {
            'python': platform.python_version(),
            'numpy': version('numpy'),
            'torch': version('torch'),
            'gymnasium': version('gymnasium'),
            'stable_baselines3': version('stable-baselines3'),
        },
    }
    try:
        repo_dir = Path(__file__).resolve().parent
        config['code_revision'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        config['working_tree_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        config['code_revision'] = None
        config['working_tree_dirty'] = None
    config_json = json.dumps(config, indent=2)
    write_file = RUN_DIR/'config.json'
    with open(write_file,'x') as f:
        f.write(config_json)
    return write_file
    
