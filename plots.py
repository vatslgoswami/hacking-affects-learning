import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_evaluation_summaries(run_dir):
    evaluations_dir = Path(run_dir) / 'evaluations'
    paths = list(evaluations_dir.glob('step_*.json'))
    if not paths:
        raise ValueError(f'No evaluation summaries found in {evaluations_dir}')

    summaries = []
    for path in paths:
        with path.open() as file:
            summaries.append(json.load(file))
    summaries.sort(key=lambda summary: summary['n_timesteps'])
    if len({summary['n_timesteps'] for summary in summaries}) != len(summaries):
        raise ValueError('Evaluation summaries contain duplicate timestep counts')
    return summaries


def create_evaluation_figure(summaries, run_name, config=None):
    steps = [summary['n_timesteps'] for summary in summaries]
    episode_counts = {summary['eval_episode_count'] for summary in summaries}
    eval_seeds = {summary['eval_seed'] for summary in summaries}
    samplers = {summary['eval_action_sampler_choice'] for summary in summaries}

    def label(values):
        return str(next(iter(values))) if len(values) == 1 else 'varies'

    model_seed = (config or {}).get('PPO_settings', {}).get('seed', 'unknown')
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout='constrained')
    fig.suptitle(f'Evaluation learning curves — {run_name}', fontsize=15)
    axes[0].set_title(
        f'Model seed: {model_seed}  |  Eval seed: {label(eval_seeds)}  |  '
        f'Episodes: {label(episode_counts)}  |  Actions: {label(samplers)}',
        fontsize=9,
    )

    metrics = [
        ('success_rate', 'Genuine success rate', '#526bf4'),
        ('average_return', 'Mean episode return', '#d77a2e'),
        ('average_episode_length', 'Mean episode length\n(all episodes)', '#25866c'),
    ]
    for axis, (key, axis_label, color) in zip(axes, metrics):
        axis.plot(steps, [summary[key] for summary in summaries], marker='o', color=color)
        axis.set_ylabel(axis_label)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylim(-0.03, 1.03)
    axes[-1].set_xlabel('Training transitions')

    if config and config.get('env_type') == 'genuine-fixed':
        start = config.get('start_pos')
        goal = config.get('fixed_goal_pos')
        if start == [3, 3] and goal == [3, 0] and config.get('board_size') == 7:
            axes[1].axhline(0.96, color='#777777', linestyle='--', label='Shortest route: 0.96')
            axes[2].axhline(4, color='#777777', linestyle='--', label='Shortest success: 4 actions')
            axes[1].legend(fontsize=8)
            axes[2].legend(fontsize=8)
    return fig


def save_evaluation_plot(run_dir):
    run_dir = Path(run_dir)
    summaries = load_evaluation_summaries(run_dir)
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    figure = create_evaluation_figure(summaries, run_dir.name, config)
    output = run_dir / 'plots' / 'learning_curves.png'
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)
    return output




def load_training_diagnostics(run_dir):
    path = Path(run_dir) / 'progress.csv'
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))

    metrics = {}
    for key in ('train/policy_gradient_loss', 'train/value_loss', 'train/entropy_loss'):
        points = []
        for row in rows:
            if row.get('train/update_timesteps') and row.get(key):
                points.append((int(row['train/update_timesteps']), float(row[key])))
        if not points:
            raise ValueError(f'No recorded values for {key} in {path}')
        metrics[key] = sorted(points)
    return metrics


def create_training_figures(metrics, run_name, model_seed='unknown'):
    losses, loss_axis = plt.subplots(figsize=(9, 5), layout='constrained')
    for key, label, color in (
        ('train/policy_gradient_loss', 'Policy loss', '#526bf4'),
        ('train/value_loss', 'Value loss', '#d77a2e'),
    ):
        steps, values = zip(*metrics[key])
        loss_axis.plot(steps, values, label=label, color=color)
    loss_axis.set_title(f'PPO policy and value loss — {run_name}\nModel seed: {model_seed}')
    loss_axis.set_xlabel('Training transitions at update')
    loss_axis.set_ylabel('Loss')
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend()

    entropy, entropy_axis = plt.subplots(figsize=(9, 5), layout='constrained')
    steps, values = zip(*metrics['train/entropy_loss'])
    entropy_axis.plot(steps, values, color='#25866c')
    entropy_axis.set_title(f'PPO entropy loss — {run_name}\nModel seed: {model_seed}')
    entropy_axis.set_xlabel('Training transitions at update')
    entropy_axis.set_ylabel('Entropy loss (negative entropy)')
    entropy_axis.grid(True, alpha=0.25)
    return losses, entropy


def save_training_plots(run_dir):
    run_dir = Path(run_dir)
    metrics = load_training_diagnostics(run_dir)
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    seed = config.get('PPO_settings', {}).get('seed', 'unknown')
    losses, entropy = create_training_figures(metrics, run_dir.name, seed)
    outputs = (run_dir / 'plots' / 'policy_value_loss.png',
               run_dir / 'plots' / 'entropy_loss.png')
    try:
        outputs[0].parent.mkdir(parents=True, exist_ok=True)
        losses.savefig(outputs[0], dpi=150)
        entropy.savefig(outputs[1], dpi=150)
    finally:
        plt.close(losses)
        plt.close(entropy)
    return outputs


def main():
    parser = argparse.ArgumentParser(description='Plot saved results for a training run.')
    parser.add_argument('run_dir', type=Path, help='Path to the run directory')
    parser.add_argument('--kind', choices=('all', 'evaluation', 'training'), default='all',
                        help='Which plots to generate (default: all)')
    args = parser.parse_args()
    if args.kind in ('all', 'evaluation'):
        print(f'Saved evaluation plot: {save_evaluation_plot(args.run_dir)}')
    if args.kind in ('all', 'training'):
        for path in save_training_plots(args.run_dir):
            print(f'Saved training plot: {path}')


if __name__ == '__main__':
    main()


