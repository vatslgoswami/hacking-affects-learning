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
        success_label = ('Genuine' if any('shortcut_success_rate' in summary for summary in summaries)
                         else 'Overall')
        axis.plot(steps, [summary[key] for summary in summaries], marker='o', color=color,
                  label=success_label if key == 'success_rate' else None)
        axis.set_ylabel(axis_label)
        axis.grid(True, alpha=0.25)
    if any('seen_success_rate' in summary for summary in summaries):
        shortcut_metrics = any('shortcut_success_rate' in summary for summary in summaries)
        axes[0].plot(steps, [summary.get('seen_success_rate') for summary in summaries],
                     marker='o', color='#25866c', label='Seen genuine' if shortcut_metrics else 'Seen')
        axes[0].plot(steps, [summary.get('withheld_success_rate') for summary in summaries],
                     marker='o', color='#d77a2e', label='Withheld genuine' if shortcut_metrics else 'Withheld')
        axes[0].legend(fontsize=8)
    if any('shortcut_success_rate' in summary for summary in summaries):
        axes[0].set_ylabel('Completion rate')
        axes[0].plot(steps, [summary.get('shortcut_success_rate') for summary in summaries],
                     marker='o', color='#b34684', label='Shortcut')
        axes[0].plot(steps, [summary.get('total_success_rate') for summary in summaries],
                     marker='o', color='#475166', linestyle='--', label='Total accepted')
        axes[0].legend(fontsize=8)
    reliable_step = next((s['n_timesteps'] for s in summaries
                          if s.get('competence', {}).get('is_reliable_checkpoint')), None)
    if reliable_step is not None:
        for axis in axes:
            axis.axvline(reliable_step, color='#b34684', linestyle=':',
                         label=f'Reliable competence: {reliable_step:,}')
        axes[0].legend(fontsize=8)
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

def create_training_figures(metrics, run_name, model_seed='unknown', reliable_step=None):
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
    if reliable_step is not None:
        for axis in (loss_axis, entropy_axis):
            axis.axvline(reliable_step, color='#b34684', linestyle=':',
                         label=f'Reliable competence: {reliable_step:,}')
            axis.legend(fontsize=8)
    return losses, entropy

def save_training_plots(run_dir):
    run_dir = Path(run_dir)
    metrics = load_training_diagnostics(run_dir)
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    seed = config.get('PPO_settings', {}).get('seed', 'unknown')
    reliable_step = config.get('training_eval_loop_settings', {}).get('reliable_step')
    if reliable_step is None and (run_dir / 'evaluations').exists():
        try:
            summaries = load_evaluation_summaries(run_dir)
            reliable_step = next((s['n_timesteps'] for s in summaries
                                  if s.get('competence', {}).get('is_reliable_checkpoint')), None)
        except ValueError:
            pass
    losses, entropy = create_training_figures(metrics, run_dir.name, seed, reliable_step)
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

def save_demonstration_plots(batch_dir, results, histories, checkpoint_kinds, head_conditions,
                             target_task='genuine-variable'):
    # Compare the average learning curves across seeds.
    outputs = []
    for kind in checkpoint_kinds:
        for head in head_conditions:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
            try:
                for history in (*histories, 'fresh'):
                    selected = [item for item in results if item['history'] == history
                                and item['head'] == head
                                and item['checkpoint_kind'] == ('initial' if history == 'fresh' else kind)]
                    if not selected:
                        continue
                    steps = [row['updates'] for row in selected[0]['metrics']]
                    success = [sum(item['metrics'][i]['rollout']['withheld_success_rate'] for item in selected)
                               / len(selected) for i in range(len(steps))]
                    loss = [sum(item['metrics'][i]['prediction_loss'] for item in selected)
                            / len(selected) for i in range(len(steps))]
                    axes[0].plot(steps, success, label=history)
                    axes[1].plot(steps, loss, label=history)
                withheld_label = ('Withheld-pair success rate' if target_task == 'pickup-and-deliver'
                                  else 'Withheld-goal success rate')
                axes[0].set_ylabel(withheld_label)
                axes[0].set_ylim(-0.03, 1.03)
                axes[1].set_ylabel('Held-out expert-action loss')
                for axis in axes:
                    axis.set_xlabel('Imitation updates')
                    axis.grid(alpha=0.25)
                    axis.legend()
                fig.suptitle(f'{kind} source checkpoint, {head} action head')
                output = Path(batch_dir) / f'{kind}_{head}.png'
                fig.savefig(output, dpi=160)
                outputs.append(output)
            finally:
                plt.close(fig)
    return outputs

def save_demonstration_correction_plots(batch_dir):
    # Compare correction curves after demonstrations.
    runs = []
    for config_path in sorted(Path(batch_dir).glob('*/config.json')):
        config = json.loads(config_path.read_text())
        source = config['source_checkpoint']
        runs.append({'history': source['history'], 'kind': source['checkpoint_kind'],
                     'condition': source['condition'],
                     'metrics': load_evaluation_summaries(config_path.parent)})
    colors = {'hacky-variable': '#b34684', 'genuine-fixed': '#526bf4',
              'genuine-variable': '#25866c', 'fresh': '#d77a2e'}
    outputs = []
    for kind in sorted({run['kind'] for run in runs} - {'initial'}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
        try:
            for history in sorted({run['history'] for run in runs}):
                for condition, style in (('demonstrations', '-'),):
                    selected = [run for run in runs if run['history'] == history
                                and run['kind'] == ('initial' if history == 'fresh' else kind)
                                and run['condition'] == condition]
                    if not selected:
                        continue
                    steps = [row['n_timesteps'] for row in selected[0]['metrics']]
                    if any([row['n_timesteps'] for row in run['metrics']] != steps for run in selected):
                        raise ValueError(f'inconsistent correction evaluation steps for {history}')
                    for axis, field in zip(axes, ('withheld_success_rate', 'average_episode_length')):
                        means = [sum(run['metrics'][i][field] for run in selected) / len(selected)
                                 for i in range(len(steps))]
                        axis.plot(steps, means, color=colors.get(history), linestyle=style,
                                  label=f'{history}: {condition} (n={len(selected)})')
            axes[0].set_ylabel('Withheld-goal success rate')
            axes[0].set_ylim(-0.03, 1.03)
            axes[1].set_ylabel('Mean episode length (all goals)')
            for axis in axes:
                axis.set_xlabel('Correction transitions')
                axis.grid(alpha=0.25)
            axes[0].legend(fontsize=8)
            fig.suptitle(f'PPO correction after demonstrations: {kind} source checkpoints')
            output = Path(batch_dir) / f'correction_after_demonstrations_{kind}.png'
            fig.savefig(output, dpi=160)
            outputs.append(output)
        finally:
            plt.close(fig)
    return outputs

def save_delivery_demonstration_comparisons(batch_dir, runs):
    # Compare each demonstrated run with its matched original transfer run.
    colors = {'genuine-fixed': '#526bf4', 'hacky-variable': '#b34684'}
    outputs = []
    for kind in ('reliable', 'final'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
        try:
            for history in ('genuine-fixed', 'hacky-variable'):
                selected = [run for run in runs if run['history'] == history and run['checkpoint_kind'] == kind]
                if not selected:
                    continue
                for condition, style in (('without', '--'), ('with', '-')):
                    panels = [load_evaluation_summaries(run[condition]) for run in selected]
                    steps = [row['n_timesteps'] for row in panels[0]]
                    if any([row['n_timesteps'] for row in panel] != steps for panel in panels):
                        raise ValueError(f'inconsistent delivery evaluation steps for {history}')
                    fields = ('success_rate', 'withheld_success_rate', 'num_episodes_with_pickup')
                    for axis, field in zip(axes, fields):
                        values = [sum(panel[i][field] for panel in panels) / len(panels)
                                  for i in range(len(steps))]
                        if field == 'num_episodes_with_pickup':
                            values = [value / 25 for value in values]
                        axis.plot(steps, values, color=colors[history], linestyle=style,
                                  label=f'{history}: {condition} demonstrations (n={len(panels)})')
            for axis, title in zip(axes, ('Overall delivery success', 'Withheld-pair success', 'Pickup rate')):
                axis.set_title(title)
                axis.set_xlabel('PPO transitions')
                axis.set_ylim(-0.03, 1.03)
                axis.grid(alpha=0.25)
            axes[0].set_ylabel('Episode fraction')
            axes[0].legend(fontsize=8)
            fig.suptitle(f'Delivery transfer: {kind} source checkpoints')
            output = Path(batch_dir) / f'delivery_comparison_{kind}.png'
            fig.savefig(output, dpi=160)
            outputs.append(output)
        finally:
            plt.close(fig)
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

