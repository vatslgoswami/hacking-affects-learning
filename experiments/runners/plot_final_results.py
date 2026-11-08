import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from plots import load_evaluation_summaries, load_training_diagnostics

ALL_SEEDS = (7, 9, 11, 13, 15, 17, 19)
DELIVERY_DEMO_SEEDS = (7, 9, 11, 13)
SOURCE_PATTERNS = (
    ('baselines', 'baseline-models/*/*/config.json'),
    ('corrections', 'corrections/*/*/config.json'),
    ('delivery_transfer', 'transfers/pickup-and-deliver/*/config.json'),
    ('post_demonstration_correction', 'demonstrations/batch-*/ppo_corrections/batch-*/*/config.json'),
    ('post_demonstration_delivery', 'demonstrations/delivery/batch-*/ppo_transfer/batch-*/*/config.json'),
)
METRICS = ('success_rate', 'seen_success_rate', 'withheld_success_rate',
           'shortcut_success_rate', 'total_success_rate', 'average_return', 'average_episode_length')

def expected_groups():
    # List the experiment conditions explicitly so a missing run cannot silently disappear.
    groups = [('baselines', task) for task in
              ('genuine-fixed', 'genuine-variable', 'hacky-variable', 'pickup-and-deliver')]
    groups += [('checker_correction', f'{history}-{checkpoint}')
               for history in ('genuine-fixed', 'hacky-variable') for checkpoint in ('reliable', 'final')]
    groups += [('action_head_reset', f'hacky-variable-{checkpoint}') for checkpoint in ('reliable', 'final')]
    groups += [('delivery_transfer', f'{history}-{checkpoint}')
               for history in ('genuine-fixed', 'genuine-variable', 'hacky-variable')
               for checkpoint in ('reliable', 'final')]
    groups += [('post_demonstration_correction', f'{history}-{checkpoint}')
               for history in ('genuine-fixed', 'hacky-variable') for checkpoint in ('reliable', 'final')]
    groups += [('post_demonstration_correction', 'fresh-initial')]
    groups += [('post_demonstration_delivery', f'{history}-{checkpoint}')
               for history in ('genuine-fixed', 'hacky-variable') for checkpoint in ('reliable', 'final')]
    return groups

def group_for_run(category, config):
    # Use the saved source history and condition, not the run folder name, to group policies.
    task = config['env_type']
    source = config.get('source_checkpoint', {})
    if category == 'baselines':
        return category, task
    history, checkpoint = source.get('history'), source.get('checkpoint_kind')
    condition = source.get('condition')
    if category == 'corrections':
        if task != 'genuine-variable' or condition not in (None, 'source', 'fresh-action-head'):
            raise ValueError(f'Unexpected correction condition: {source}')
        category = 'action_head_reset' if condition == 'fresh-action-head' else 'checker_correction'
    elif category == 'delivery_transfer':
        if task != 'pickup-and-deliver' or condition != 'source':
            raise ValueError(f'Unexpected delivery transfer: {source}')
    elif category == 'post_demonstration_correction':
        if task != 'genuine-variable' or condition != 'demonstrations':
            raise ValueError(f'Unexpected demonstration correction: {source}')
    elif category == 'post_demonstration_delivery':
        if task != 'pickup-and-deliver' or condition != 'demonstrations':
            raise ValueError(f'Unexpected demonstration delivery: {source}')
    return category, f'{history}-{checkpoint}'

def settings_signature(config):
    # Seeds differ by design, but task settings and evaluation rules must match within a figure.
    ppo = config['PPO_settings']
    loop = config['training_eval_loop_settings']
    return {
        'environment': {key: config.get(key) for key in (
            'env_type', 'board_size', 'start_pos', 'fixed_goal_pos', 'max_actions',
            'policy_observation_fields', 'goal_split', 'delivery_split', 'architecture')},
        'ppo': {key: ppo.get(key) for key in ('rollout_size', 'learning_rate', 'gamma')},
        'training': {key: loop.get(key) for key in (
            'training_budget', 'entropy_schedule', 'delivery_curriculum', 'eval_episode_count',
            'eval_action_sampler_choice', 'eval_env_seed', 'confirmation_seed',
            'confirmation_env_seed', 'competence_thresholds')},
    }

def read_run(run_dir, config):
    # Read only the measurements needed for the final figures and their provenance.
    summaries = load_evaluation_summaries(run_dir)
    entropy = load_training_diagnostics(run_dir)['train/entropy_loss']
    loop = config['training_eval_loop_settings']
    steps = [row['n_timesteps'] for row in summaries]
    if steps != loop['evaluation_schedule']:
        raise ValueError(f'Evaluation schedule differs from config: {run_dir}')
    if len(set(step for step, _ in entropy)) != len(entropy):
        raise ValueError(f'Duplicate entropy update steps: {run_dir}')
    reliable = loop['reliable_step']
    marked = [row['n_timesteps'] for row in summaries
              if row.get('competence', {}).get('is_reliable_checkpoint')]
    if marked != ([] if reliable is None else [reliable]):
        raise ValueError(f'Competence record differs from config: {run_dir}')
    metrics = []
    for row in summaries:
        metrics.append({'step': row['n_timesteps'], **{key: row[key] for key in METRICS if key in row}})
    return {
        'seed': config['PPO_settings']['seed'],
        'eval_seed': loop['eval_seed'],
        'source_run': str(run_dir),
        'source_checkpoint': config.get('source_checkpoint'),
        'code_revision': config.get('code_revision'),
        'working_tree_dirty': config.get('working_tree_dirty'),
        'reliable_step': reliable,
        'settings': settings_signature(config),
        'evaluations': metrics,
        'entropy': [[step, value] for step, value in entropy],
    }

def collect_runs(runs_dir):
    # Search only the original experiment folders and require complete matched seed sets.
    expected = set(expected_groups())
    found = defaultdict(dict)
    for category, pattern in SOURCE_PATTERNS:
        for path in sorted(Path(runs_dir).glob(pattern)):
            config = json.loads(path.read_text())
            key = group_for_run(category, config)
            if key not in expected:
                raise ValueError(f'Unexpected experiment group {key}: {path}')
            seed = config['PPO_settings']['seed']
            selected = DELIVERY_DEMO_SEEDS if key[0] == 'post_demonstration_delivery' else ALL_SEEDS
            if seed not in selected:
                continue
            if seed in found[key]:
                raise ValueError(f'Duplicate seed {seed} in {key}')
            if not list((path.parent / 'checkpoints').glob('*_final.zip')):
                raise ValueError(f'Run is incomplete: {path.parent}')
            found[key][seed] = read_run(path.parent, config)

    groups = []
    for key in expected_groups():
        selected = DELIVERY_DEMO_SEEDS if key[0] == 'post_demonstration_delivery' else ALL_SEEDS
        if set(found[key]) != set(selected):
            raise ValueError(f'Missing or unexpected seeds in {key}: {sorted(found[key])}')
        runs = [found[key][seed] for seed in selected]
        if any(run['settings'] != runs[0]['settings'] for run in runs[1:]):
            raise ValueError(f'Incompatible experiment settings in {key}')
        evaluation_steps = [row['step'] for row in runs[0]['evaluations']]
        entropy_steps = [step for step, _ in runs[0]['entropy']]
        if any([row['step'] for row in run['evaluations']] != evaluation_steps
               or [step for step, _ in run['entropy']] != entropy_steps for run in runs[1:]):
            raise ValueError(f'Incompatible evaluation or entropy steps in {key}')
        groups.append({'category': key[0], 'name': key[1], 'runs': runs})
    return {'format_version': 1, 'groups': groups}

def aggregate(group):
    # Give each seed equal weight, including seeds that never reached competence.
    runs = group['runs']
    if not runs or len({run['seed'] for run in runs}) != len(runs):
        raise ValueError(f'Missing or duplicate seeds in {group["name"]}')
    reference_steps = [row['step'] for row in runs[0]['evaluations']]
    reference_entropy_steps = [step for step, _ in runs[0]['entropy']]
    if any(run['settings'] != runs[0]['settings']
           or [row['step'] for row in run['evaluations']] != reference_steps
           or [step for step, _ in run['entropy']] != reference_entropy_steps for run in runs[1:]):
        raise ValueError(f'Incompatible settings or steps in {group["name"]}')
    steps = np.array([row['step'] for row in runs[0]['evaluations']])
    metrics = {}
    for name in METRICS:
        availability = [[name in row for row in run['evaluations']] for run in runs]
        if not any(any(row) for row in availability):
            continue
        if not all(all(row) for row in availability):
            raise ValueError(f'Incomplete {name} values in {group["category"]}/{group["name"]}')
        values = np.array([[row[name] for row in run['evaluations']] for run in runs], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite {name} values in {group["name"]}')
        metrics[name] = (values.mean(axis=0), values.std(axis=0))
    values = np.array([[value for _, value in run['entropy']] for run in runs], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite entropy values in {group["name"]}')
    entropy_steps = np.array([step for step, _ in runs[0]['entropy']])
    reached = [run['reliable_step'] for run in runs if run['reliable_step'] is not None]
    return steps, metrics, entropy_steps, (values.mean(axis=0), values.std(axis=0)), reached

def make_figure(group):
    steps, metrics, entropy_steps, entropy, reached = aggregate(group)
    runs = group['runs']
    # Keep the conditional competence mean separate from the all-seed learning curves.
    fraction = f'{len(reached)}/{len(runs)} seeds'
    reliability = (f'{fraction}; mean among reached: {np.mean(reached):,.0f} steps'
                   if reached else f'{fraction}; not reached within budget')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, layout='constrained')
    source = runs[0]['source_checkpoint']
    if source:
        title = (f'{group["category"].replace("_", " ").title()} | '
                 f'Source: {source["history"]} | Source checkpoint: {source["checkpoint_kind"]}')
    else:
        title = f'Baseline | Task: {group["name"]}'
    thresholds = runs[0]['settings']['training']['competence_thresholds'] or {}
    is_shortcut = thresholds.get('success_measure') == 'seen_shortcut_success_rate'
    competence_label = 'Reliable shortcut competence' if is_shortcut else 'Reliable competence'
    seed_text = ', '.join(str(run['seed']) for run in runs)
    fig.suptitle(f'{title}\nSeeds: {seed_text} | {competence_label}: {reliability}', fontsize=13)

    # Show genuine, seen, withheld and shortcut outcomes when the task records them.
    success_axis = axes[0, 0]
    for key, label, color, style in (
        ('success_rate', 'Genuine success', '#315dc1', '-'),
        ('seen_success_rate', 'Seen genuine', '#20846a', '--'),
        ('withheld_success_rate', 'Withheld genuine', '#d17a24', '--'),
        ('shortcut_success_rate', 'Shortcut success', '#a64480', '-.'),
        ('total_success_rate', 'Total accepted', '#676d78', ':'),
    ):
        if key not in metrics:
            continue
        mean, std = metrics[key]
        success_axis.plot(steps, mean, label=label, color=color, linestyle=style, linewidth=1.8)
        if key == 'success_rate':
            success_axis.fill_between(steps, np.maximum(0, mean - std), np.minimum(1, mean + std),
                                      color=color, alpha=0.15, label='Genuine: ±1 SD')
    if reached:
        success_axis.axvline(np.mean(reached), color='#303030', linestyle=':', linewidth=1.4,
                             label=('Mean shortcut confirmation step among reached' if is_shortcut
                                    else 'Mean confirmation step among reached'))
    success_axis.set_title('Evaluation success')
    success_axis.set_ylabel('Episode fraction')
    success_axis.set_ylim(-0.03, 1.03)
    success_axis.legend(fontsize=8, loc='best')

    # Average return, length and logged negative entropy over the same seeds.
    for axis, key, label, color in (
        (axes[0, 1], 'average_return', 'Mean episode return', '#ca7625'),
        (axes[1, 0], 'average_episode_length', 'Mean episode length', '#20846a'),
    ):
        mean, std = metrics[key]
        axis.plot(steps, mean, color=color, linewidth=1.8)
        lower, upper = mean - std, mean + std
        if key == 'average_episode_length':
            limit = runs[0]['settings']['environment']['max_actions']
            lower, upper = np.maximum(0, lower), np.minimum(limit, upper)
            axis.set_ylim(0, limit + 1)
        axis.fill_between(steps, lower, upper, color=color, alpha=0.15)
        axis.set_title(label)
        axis.set_ylabel(label)
    mean, std = entropy
    axes[1, 1].plot(entropy_steps, mean, color='#8b5abd', linewidth=1.8)
    axes[1, 1].fill_between(entropy_steps, mean - std, mean + std, color='#8b5abd', alpha=0.15)
    axes[1, 1].set_title('PPO entropy loss')
    axes[1, 1].set_ylabel('Entropy loss (negative entropy)')
    for axis in axes.flat:
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1000:g}k'))
        axis.grid(alpha=0.25)
    for axis in axes[1]:
        axis.set_xlabel('Environment transitions')
    return fig

def write_outputs(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    index = []
    groups = list(data['groups'])
    # Add matched references for comparing delivery with and without demonstrations.
    for group in data['groups']:
        if group['category'] == 'delivery_transfer' and group['name'].startswith(('genuine-fixed-', 'hacky-variable-')):
            matched = [run for run in group['runs'] if run['seed'] in DELIVERY_DEMO_SEEDS]
            if tuple(run['seed'] for run in matched) != DELIVERY_DEMO_SEEDS:
                raise ValueError(f'Missing matched delivery seeds in {group["name"]}')
            groups.append({'category': 'delivery_transfer_matched', 'name': group['name'], 'runs': matched})
    for group in groups:
        # Generate one figure and one summary row for each experiment condition.
        category, name = group['category'], group['name']
        path = output_dir / category / f'{name}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = make_figure(group)
        try:
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
        runs = group['runs']
        reached = [run for run in runs if run['reliable_step'] is not None]
        mean = np.mean([run['reliable_step'] for run in reached]) if reached else None
        summary.append({
            'category': category, 'condition': name, 'figure': str(path.relative_to(output_dir)),
            'seeds': ', '.join(str(run['seed']) for run in runs), 'seed_count': len(runs),
            'reached_seeds': ', '.join(str(run['seed']) for run in reached),
            'reached_count': len(reached), 'mean_confirmation_step_among_reached': mean,
            'requested_budget': runs[0]['settings']['training']['training_budget'],
            'actual_final_step': runs[0]['evaluations'][-1]['step'],
        })
        index.append((category, name, str(path.relative_to(output_dir)), len(reached), len(runs)))
    with (output_dir / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    write_index(output_dir, index)

def write_index(output_dir, index):
    # Explain how the seed averages and competence summaries should be read.
    lines = [
        '# Combined experiment plots', '',
        'Each figure averages one experiment condition across model seeds. Shading is ±1 population standard deviation across those seeds, not a confidence interval. Success and episode-length shading is clipped to the task’s possible range. All selected seeds, including those that never reached competence, are included in every curve.', '',
        'The competence count uses each run’s saved confirmation step. The dotted line and mean are calculated only among seeds that reached competence; they are not the point where the average curve crosses a threshold. No line is drawn when none reached competence.', '',
        'For the shortcut baseline, competence measures shortcut success and efficiency, while the blue curve measures genuine success. Other tasks use their own saved success and efficiency thresholds.', '',
        'Source checkpoint labels distinguish the earlier reliable checkpoint from the final source checkpoint. PPO follow-up steps begin at zero. Entropy loss is negative entropy measured on training rollouts and begins after the first PPO update.', '',
        'Baseline and regular follow-up groups use seeds 7, 9, 11, 13, 15, 17 and 19. Delivery after demonstrations uses the four completed seeds shared by every group: 7, 9, 11 and 13.', '',
        'For a matched comparison with delivery after demonstrations, use the four-seed `delivery_transfer_matched` figures. The regular `delivery_transfer` figures contain all seven seeds.', '',
        'Beacon PPO after demonstrations requested 50,000 transitions; ordinary correction requested 200,000. Their entropy schedules therefore decay over different absolute step counts. Delivery runs requested 300,000 steps. PPO completes full rollouts, so saved final steps can exceed the requested budget.', '',
        'Entropy loss is measured on training states, not a shared evaluation panel, so it reflects both visited states and policy uncertainty. These figures summarize PPO runs. The supervised imitation stage has different metrics and does not have a PPO entropy-loss curve. See the per-run artifacts for its metrics.', '',
        '## Figures', '',
        '| Experiment | Condition | Reliable seeds |', '| --- | --- | --- |',
    ]
    for category, name, path, reached, total in index:
        lines.append(f'| {category.replace("_", " ")} | [{name}]({path}) | {reached}/{total} |')
    lines += [
        '', '## Reproduce', '',
        'The bundled `data/seed_metrics.json.gz` contains the original per-seed measurements and provenance used for these figures. It includes follow-up seeds that are not in the smaller Hugging Face model upload.', '',
        'From the project root, run `python -m experiments.runners.plot_final_results` to regenerate the figures, this index and `summary.csv` from the bundled data. Use `python -m experiments.runners.plot_final_results --from-runs` to rebuild the bundled data from the complete original `runs/` folders.', '',
        'The CSV records seed counts, reliable seeds, conditional mean confirmation steps and both requested and actual final budgets. Repeated evaluation panels and the small number of seeds limit how broadly the results can be interpreted.',
    ]
    (output_dir / 'README.md').write_text('\n'.join(lines) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Regenerate combined plots from saved per-seed metrics.')
    parser.add_argument('--from-runs', action='store_true', help='Rebuild the bundled data from original runs')
    parser.add_argument('--runs-dir', type=Path, default=Path('runs'))
    parser.add_argument('--output-dir', type=Path, default=Path('final_plots'))
    args = parser.parse_args()
    data_path = args.output_dir / 'data' / 'seed_metrics.json.gz'
    if args.from_runs:
        # Save the underlying observations so others can reproduce the figures without all checkpoints.
        data = collect_runs(args.runs_dir)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        data_path.write_bytes(gzip.compress(encoded, mtime=0))
    else:
        data = json.loads(gzip.decompress(data_path.read_bytes()))
    write_outputs(data, args.output_dir)
    print(f'Saved {len(data["groups"])} experiment figures and 4 matched reference figures to {args.output_dir}')

if __name__ == '__main__':
    main()
