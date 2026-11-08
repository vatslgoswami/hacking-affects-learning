import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from demonstrations import make_demonstrations, run_imitation, save_demonstrations
from correction import run_correction_from_checkpoint
from plots import save_demonstration_plots, save_demonstration_correction_plots

BASELINE_DIR = Path('runs/baseline-models')
HISTORIES = ('hacky-variable', 'genuine-fixed')
HEAD_CONDITIONS = ['intact']
CHECKPOINTS = ('reliable', 'final')

def build_run_plan(seed_numbers):
    plan = []
    for number in seed_numbers:
        paired_seed = None
        paired_eval_seed = None
        for history in HISTORIES:
            source = BASELINE_DIR / f'{history}-beacon' / f'run-{history}-seed-#{number}'
            config = json.loads((source / 'config.json').read_text())
            if config['env_type'] != history:
                raise ValueError(f'Source history does not match {source}')
            seed = config['PPO_settings']['seed']
            eval_seed = config['training_eval_loop_settings']['eval_seed']
            if paired_seed is None:
                paired_seed, paired_eval_seed = seed, eval_seed
            if (seed, eval_seed) != (paired_seed, paired_eval_seed):
                raise ValueError(f'Seed block {number} does not match across histories')
            if history == 'genuine-fixed':
                initial = source / 'checkpoints' / 'step_0.zip'
                if not initial.is_file():
                    raise FileNotFoundError(initial)
                for head in HEAD_CONDITIONS:
                    plan.append({'history': 'fresh', 'source_run': source, 'checkpoint': initial,
                                 'checkpoint_kind': 'initial', 'head': head, 'seed': seed,
                                 'eval_seed': eval_seed, 'seed_number': number})
            for kind in CHECKPOINTS:
                paths = list((source / 'checkpoints').glob(f'step_*_{kind}*.zip'))
                if len(paths) != 1:
                    raise ValueError(f'Expected one {kind} checkpoint in {source}')
                for head in HEAD_CONDITIONS:
                    plan.append({'history': history, 'source_run': source, 'checkpoint': paths[0],
                                 'checkpoint_kind': kind, 'head': head, 'seed': seed,
                                 'eval_seed': eval_seed, 'seed_number': number})
    return plan

def parse_args():
    parser = argparse.ArgumentParser(description='Run common demonstrations across source histories.')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 8)),
                        help='Baseline folder labels to include (default: 1 through 7)')
    parser.add_argument('--updates', type=int, default=1000, help='Imitation updates per run')
    parser.add_argument('--batch-size', type=int, default=64, help='Examples in each update')
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--eval-interval', type=int, default=100, help='Updates between evaluations')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/demonstrations'))
    parser.add_argument('--correction-batch', type=Path,
                        help='Run PPO corrections from an existing demonstration batch')
    parser.add_argument('--correction-budget', type=int, default=50_000,
                        help='PPO training steps per correction run')
    parser.add_argument('--skip-correction', action='store_true',
                        help='Only run the demonstrations')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('seed labels must be unique')
    if (args.updates < 1 or args.batch_size < 1 or args.eval_interval < 1
            or args.learning_rate <= 0 or args.correction_budget < 1):
        raise ValueError('training settings must be positive')
    if args.correction_batch and args.skip_correction:
        raise ValueError('cannot skip corrections when selecting a correction batch')

    return args

def build_correction_plan(batch_dir):
    batch_dir = Path(batch_dir)
    batch_config = json.loads((batch_dir / 'config.json').read_text())
    data_path = batch_dir / 'demonstrations.json'
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if data_hash != batch_config['demonstrations_sha256']:
        raise ValueError('demonstration dataset does not match the batch config')
    configs = sorted(batch_dir.glob('*/config.json'))
    if len(configs) != batch_config['number_of_runs']:
        raise ValueError('demonstration batch is incomplete')

    # Check which source checkpoint produced each imitation model.
    runs = []
    for path in configs:
        config = json.loads(path.read_text())
        if config['head_condition'] != 'intact':
            raise ValueError('corrections currently require intact-head demonstrations')
        if config['demonstrations_sha256'] != data_hash:
            raise ValueError(f'demonstration dataset differs in {path.parent}')
        source = Path(config['checkpoint'])
        demonstrated = path.parent / 'final_model.zip'
        if not source.is_file() or not demonstrated.is_file():
            raise FileNotFoundError(f'missing source or demonstrated model for {path.parent}')
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if source_hash != config['checkpoint_sha256']:
            raise ValueError(f'source checkpoint changed for {path.parent}')
        source_run = Path(config['source_run'])
        source_config = json.loads((source_run / 'config.json').read_text())
        if (source_config['PPO_settings']['seed'] != config['seed']
                or source_config['training_eval_loop_settings']['eval_seed'] != config['eval_seed']):
            raise ValueError(f'seed settings do not match the source run for {path.parent}')
        runs.append({'history': config['history'], 'checkpoint_kind': config['checkpoint_kind'],
                      'source_run': source_run, 'source': source, 'demonstrated': demonstrated,
                      'seed': config['seed'], 'demonstration_run': path.parent,
                      'demonstration_updates': config['updates'], 'dataset_sha256': data_hash,
                      'original_checkpoint_sha256': source_hash})
    return runs

def run_demonstration_corrections(batch_dir, training_budget):
    runs = build_correction_plan(batch_dir)
    output_dir = Path(batch_dir) / 'ppo_corrections' / f'batch-{datetime.now():%Y%m%d-%H%M%S-%f}'
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'demonstration_batch': str(batch_dir), 'training_budget': training_budget,
                'number_of_runs': len(runs), 'condition': 'demonstrations'}
    (output_dir / 'config.json').write_text(json.dumps(manifest, indent=2))

    # Continue each demonstrated policy with a fresh PPO optimizer.
    for count, run in enumerate(runs, 1):
        lineage = {'demonstration_run': str(run['demonstration_run']),
                   'demonstration_updates': run['demonstration_updates'],
                   'demonstrations_sha256': run['dataset_sha256'],
                   'original_checkpoint': str(run['source']),
                   'original_checkpoint_sha256': run['original_checkpoint_sha256'],
                   'head_condition': 'intact'}
        print(f"\nCorrection {count}/{len(runs)}: {run['history']}, {run['checkpoint_kind']}, "
              f"seed {run['seed']}, demonstrations", flush=True)
        result = run_correction_from_checkpoint(
            run['demonstrated'], run['source_run'], history=run['history'],
            checkpoint_kind=run['checkpoint_kind'], output_dir=output_dir,
            training_budget=training_budget, condition='demonstrations', lineage=lineage,
        )
        print(f'Saved to {result}', flush=True)
    for plot in save_demonstration_correction_plots(output_dir):
        print(f'Saved correction plot: {plot}', flush=True)
    return output_dir

def main():
    args = parse_args()

    if args.correction_batch:
        output = run_demonstration_corrections(args.correction_batch, args.correction_budget)
        print(f'\nCorrections saved to {output}')
        return

    # Find every required checkpoint before starting the batch.
    plan = build_run_plan(args.seeds)

    # Save one shared set of demonstrations for all runs.
    batch_dir = args.output_dir / f'batch-{datetime.now():%Y%m%d-%H%M%S-%f}'
    batch_dir.mkdir(parents=True, exist_ok=False)
    data = make_demonstrations()
    data_hash = save_demonstrations(data, batch_dir / 'demonstrations.json')
    batch_config = {'seed_numbers': args.seeds, 'updates': args.updates, 'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate, 'eval_interval': args.eval_interval,
                    'demonstrations_sha256': data_hash, 'number_of_runs': len(plan)}
    (batch_dir / 'config.json').write_text(json.dumps(batch_config, indent=2))
    # Train and evaluate each source checkpoint.
    results = []
    for index, item in enumerate(plan, 1):
        name = f"{item['history']}-{item['checkpoint_kind']}-{item['head']}-seed-{item['seed']}"
        print(f'\nRun {index}/{len(plan)}: {name}', flush=True)
        metadata = {'history': item['history'], 'checkpoint_kind': item['checkpoint_kind'],
                    'source_run': str(item['source_run']), 'seed_number': item['seed_number'],
                    'demonstrations_sha256': data_hash}
        metrics = run_imitation(item['checkpoint'], data, batch_dir / name, seed=item['seed'],
                                eval_seed=item['eval_seed'], metadata=metadata,
                                head_condition=item['head'], updates=args.updates,
                                batch_size=args.batch_size, learning_rate=args.learning_rate,
                                eval_interval=args.eval_interval)
        results.append({**metadata, 'head': item['head'], 'metrics': metrics})
    # Compare the learning curves after all runs finish.
    save_demonstration_plots(batch_dir, results, HISTORIES, CHECKPOINTS, HEAD_CONDITIONS)
    print(f'\nResults saved to {batch_dir}')
    if not args.skip_correction:
        output = run_demonstration_corrections(batch_dir, args.correction_budget)
        print(f'Corrections saved to {output}')

if __name__ == '__main__':
    main()
