import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from experiments.common import find_checkpoint, load_source_config
from experiments.correction import run_correction_from_checkpoint
from experiments.demonstrations import make_delivery_demonstrations, run_imitation, save_demonstrations
from experiments.settings import DELIVERY_BUDGET, DELIVERY_CURRICULUM, ENTROPY_SCHEDULE
from plots import save_delivery_demonstration_comparisons, save_demonstration_plots

BASELINE_DIR = Path('runs/baseline-models')
TRANSFER_DIR = Path('runs/transfers/pickup-and-deliver')
HISTORIES = ('hacky-variable', 'genuine-fixed')
CHECKPOINTS = ('reliable', 'final')

def build_run_plan(seed_numbers):
    # Match every source checkpoint to its existing no-demonstration transfer.
    transfers = {}
    for path in TRANSFER_DIR.glob('*/config.json'):
        config = json.loads(path.read_text())
        source = config.get('source_checkpoint', {})
        if source.get('history') not in HISTORIES or source.get('condition') != 'source':
            continue
        key = (source['history'], source['checkpoint_kind'], config['PPO_settings']['seed'])
        transfers.setdefault(key, []).append((path.parent, config))

    # Verify each source checkpoint and its matched transfer used the same settings.
    runs = []
    for number in seed_numbers:
        for history in HISTORIES:
            source_run = BASELINE_DIR / f'{history}-beacon' / f'run-{history}-seed-#{number}'
            config = load_source_config(source_run)
            seed = config['PPO_settings']['seed']
            if config['env_type'] != history:
                raise ValueError(f'incorrect source history in {source_run}')
            for kind in CHECKPOINTS:
                checkpoint = find_checkpoint(source_run, kind)
                source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                matches = transfers.get((history, kind, seed), [])
                if len(matches) != 1:
                    raise ValueError(f'expected one matching original transfer for {history}, {kind}, seed {seed}')
                transfer_dir, transfer_config = matches[0]
                settings = transfer_config['training_eval_loop_settings']
                source_eval = config['training_eval_loop_settings']
                if (transfer_config['source_checkpoint']['sha256'] != source_hash
                        or transfer_config['env_type'] != 'pickup-and-deliver'
                        or settings['training_budget'] != DELIVERY_BUDGET
                        or settings['delivery_curriculum'] != DELIVERY_CURRICULUM
                        or settings['entropy_schedule'] != ENTROPY_SCHEDULE
                        or any(settings[name] != source_eval[name] for name in (
                            'eval_seed', 'eval_env_seed', 'eval_episode_count',
                            'eval_action_sampler_choice', 'confirmation_seed', 'confirmation_env_seed'))):
                    raise ValueError(f'original transfer settings do not match {source_run}')
                runs.append({'history': history, 'checkpoint_kind': kind, 'seed_number': number,
                             'seed': seed, 'eval_seed': source_eval['eval_seed'],
                             'source_run': source_run, 'checkpoint': checkpoint,
                             'source_hash': source_hash, 'without': transfer_dir})
    return runs

def run_delivery_imitation(plan, batch_dir, *, num_episodes=100, updates=1000,
                           batch_size=64, learning_rate=0.0003, eval_interval=100):
    # Save one shared dataset before any model sees the examples.
    batch_dir.mkdir(parents=True, exist_ok=False)
    data = make_delivery_demonstrations(num_episodes=num_episodes)
    data_hash = save_demonstrations(data, batch_dir / 'demonstrations.json')
    config = {'task': 'pickup-and-deliver', 'demonstration_stage': 'after-pickup',
              'training_episodes': num_episodes, 'heldout_episodes': num_episodes,
              'dataset_seed': 2026, 'demonstrations_sha256': data_hash,
              'updates': updates, 'batch_size': batch_size, 'learning_rate': learning_rate,
              'eval_interval': eval_interval, 'number_of_runs': len(plan),
              'ppo_training_budget': DELIVERY_BUDGET}
    (batch_dir / 'config.json').write_text(json.dumps(config, indent=2))

    # Give each policy the same examples and update schedule within its seed block.
    results = []
    for number, run in enumerate(plan, 1):
        name = f"{run['history']}-{run['checkpoint_kind']}-intact-seed-{run['seed']}"
        print(f'\nImitation {number}/{len(plan)}: {name}', flush=True)
        metadata = {'history': run['history'], 'checkpoint_kind': run['checkpoint_kind'],
                    'source_run': str(run['source_run']), 'seed_number': run['seed_number'],
                    'demonstrations_sha256': data_hash, 'original_transfer': str(run['without'])}
        metrics = run_imitation(run['checkpoint'], data, batch_dir / name, seed=run['seed'],
                                eval_seed=run['eval_seed'], metadata=metadata, updates=updates,
                                batch_size=batch_size, learning_rate=learning_rate, eval_interval=eval_interval,
                                target_task='pickup-and-deliver')
        results.append({**metadata, 'head': 'intact', 'metrics': metrics})
    save_demonstration_plots(batch_dir, results, HISTORIES, CHECKPOINTS, ('intact',),
                             target_task='pickup-and-deliver')

def run_delivery_ppo(plan, batch_dir):
    # Check that every saved imitation model matches its source and shared dataset.
    batch_config = json.loads((batch_dir / 'config.json').read_text())
    data_hash = hashlib.sha256((batch_dir / 'demonstrations.json').read_bytes()).hexdigest()
    if (batch_config['task'] != 'pickup-and-deliver'
            or batch_config['demonstrations_sha256'] != data_hash
            or batch_config['number_of_runs'] != len(plan)):
        raise ValueError('demonstration batch does not match the requested runs')
    for run in plan:
        name = f"{run['history']}-{run['checkpoint_kind']}-intact-seed-{run['seed']}"
        demo_dir = batch_dir / name
        config = json.loads((demo_dir / 'config.json').read_text())
        if (config['checkpoint_sha256'] != run['source_hash']
                or config['demonstrations_sha256'] != data_hash
                or config['target_task'] != 'pickup-and-deliver'
                or config['head_condition'] != 'intact'
                or config['seed'] != run['seed'] or config['eval_seed'] != run['eval_seed']
                or not (demo_dir / 'final_model.zip').is_file()):
            raise ValueError(f'imitation model does not match {name}')

    # Create a separate batch for the follow-up PPO runs and their comparisons.
    output_dir = batch_dir / 'ppo_transfer' / f'batch-{datetime.now():%Y%m%d-%H%M%S-%f}'
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'config.json').write_text(json.dumps({
        'task': 'pickup-and-deliver', 'training_budget': DELIVERY_BUDGET,
        'demonstration_batch': str(batch_dir), 'number_of_runs': len(plan),
        'demonstrations_sha256': data_hash,
    }, indent=2))
    # Continue each demonstrated model and pair its results with the original transfer.
    comparisons = []
    for number, run in enumerate(plan, 1):
        name = f"{run['history']}-{run['checkpoint_kind']}-intact-seed-{run['seed']}"
        demo_dir = batch_dir / name
        print(f'\nDelivery PPO {number}/{len(plan)}: {name}', flush=True)
        lineage = {'demonstration_run': str(demo_dir), 'demonstration_updates': batch_config['updates'],
                   'demonstration_episodes': batch_config['training_episodes'],
                   'demonstrations_sha256': data_hash, 'original_checkpoint': str(run['checkpoint']),
                   'original_checkpoint_sha256': run['source_hash'], 'original_transfer': str(run['without']),
                   'head_condition': 'intact'}
        output = run_correction_from_checkpoint(
            demo_dir / 'final_model.zip', run['source_run'], history=run['history'],
            checkpoint_kind=run['checkpoint_kind'], output_dir=output_dir,
            target_task='pickup-and-deliver', condition='demonstrations', lineage=lineage,
        )
        comparisons.append({'history': run['history'], 'checkpoint_kind': run['checkpoint_kind'],
                            'seed': run['seed'], 'without': run['without'], 'with': output})
    for path in save_delivery_demonstration_comparisons(output_dir, comparisons):
        print(f'Saved comparison plot: {path}', flush=True)
    return output_dir

def main():
    # Parse the batch settings and select new imitation runs or an existing batch.
    parser = argparse.ArgumentParser(description='Teach the delivery stage, then run PPO.')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 8)),
                        help='Baseline folder labels (default: 1 through 7)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training demonstrations')
    parser.add_argument('--updates', type=int, default=1000, help='Imitation updates per model')
    parser.add_argument('--imitation-only', action='store_true', help='Stop before the PPO follow-up')
    parser.add_argument('--ppo-from-batch', type=Path, help='Use an existing complete imitation batch')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/demonstrations/delivery'))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.updates < 1:
        raise ValueError('seed labels must be unique and updates must be positive')
    if args.imitation_only and args.ppo_from_batch:
        raise ValueError('cannot combine imitation-only with ppo-from-batch')
    plan = build_run_plan(args.seeds)
    if args.ppo_from_batch:
        batch_dir = args.ppo_from_batch
    else:
        batch_dir = args.output_dir / f'batch-{datetime.now():%Y%m%d-%H%M%S-%f}'
        run_delivery_imitation(plan, batch_dir, num_episodes=args.episodes, updates=args.updates)
    if not args.imitation_only:
        print(f'PPO results saved to {run_delivery_ppo(plan, batch_dir)}')
    print(f'Demonstration batch saved to {batch_dir}')

if __name__ == '__main__':
    main()
