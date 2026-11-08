from pathlib import Path

from experiments.correction import run_correction
from experiments.common import find_checkpoint

def main():
    baseline_dir = Path('runs/baseline-models')
    task_modes = ('hacky-variable', 'genuine-fixed')
    checkpoint_kinds = ('reliable', 'final')

    # Check that every source run and checkpoint exists before starting training.
    runs = []
    for task_mode in task_modes:
        for seed_number in [6, 7]:
            source_run = baseline_dir / f'{task_mode}-beacon' / f'run-{task_mode}-seed-#{seed_number}'
            if not (source_run / 'config.json').is_file():
                raise FileNotFoundError(f'Missing source configuration: {source_run}')
            for checkpoint_kind in checkpoint_kinds:
                find_checkpoint(source_run, checkpoint_kind)
                runs.append((source_run, checkpoint_kind))

    # Train one correction run for each source checkpoint.
    for number, (source_run, checkpoint_kind) in enumerate(runs, start=1):
        print(f'\nCorrection {number}/{len(runs)}: {source_run.name}, {checkpoint_kind}', flush=True)
        output = run_correction(source_run, checkpoint_kind=checkpoint_kind)
        print(f'Saved to {output}', flush=True)

if __name__ == '__main__':
    main()
