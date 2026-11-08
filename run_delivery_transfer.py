import json
from pathlib import Path

from correction import run_correction

def main():
    baseline_dir = Path('runs/baseline-models')
    output_dir = Path('runs/transfers/pickup-and-deliver')
    task_modes = ('hacky-variable', 'genuine-fixed', 'genuine-variable')
    checkpoint_kinds = ('reliable', 'final')

    # Check the source checkpoints and match each seed to its fresh control.
    runs = []
    for seed_number in range(1, 8):
        control = baseline_dir / 'pickup-and-deliver' / f'run-pickup-and-deliver-seed-#{seed_number}'
        with (control / 'config.json').open() as file:
            control_seed = json.load(file)['PPO_settings']['seed']
        for task_mode in task_modes:
            source_run = baseline_dir / f'{task_mode}-beacon' / f'run-{task_mode}-seed-#{seed_number}'
            with (source_run / 'config.json').open() as file:
                config = json.load(file)
            if config['env_type'] != task_mode or config['PPO_settings']['seed'] != control_seed:
                raise ValueError(f'Source task or seed does not match the control: {source_run}')
            for checkpoint_kind in checkpoint_kinds:
                checkpoints = list((source_run / 'checkpoints').glob(f'step_*_{checkpoint_kind}*.zip'))
                if len(checkpoints) != 1:
                    raise ValueError(f'Expected one {checkpoint_kind} checkpoint in {source_run}')
                runs.append((source_run, checkpoint_kind))

    # Train each source policy on pickup and delivery.
    for number, (source_run, checkpoint_kind) in enumerate(runs, start=1):
        print(f'\nDelivery transfer {number}/{len(runs)}: {source_run.name}, {checkpoint_kind}', flush=True)
        output = run_correction(source_run, checkpoint_kind=checkpoint_kind,
                                target_task='pickup-and-deliver', output_dir=output_dir)
        print(f'Saved to {output}', flush=True)

if __name__ == '__main__':
    main()
