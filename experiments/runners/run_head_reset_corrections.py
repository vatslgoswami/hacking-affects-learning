from pathlib import Path

from experiments.common import find_checkpoint
from experiments.correction import run_correction_from_checkpoint

def main():
    baseline_dir = Path('runs/baseline-models/hacky-variable-beacon')
    output_dir = Path('runs/corrections/hacky-variable-head-reset')
    runs = []

    # Find both source checkpoints for each seed before starting the batch.
    for number in range(1, 8):
        source = baseline_dir / f'run-hacky-variable-seed-#{number}'
        if not (source / 'config.json').is_file():
            raise FileNotFoundError(f'Missing source configuration: {source}')
        for kind in ('reliable', 'final'):
            checkpoint = find_checkpoint(source, kind)
            runs.append((source, kind, checkpoint))

    # Train each reset policy on the genuine-variable task.
    for number, (source, kind, checkpoint) in enumerate(runs, start=1):
        print(f'\nCorrection {number}/{len(runs)}: {source.name}, {kind}', flush=True)
        output = run_correction_from_checkpoint(
            checkpoint, source, history='hacky-variable', checkpoint_kind=kind,
            output_dir=output_dir, condition='fresh-action-head', reset_action_head=True,
        )
        print(f'Saved to {output}', flush=True)

if __name__ == '__main__':
    main()
