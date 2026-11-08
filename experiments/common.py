import json
from pathlib import Path

def load_source_config(source_run):
    # Read the settings saved with a source policy.
    with (Path(source_run) / 'config.json').open() as file:
        return json.load(file)

def find_checkpoint(source_run, checkpoint_kind):
    # Require one matching checkpoint so a batch cannot silently use the wrong model.
    checkpoints_dir = Path(source_run) / 'checkpoints'
    matches = ([checkpoints_dir / 'step_0.zip'] if checkpoint_kind == 'initial'
               else list(checkpoints_dir.glob(f'step_*_{checkpoint_kind}*.zip')))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f'Expected one {checkpoint_kind} checkpoint in {checkpoints_dir}')
    return matches[0]
