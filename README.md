# Reward Hacking and Subsequent Learning

Can learning a reward-hacking shortcut make an agent slower to learn later? This project explores that question in a small gridworld using reinforcement learning. Agents are trained with PPO using Gymnasium, Stable-Baselines3 and PyTorch.

## The experiment

An agent moves around a 7×7 board to reach and claim a beacon. We compare three training histories:

- **Fixed:** the beacon stays in one place.
- **Variable:** the beacon changes location, so the agent must learn to navigate.
- **Shortcut:** a faulty checker rewards claiming at a fixed location, even when the beacon is somewhere else.

We then fix the checker and see how quickly the agents learn to reach the real beacon. We also give them a new pickup-and-delivery task and compare their learning with a fresh agent.

## Findings

- The shortcut agent generally learned the corrected task more slowly than the fixed-goal agent. It also struggled with pickup and delivery, while the variable-goal agent generally learned it.
- Later source checkpoints often adapted more slowly than earlier checkpoints that had already learned their original task. This shows a difference in adaptability, but does not establish why it happened.
- Demonstrations helped agents learn the corrected task. Resetting the shortcut agent's action head also improved correction in most runs, suggesting that its old action preferences may have been an obstacle.

These findings come from a small gridworld. They do not establish how reward hacking affects learning in language models.

## Setup

The experiments were run on CPU with Python 3.12.3. From the project root, create a virtual environment and install the versions recorded for the runs:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.5.4 torch==2.15.0 gymnasium==1.4.0 \
  stable-baselines3==2.9.2 matplotlib==3.11.2 pygame==2.6.1 rich==15.0.0 tqdm==4.70.1
```

## Try it

Open the board and move the agent yourself:

```bash
python board_visualizer.py
```

Use the arrow keys to move, `C` to claim, `P` to pick up and `R` to reset. The visualizer also displays controls for replaying saved episodes.

To replay recorded evaluation episodes, download the [experiment artifacts](https://huggingface.co/datasets/vatslgoswami/hacking-affects-learning-artifacts/tree/main) and place their `runs/` folder at the project root. For example:

```bash
python board_visualizer.py --run 'runs/baseline-models/hacky-variable-beacon/run-hacky-variable-seed-#1' --all-checkpoints
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

To train baselines:

```bash
python -m experiments.runners.run_baselines
```

The baseline runner currently defaults to **seven full pickup-and-delivery runs**, one per seed. Change `task_modes` and `seeds` in `experiments/runners/run_baselines.py` to choose other runs.

## Project files and results

- `rl_envs/`: board mechanics, Gymnasium adapter and task splits.
- `training/`: PPO training, evaluation and the policy network.
- `experiments/`: correction, demonstrations and experiment runners.
- `board_visualizer.py` and `plots.py`: episode playback and learning curves.
- [`final_plots/`](final_plots/): combined learning curves and reproducible per-seed data.
- `tests/`: checks for the environment and experiment pipeline.

The [Hugging Face artifact repository](https://huggingface.co/datasets/vatslgoswami/hacking-affects-learning-artifacts/tree/main) contains baseline runs for all seven seeds and selected follow-up runs for seed 7. Run configs, evaluation summaries and saved models are included. Training outputs are written to `runs/`.
