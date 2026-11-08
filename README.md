# Reward Hacking and Subsequent Learning

Can learning a reward-hacking shortcut make an agent slower to learn later tasks? This project explores that question in a small gridworld using RL. Agents are trained with PPO using Gymnasium, Stable-Baselines3 and PyTorch. Games are visualizes in PyGame.

## The experiment

An agent moves around a 7×7 board to reach and claim a beacon. We compare three training histories:

- **Fixed:** the beacon stays in one place. Let's call this the *fixed agent*
- **Variable:** the beacon changes location, so the agent must learn to navigate. This is the *variable agent.*
- **Shortcut:** a faulty checker rewards claiming at a fixed 'shortcut ' location (this is hidden from the agent), even when the beacon is somewhere else. This is the *shortcut agent.*

We then fix the checker and see how quickly the agents learn to reach the real beacon. We also give them a new pickup-and-delivery task and compare their learning with a fresh agent.

## Findings

- The shortcut agent was slower to learn newer tasks such as the genuine beacon collection: even when compared to a fixed agent who was also trained on a similar narrow task.
- Model checkpoints towards the end of the training budgets were *less adaptive* than earlier checkpoints, who were already competent at completing their task. A possible reason could be that as networks become more 'sure' of the right policy, they lose the ability to easily adapt to other situations since their logits become more deterministic.
- <u>Interventions:</u> i) Supervised training of agents on valid demonstrations helped them learn the new task.
  ii) Resetting the shortcut agent's action head also improved correction in most runs, suggesting that its old action preferences may have been an obstacle.

These findings come from a small gridworld with an MLP. They do not translate directly to situations with LLMs or larger networks.

## Setup

The experiments were run on CPU with Python 3.12.3. From the project root, create a virtual environment and install the versions recorded for the runs:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Try it

Open the board and move the agent yourself:

```bash
python board_visualizer.py
```

Use the arrow keys to move, `C` to claim, `P` to pick up and `R` to reset. The visualizer also displays controls for replaying saved episodes.

To replay recorded evaluation episodes, download the [trained models](https://huggingface.co/datasets/vatslgoswami/hacking-affects-learning-artifacts/tree/main) and place their `runs/` folder at the project root. For example:

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

The baseline runner currently defaults to **seven full pickup-and-delivery runs**, each with a different seed. Change `task_modes` and `seeds` in `experiments/runners/run_baselines.py` to choose other runs.

## Project files and results

- `rl_envs/`: board mechanics, Gymnasium adapter and task splits.
- `training/`: PPO training, evaluation and the policy network.
- `experiments/`: correction, demonstrations and experiment runners.
- `board_visualizer.py` and `plots.py`: episode playback and learning curves.
- [`final_plots/`](final_plots/): combined learning curves and reproducible per-seed data.
- `tests/`: checks for the environment and experiment pipeline.

The [Hugging Face artifact repository](https://huggingface.co/datasets/vatslgoswami/hacking-affects-learning-artifacts/tree/main) contains baseline runs for all seven seeds and selected follow-up experiments for seed 7. Run configs, evaluation summaries and saved models are included. Training outputs are written to `runs/`.
