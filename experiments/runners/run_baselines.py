import time

from stable_baselines3 import PPO

from rl_envs.environment import EnvType
from rl_envs.gym_adapter import GridWorld
from rl_envs.task_splits import development_delivery_split, development_goal_split
from experiments.settings import DELIVERY_BUDGET, DELIVERY_CURRICULUM, ENTROPY_SCHEDULE, SOURCE_BUDGET
from plots import save_evaluation_plot, save_training_plots
from training.mlp_architecture import SharedTrunk
from training.train import get_run_settings, train_and_eval_loop
from utils import configure_training_logger, create_run_folder, save_run_config

def run_baseline(task_mode: EnvType, model_seed, eval_seed):
    # Create training and evaluation environments with the same task split.
    training_budget = DELIVERY_BUDGET if task_mode == 'pickup-and-deliver' else SOURCE_BUDGET
    delivery_curriculum = DELIVERY_CURRICULUM.copy() if task_mode == 'pickup-and-deliver' else None
    entropy_schedule = ENTROPY_SCHEDULE.copy()
    goal_split = development_goal_split() if task_mode in ('genuine-variable', 'hacky-variable') else None
    delivery_split = development_delivery_split() if task_mode == 'pickup-and-deliver' else None
    env_args = (7, [3, 3], [3, 0], 30, task_mode)
    train_env = GridWorld(*env_args, goal_split=goal_split, delivery_split=delivery_split)
    test_env = GridWorld(*env_args, goal_split=goal_split, delivery_split=delivery_split)

    # Start a fresh policy with the same PPO settings as the original baseline runs.
    model = PPO(
        'MlpPolicy', train_env, gamma=1, verbose=2, device='cpu', seed=model_seed,
        ent_coef=entropy_schedule['initial_coef'],
        policy_kwargs={'features_extractor_class': SharedTrunk, 'net_arch': dict(pi=[], vf=[])},
    )
    run_id = f'{train_env.env_type}-{time.ctime()}'
    run_dir, checkpoints_dir, evals_dir, _ = create_run_folder('./runs', run_id)
    logger = configure_training_logger(model, run_dir)
    try:
        # Train while the callbacks evaluate the policy and save its checkpoints.
        model, loop_settings = train_and_eval_loop(
            model, test_env, training_budget, eval_seed, checkpoints_dir, evals_dir,
            delivery_curriculum=delivery_curriculum, entropy_schedule=entropy_schedule,
        )
        settings = get_run_settings(model, train_env, loop_settings)
        save_run_config(run_dir, settings, run_id)
        print(f'Saved evaluation plot: {save_evaluation_plot(run_dir)}')
        for plot_path in save_training_plots(run_dir):
            print(f'Saved training plot: {plot_path}')
    finally:
        logger.close()
        train_env.close()
        test_env.close()
    return run_dir

def main():
    # Run the same frozen seed pairs and default task as the original baseline script.
    seeds = [(7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20)]
    task_modes: list[EnvType] = ['pickup-and-deliver']
    for task_mode in task_modes:
        for model_seed, eval_seed in seeds:
            run_baseline(task_mode, model_seed, eval_seed)

if __name__ == '__main__':
    main()
