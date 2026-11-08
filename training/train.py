from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from rl_envs.gym_adapter import GridWorld
from training.callbacks import DeliveryCurriculumCallback, EntropyScheduleCallback, EvaluationCheckpointCallback

def _check_training_environments(model, test_env):
    # Training and evaluation must come from the same 'seen'/'witheld' or train/test split.
    training_env = model.get_env()
    if training_env is None or any(split != test_env.goal_split for split in training_env.get_attr('goal_split')):
        raise ValueError('training and evaluation environments must use the same goal split')
    if any(split != test_env.delivery_split for split in training_env.get_attr('delivery_split')):
        raise ValueError('training and evaluation environments must use the same delivery split')
    if test_env.env_type == 'hacky-variable' and test_env.goal_split is None:
        raise ValueError('shortcut training requires a goal split for competence evaluation')

def train_and_eval_loop(model:PPO, test_env, training_budget,
                        eval_seed, checkpoints_dir, evals_dir, num_eval_episodes=25,
                        is_eval_greedy=False, checkpoint_interval=10_000, eval_env_seed=10,
                        confirmation_seed=11, confirmation_env_seed=12, delivery_curriculum=None,
                        entropy_schedule=None, reset_num_timesteps=False):
    # Check settings before starting training.
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    _check_training_environments(model, test_env)

    # Set up evaluation and optional training schedules.
    callback = EvaluationCheckpointCallback(test_env, eval_seed, checkpoints_dir, evals_dir,
                                            num_eval_episodes, is_eval_greedy, checkpoint_interval, eval_env_seed,
                                            confirmation_seed, confirmation_env_seed)
    callbacks: list[BaseCallback] = []
    if entropy_schedule is not None:
        callbacks.append(EntropyScheduleCallback(training_budget, **entropy_schedule))
    callbacks.append(callback)
    if delivery_curriculum is not None:
        if test_env.env_type != 'pickup-and-deliver':
            raise ValueError('delivery curriculum requires a delivery environment')
        callbacks.append(DeliveryCurriculumCallback(training_budget, **delivery_curriculum))
    # Train at once: the callbacks handle evaluations, checkpointing and assisted curriculum.
    model.learn(training_budget, callback=CallbackList(callbacks), progress_bar=True,
                reset_num_timesteps=reset_num_timesteps)

    # Record the settings and checkpoint steps for the run config.
    competence_thresholds: dict[str, float | str] | None = None
    if callback.thresholds:
        competence_thresholds = {
            'success_rate': callback.thresholds[0], 'average_extra_actions': callback.thresholds[1]
        }
        if test_env.env_type == 'hacky-variable':
            competence_thresholds['success_measure'] = 'seen_shortcut_success_rate'
            competence_thresholds['efficiency_measure'] = 'seen_shortcut_average_extra_actions'
    train_eval_loop_settings = {
        'eval_episode_count': num_eval_episodes,
        'eval_action_sampler_choice' : "greedy" if is_eval_greedy else "stochastic",
        'eval_seed': eval_seed,
        'eval_env_seed': eval_env_seed,
        'confirmation_seed': confirmation_seed,
        'confirmation_env_seed': confirmation_env_seed,
        'training_budget':training_budget,
        'delivery_curriculum': delivery_curriculum,
        'entropy_schedule': entropy_schedule,
        'checkpoint_interval': checkpoint_interval,
        'evaluation_interval': model.n_steps * model.n_envs,
        'checkpoint_schedule': callback.checkpoint_schedule,
        'evaluation_schedule': callback.evaluation_schedule,
        'reliable_step': callback.reliable_step,
        'first_passing_step': callback.first_passing_step,
        'competence_thresholds': competence_thresholds,
    }
    return model, train_eval_loop_settings

def get_model_architecture(model:PPO):
    return str(model.policy)

def get_model_ppo_settings(model:PPO):
    ppo_settings = {
        'rollout_size': model.n_steps * model.n_envs,
        'learning_rate': model.learning_rate,
        'gamma': model.gamma,
        'ent_coef': model.ent_coef,
        'seed': model.seed,
        'n_timesteps': model.num_timesteps,  
    }
    return ppo_settings

def get_run_settings(model:PPO, env:GridWorld, train_eval_loop_settings):
    env_settings = env.get_env_settings()
    ppo_settings = get_model_ppo_settings(model)
    model_settings = {
        'architecture': get_model_architecture(model),
        'PPO_settings': ppo_settings,
        'training_eval_loop_settings': train_eval_loop_settings
        }

    return env_settings | model_settings
