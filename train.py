from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gym_adapter import GridWorld
import time
from evals import evaluate_model
from mlp_architecture import SharedTrunk
from plots import save_evaluation_plot, save_training_plots
from utils import configure_training_logger, create_run_folder, save_checkpoint, save_evaluation_episodes, save_evaluation_summary, save_run_config

class EvaluationCheckpointCallback(BaseCallback):
    def __init__(self, test_env, eval_seed, checkpoints_dir, evals_dir,
                 num_eval_episodes=25, is_eval_greedy=False, checkpoint_interval=10_000,
                 eval_env_seed=10, confirmation_seed=11, confirmation_env_seed=12):
        super().__init__()
        self.test_env = test_env
        self.eval_seed = eval_seed
        self.eval_env_seed = eval_env_seed
        self.confirmation_seed = confirmation_seed
        self.confirmation_env_seed = confirmation_env_seed
        self.num_eval_episodes = num_eval_episodes
        self.is_eval_greedy = is_eval_greedy

        self.checkpoints_dir = checkpoints_dir
        self.evals_dir = evals_dir
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_schedule = []
        self.evaluation_schedule = []
        self.next_checkpoint = None

        self.previous_passing_step = None
        self.reliable_step = None
        self.first_passing_step = None
        # Shortcut and delivery success need their task-specific labels first.
        self.thresholds = {
            'genuine-fixed': (0.90, 2),
            'genuine-variable': (0.80, 4),
        }.get(test_env.env_type)

    def _passes_competence(self, metrics):
        if self.thresholds is None:
            raise ValueError('competence thresholds are not defined for this environment')
        success_threshold, extra_limit = self.thresholds
        extra_actions = metrics['average_extra_actions']
        return (metrics['success_rate'] >= success_threshold
                and extra_actions is not None and extra_actions <= extra_limit)

    def _check_competence(self, steps, metrics):
        passes = None
        first_passing_step = None
        newly_reliable = False
        confirmation = None

        if self.thresholds is not None:
            passes = self._passes_competence(metrics)

            if passes and self.previous_passing_step is not None and self.reliable_step is None:
                confirmation_metrics, _ = evaluate_model(
                    self.model, self.test_env, self.test_env.env_type,
                    self.num_eval_episodes, self.confirmation_seed, self.is_eval_greedy,
                    record_episodes=False, eval_env_seed=self.confirmation_env_seed,
                )
                confirmation_passes = self._passes_competence(confirmation_metrics)
                confirmation = {
                    'eval_seed': self.confirmation_seed,
                    'eval_env_seed': self.confirmation_env_seed,
                    'eval_episode_count': self.num_eval_episodes,
                    'eval_action_sampler_choice': 'greedy' if self.is_eval_greedy else 'stochastic',
                    'metrics': confirmation_metrics,
                    'passes': confirmation_passes,
                }
                if confirmation_passes:
                    first_passing_step = self.previous_passing_step
                    self.first_passing_step = first_passing_step
                    self.reliable_step = steps
                    newly_reliable = True

            self.previous_passing_step = steps if passes else None

        competence_metrics = {
            'success_threshold': self.thresholds[0] if self.thresholds else None,
            'extra_actions_limit': self.thresholds[1] if self.thresholds else None,
            'passes': passes,
            'first_passing_step': first_passing_step,
            'reliable_step': self.reliable_step,
            'is_reliable_checkpoint': newly_reliable,
            'confirmation': confirmation,
        }
        return competence_metrics

    def _save_checkpoint_if_needed(self, steps, episodes, competence, *, initial, final):
        periodic_checkpoint = self.next_checkpoint is not None and steps >= self.next_checkpoint
        reliable_checkpoint = competence['is_reliable_checkpoint']
        if not (initial or periodic_checkpoint or reliable_checkpoint or final):
            return

        suffix = ('_reliable' if reliable_checkpoint else '') + ('_final' if final else '')
        save_checkpoint(self.model, self.checkpoints_dir, suffix)
        save_evaluation_episodes(self.evals_dir, steps, episodes, self.eval_seed,
                                 self.is_eval_greedy, competence, self.eval_env_seed)
        self.checkpoint_schedule.append(steps)
        while self.next_checkpoint is not None and steps >= self.next_checkpoint:
            self.next_checkpoint += self.checkpoint_interval

    def _log_evaluation(self, steps, metrics, competence):
        success_rate = metrics['success_rate']
        avg_return = metrics['average_return']
        episode_length = metrics['average_episode_length']
        print(f"Evaluation at {steps} transitions:\nSuccess Rate: {success_rate}\n"
              f"Avg Return: {avg_return}\nEpisode Length: {episode_length}\n"
              f"Extra actions among successes: {metrics['average_extra_actions']}\n"
              f"Competence confirmed here: {competence['is_reliable_checkpoint']}\n======================")

    def _save_and_evaluate(self, final=False, initial=False):
        steps = self.model.num_timesteps
        metrics, episodes = evaluate_model(
            self.model, self.test_env, self.test_env.env_type,
            self.num_eval_episodes, self.eval_seed, self.is_eval_greedy,
            record_episodes=True, eval_env_seed=self.eval_env_seed,
        )
        competence = self._check_competence(steps, metrics)
        save_evaluation_summary(self.evals_dir, steps, metrics, self.num_eval_episodes,
                                self.eval_seed, self.is_eval_greedy, competence, self.eval_env_seed)
        self._save_checkpoint_if_needed(steps, episodes, competence, initial=initial, final=final)
        self.evaluation_schedule.append(steps)
        self._log_evaluation(steps, metrics, competence)

    def _on_training_start(self):
        self.next_checkpoint = self.model.num_timesteps + self.checkpoint_interval
        self._save_and_evaluate(initial=True)

    def _on_rollout_start(self):
        # At this point, the previous rollout's PPO update has finished.
        if 'train/policy_gradient_loss' in self.model.logger.name_to_value:
            self.model.logger.record('train/update_timesteps', self.model.num_timesteps)
        if self.model.num_timesteps != self.evaluation_schedule[-1]:
            self._save_and_evaluate()

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.model.num_timesteps != self.evaluation_schedule[-1]:
            self._save_and_evaluate(final=True)
        # SB3 normally dumps the previous update's metrics at the next rollout.
        if 'train/policy_gradient_loss' in self.model.logger.name_to_value:
            self.model.logger.record('train/update_timesteps', self.model.num_timesteps)
            self.model.logger.record('time/total_timesteps', self.model.num_timesteps)
            self.model.logger.dump(step=self.model.num_timesteps)

def train_and_eval_loop(model:PPO, test_env, training_budget,
                        eval_seed, checkpoints_dir, evals_dir, num_eval_episodes=25,
                        is_eval_greedy=False, checkpoint_interval=10_000, eval_env_seed=10,
                        confirmation_seed=11, confirmation_env_seed=12):
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    callback = EvaluationCheckpointCallback(test_env, eval_seed, checkpoints_dir, evals_dir,
                                            num_eval_episodes, is_eval_greedy, checkpoint_interval, eval_env_seed,
                                            confirmation_seed, confirmation_env_seed)
    model.learn(training_budget, callback=callback, progress_bar=True, reset_num_timesteps=False)

    train_eval_loop_settings = {
        'eval_episode_count': num_eval_episodes,
        'eval_action_sampler_choice' : "greedy" if is_eval_greedy else "stochastic",
        'eval_seed': eval_seed,
        'eval_env_seed': eval_env_seed,
        'confirmation_seed': confirmation_seed,
        'confirmation_env_seed': confirmation_env_seed,
        'training_budget':training_budget,
        'checkpoint_interval': checkpoint_interval,
        'evaluation_interval': model.n_steps * model.n_envs,
        'checkpoint_schedule': callback.checkpoint_schedule,
        'evaluation_schedule': callback.evaluation_schedule,
        'reliable_step': callback.reliable_step,
        'first_passing_step': callback.first_passing_step,
        'competence_thresholds': {
            'success_rate': callback.thresholds[0], 'average_extra_actions': callback.thresholds[1]
        } if callback.thresholds else None,
    }
    return model, train_eval_loop_settings

def get_model_architecture(model:PPO):
    return str(model.policy)

def get_model_ppo_settings(model:PPO):
    ppo_settings = {
        'rollout_size': model.n_steps * model.n_envs,
        'learning_rate': model.learning_rate,
        'gamma': model.gamma,
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

if __name__=="__main__":
    model_seed = 3
    eval_seed = 100
    train_env = GridWorld(7, [3,3], [3,0], 30, "genuine-fixed")
    test_env = GridWorld(7, [3,3], [3,0], 30, "genuine-fixed") 

    model = PPO(
        "MlpPolicy", 
        train_env, 
        gamma=1,
        verbose=2,
        device="cpu",
        seed=model_seed,
        policy_kwargs={
            'features_extractor_class': SharedTrunk,
            'net_arch': dict(pi=[], vf = []),
        }
    )

    run_id = time.ctime()
    BASE_DIR = './runs'
    run_dir, checkpoints_dir, evals_dir, plots_dir = create_run_folder(BASE_DIR, run_id)
    logger = configure_training_logger(model, run_dir)
    #run the train_eval_loop
    try:
        model, train_eval_loop_settings = train_and_eval_loop(model, test_env, 20000, eval_seed, checkpoints_dir, evals_dir)
        settings = get_run_settings(model, train_env, train_eval_loop_settings)
        save_run_config(run_dir, settings, run_id)
        print(f'Saved evaluation plot: {save_evaluation_plot(run_dir)}')
        for plot_path in save_training_plots(run_dir):
            print(f'Saved training plot: {plot_path}')
    finally:
        logger.close()
