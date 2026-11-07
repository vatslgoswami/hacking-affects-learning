from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gym_adapter import GridWorld
import time
from evals import evaluate_model
from policy_architecture import SharedTrunk
from plots import save_evaluation_plot, save_training_plots
from utils import configure_training_logger, create_run_folder, save_checkpoint, save_evaluation_episodes, save_evaluation_summary, save_run_config

class EvaluationCheckpointCallback(BaseCallback):
    def __init__(self, test_env, eval_seed, checkpoints_dir, evals_dir,
                 num_eval_episodes, is_eval_greedy, checkpoint_interval):
        super().__init__()
        self.test_env = test_env
        self.eval_seed = eval_seed
        self.checkpoints_dir = checkpoints_dir
        self.evals_dir = evals_dir
        self.num_eval_episodes = num_eval_episodes
        self.is_eval_greedy = is_eval_greedy
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_schedule = []
        self.next_checkpoint = None

    def _save_and_evaluate(self):
        steps = self.model.num_timesteps
        save_checkpoint(self.model, self.checkpoints_dir)
        metrics, episodes = evaluate_model(self.model, self.test_env, "genuine-fixed",
                                 self.num_eval_episodes, self.eval_seed, self.is_eval_greedy,
                                 record_episodes=True)
        save_evaluation_summary(self.evals_dir, steps, metrics, self.num_eval_episodes,
                                self.eval_seed, self.is_eval_greedy)
        save_evaluation_episodes(self.evals_dir, steps, episodes, self.eval_seed, self.is_eval_greedy)
        self.checkpoint_schedule.append(steps)
        success_rate, avg_return, episode_length = metrics
        print(f"Checkpoint at {steps} transitions:\nSuccess Rate: {success_rate}\n"
              f"Avg Return: {avg_return}\nEpisode Length: {episode_length}\n======================")

    def _on_training_start(self):
        self._save_and_evaluate()
        self.next_checkpoint = self.model.num_timesteps + self.checkpoint_interval

    def _on_rollout_start(self):
        if self.next_checkpoint==None:
            raise ValueError(f"'self.next_checkpoint' cannot be None")
        
        # At this point, the previous rollout's PPO update has finished.
        if 'train/policy_gradient_loss' in self.model.logger.name_to_value:
            self.model.logger.record('train/update_timesteps', self.model.num_timesteps)
        if self.model.num_timesteps >= self.next_checkpoint:
            self._save_and_evaluate()
            while self.next_checkpoint <= self.model.num_timesteps:
                self.next_checkpoint += self.checkpoint_interval

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.model.num_timesteps != self.checkpoint_schedule[-1]:
            self._save_and_evaluate()
        # SB3 normally dumps the previous update's metrics at the next rollout.
        if 'train/policy_gradient_loss' in self.model.logger.name_to_value:
            self.model.logger.record('train/update_timesteps', self.model.num_timesteps)
            self.model.logger.record('time/total_timesteps', self.model.num_timesteps)
            self.model.logger.dump(step=self.model.num_timesteps)


def train_and_eval_loop(model:PPO, test_env, training_budget,
                        eval_seed, checkpoints_dir, evals_dir, num_eval_episodes=25,
                        is_eval_greedy=False, checkpoint_interval=10000):
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    callback = EvaluationCheckpointCallback(test_env, eval_seed, checkpoints_dir, evals_dir,
                                            num_eval_episodes, is_eval_greedy, checkpoint_interval)
    model.learn(training_budget, callback=callback, progress_bar=True, reset_num_timesteps=False)

    train_eval_loop_settings = {
        'eval_episode_count': num_eval_episodes,
        'eval_action_sampler_choice' : "greedy" if is_eval_greedy else "stochastic",
        'eval_seed': eval_seed,
        'training_budget':training_budget,
        'checkpoint_schedule': callback.checkpoint_schedule,
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
    model_seed = 1
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
