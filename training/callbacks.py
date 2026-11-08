from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from typing import cast

from training.evals import evaluate_model
from utils import save_checkpoint, save_evaluation_episodes, save_evaluation_summary

class EntropyScheduleCallback(BaseCallback):
    """Reduce the entropy coefficient over training."""

    def __init__(self, training_budget, initial_coef, final_coef, decay_fraction):
        super().__init__()
        if training_budget <= 0 or not 0 <= final_coef <= initial_coef:
            raise ValueError('entropy coefficients must satisfy 0 <= final <= initial')
        if not 0 < decay_fraction <= 1:
            raise ValueError('entropy decay fraction must be between 0 and 1')
        self.training_budget = training_budget
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.decay_fraction = decay_fraction
        self.start_timesteps = 0

    def _on_training_start(self):
        # Set the starting coefficient before the initial checkpoint is saved.
        self.start_timesteps = self.model.num_timesteps
        cast(PPO, self.model).ent_coef = self.initial_coef

    def _on_rollout_start(self):
        # Gradually reduce the coefficient to the floor.
        elapsed = self.model.num_timesteps - self.start_timesteps
        progress = min(elapsed / (self.training_budget * self.decay_fraction), 1.0)
        coefficient = self.initial_coef + progress * (self.final_coef - self.initial_coef)
        cast(PPO, self.model).ent_coef = coefficient
        self.logger.record('schedule/entropy_coefficient', coefficient)

    def _on_step(self):
        return True

class DeliveryCurriculumCallback(BaseCallback):
    """Decrease assisted reset probability at the start of each rollout."""

    def __init__(self, training_budget, initial_probability=0.5, decay_fraction=0.25, max_distance=1,
                 initial_fraction=0.0):
        super().__init__()
        if training_budget <= 0 or not 0 <= initial_probability <= 1 or not 0 < decay_fraction <= 1:
            raise ValueError('invalid delivery curriculum settings')
        if not 0 <= initial_fraction < 1 or initial_fraction + decay_fraction > 1:
            raise ValueError('initial and decay fractions must fit within the training budget')
        if not isinstance(max_distance, int) or isinstance(max_distance, bool) or max_distance < 0:
            raise ValueError('max_distance must be a nonnegative integer')
        self.training_budget = training_budget
        self.initial_probability = initial_probability
        self.decay_fraction = decay_fraction
        self.initial_fraction = initial_fraction
        self.max_distance = max_distance
        self.start_timesteps = 0

    def _on_training_start(self):
        # Measure the schedule from the start of this training run.
        self.start_timesteps = self.model.num_timesteps

    def _on_rollout_start(self):
        # Start with normal episodes, then introduce and reduce assistance.
        elapsed = self.model.num_timesteps - self.start_timesteps
        assistance_start = self.training_budget * self.initial_fraction
        if elapsed < assistance_start:
            probability = 0.0
        else:
            fraction = min((elapsed - assistance_start) / (self.training_budget * self.decay_fraction), 1.0)
            probability = self.initial_probability * (1.0 - fraction)
        self.training_env.env_method('set_delivery_assistance', probability, self.max_distance)
        self.logger.record('curriculum/assisted_start_probability', probability)

    def _on_step(self):
        return True

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
        self.thresholds = {
            'genuine-fixed': (0.90, 2),
            'genuine-variable': (0.80, 4),
            'pickup-and-deliver': (0.80, 4),
            'hacky-variable': (0.80, 2),
        }.get(test_env.env_type)
        if test_env.env_type == 'hacky-variable' and test_env.goal_split is None:
            self.thresholds = None

    def _passes_competence(self, metrics):
        # Check the task's success and efficiency thresholds.
        if self.thresholds is None:
            raise ValueError('competence thresholds are not defined for this environment')
        success_threshold, extra_limit = self.thresholds
        if self.test_env.env_type == 'hacky-variable':
            extra_actions = metrics['seen_shortcut_average_extra_actions']
            success_rate = metrics['seen_shortcut_success_rate']
        else:
            extra_actions = metrics.get('seen_average_extra_actions', metrics['average_extra_actions'])
            success_rate = metrics.get('seen_success_rate', metrics['success_rate'])
        return (success_rate >= success_threshold
                and extra_actions is not None and extra_actions <= extra_limit)

    def _check_competence(self, steps, metrics):
        # Track consecutive passes and confirm them on another panel.
        passes = None
        first_passing_step = None
        newly_reliable = False
        confirmation = None

        if self.thresholds is not None:
            passes = self._passes_competence(metrics)

            if passes and self.previous_passing_step is not None and self.reliable_step is None:
                # Test the same policy on the second evaluation ('confirmation') panel.
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
                if self.test_env.env_type == 'pickup-and-deliver':
                    # Record any pairs shared by the two panels.
                    def pairs(panel):
                        return {(tuple(item['package']), tuple(item['destination'])) for item in panel}

                    confirmation['overlap_with_monitoring_panel'] = len(
                        pairs(metrics['delivery_panel']) & pairs(confirmation_metrics['delivery_panel']))
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
        if self.test_env.env_type == 'hacky-variable':
            competence_metrics['success_measure'] = 'seen_shortcut_success_rate'
            competence_metrics['efficiency_measure'] = 'seen_shortcut_average_extra_actions'
        return competence_metrics

    def _save_checkpoint_if_needed(self, steps, episodes, competence, *, initial, final):
        # Save models at scheduled steps and at important milestones.
        periodic_checkpoint = self.next_checkpoint is not None and steps >= self.next_checkpoint
        reliable_checkpoint = competence['is_reliable_checkpoint']
        if not (initial or periodic_checkpoint or reliable_checkpoint or final):
            return

        # Save the matching episodes so the checkpoint can be replayed.
        suffix = ('_reliable' if reliable_checkpoint else '') + ('_final' if final else '')
        save_checkpoint(self.model, self.checkpoints_dir, suffix)
        save_evaluation_episodes(self.evals_dir, steps, episodes, self.eval_seed,
                                 self.is_eval_greedy, competence, self.eval_env_seed)
        self.checkpoint_schedule.append(steps)
        while self.next_checkpoint is not None and steps >= self.next_checkpoint:
            self.next_checkpoint += self.checkpoint_interval

    def _log_evaluation(self, steps, metrics, competence):
        # Summarize the evaluation and current competence status.
        success_rate = metrics['success_rate']
        avg_return = metrics['average_return']
        episode_length = metrics['average_episode_length']
        if competence['is_reliable_checkpoint']:
            competence_status = f"confirmed at this evaluation ({steps:,} steps)"
        elif competence['reliable_step'] is not None:
            competence_status = f"previously confirmed at {competence['reliable_step']:,} steps"
        elif competence['passes'] is None:
            competence_status = 'not checked for this task'
        else:
            competence_status = 'not yet confirmed'
        success_label = 'Genuine success rate' if self.test_env.env_type == 'hacky-variable' else 'Success Rate'
        print(f"Evaluation at {steps} transitions:\n{success_label}: {success_rate}\n"
              f"Avg Return: {avg_return}\nEpisode Length: {episode_length}\n"
              f"Extra actions among successes: {metrics['average_extra_actions']}\n"
              f"Competence: {competence_status}\n======================")
        if self.test_env.env_type == 'hacky-variable':
            print(f"Shortcut success rate: {metrics['shortcut_success_rate']}\n"
                  f"Shortcut visit rate: {metrics['shortcut_visit_rate']}\n"
                  f"Total accepted success rate: {metrics['total_success_rate']}")

    def _save_and_evaluate(self, final=False, initial=False):
        # Evaluate the current policy and save its summary.
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
        # Evaluate the policy before it learns from this run.
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
        # Evaluate the final policy after its last update.
        if self.model.num_timesteps != self.evaluation_schedule[-1]:
            self._save_and_evaluate(final=True)
        # SB3 normally dumps the previous update's metrics at the next rollout.
        if 'train/policy_gradient_loss' in self.model.logger.name_to_value:
            self.model.logger.record('train/update_timesteps', self.model.num_timesteps)
            self.model.logger.record('time/total_timesteps', self.model.num_timesteps)
            self.model.logger.dump(step=self.model.num_timesteps)

