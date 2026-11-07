import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from plots import create_evaluation_figure, load_evaluation_summaries, save_evaluation_plot


class TestEvaluationPlots(unittest.TestCase):
    def test_plot_uses_saved_values_in_timestep_order(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            evaluations = run_dir / 'evaluations'
            evaluations.mkdir()
            for step, success, avg_return, length in (
                (10240, 0.8, 0.7, 7.0),
                (0, 0.04, -0.2, 29.0),
            ):
                summary = {
                    'n_timesteps': step,
                    'success_rate': success,
                    'average_return': avg_return,
                    'average_episode_length': length,
                    'eval_episode_count': 25,
                    'eval_seed': 100,
                    'eval_action_sampler_choice': 'stochastic',
                }
                (evaluations / f'step_{step}.json').write_text(json.dumps(summary))

            summaries = load_evaluation_summaries(run_dir)
            figure = create_evaluation_figure(summaries, 'test')
            try:
                self.assertEqual(list(figure.axes[0].lines[0].get_xdata()), [0, 10240])
                self.assertEqual(list(figure.axes[0].lines[0].get_ydata()), [0.04, 0.8])
                self.assertEqual(list(figure.axes[1].lines[0].get_ydata()), [-0.2, 0.7])
                self.assertEqual(list(figure.axes[2].lines[0].get_ydata()), [29.0, 7.0])
            finally:
                plt.close(figure)

            output = save_evaluation_plot(run_dir)
            self.assertEqual(output, run_dir / 'plots' / 'learning_curves.png')
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_missing_summaries_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'No evaluation summaries'):
                load_evaluation_summaries(directory)


if __name__ == '__main__':
    unittest.main(verbosity=2)
