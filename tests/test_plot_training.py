import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from plots import create_training_figures, load_training_diagnostics, save_training_plots


class TestTrainingPlots(unittest.TestCase):
    def test_plots_use_update_counts_and_saved_losses(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            fields = ['time/total_timesteps', 'train/update_timesteps',
                      'train/policy_gradient_loss', 'train/value_loss', 'train/entropy_loss']
            with (run_dir / 'progress.csv').open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerow([8, '', '', '', ''])
                writer.writerow([24, 16, -0.2, 0.4, -1.1])
                writer.writerow([16, 8, -0.1, 0.7, -1.5])

            metrics = load_training_diagnostics(run_dir)
            losses, entropy = create_training_figures(metrics, 'test')
            try:
                self.assertEqual(list(losses.axes[0].lines[0].get_xdata()), [8, 16])
                self.assertEqual(list(losses.axes[0].lines[0].get_ydata()), [-0.1, -0.2])
                self.assertEqual(list(losses.axes[0].lines[1].get_ydata()), [0.7, 0.4])
                self.assertEqual(list(entropy.axes[0].lines[0].get_xdata()), [8, 16])
                self.assertEqual(list(entropy.axes[0].lines[0].get_ydata()), [-1.5, -1.1])
            finally:
                plt.close(losses)
                plt.close(entropy)

            paths = save_training_plots(run_dir)
            self.assertEqual([path.name for path in paths], ['policy_value_loss.png', 'entropy_loss.png'])
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))

    def test_missing_metric_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'progress.csv').write_text('train/update_timesteps\n8\n')
            with self.assertRaisesRegex(ValueError, 'No recorded values for train/policy_gradient_loss'):
                load_training_diagnostics(directory)


if __name__ == '__main__':
    unittest.main(verbosity=2)
