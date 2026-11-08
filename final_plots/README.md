# Combined experiment plots

Each figure averages one experiment condition across model seeds. Shading is ±1 population standard deviation across those seeds, not a confidence interval. Success and episode-length shading is clipped to the task’s possible range. All selected seeds, including those that never reached competence, are included in every curve.

The competence count uses each run’s saved confirmation step. The dotted line and mean are calculated only among seeds that reached competence; they are not the point where the average curve crosses a threshold. No line is drawn when none reached competence.

For the shortcut baseline, competence measures shortcut success and efficiency, while the blue curve measures genuine success. Other tasks use their own saved success and efficiency thresholds.

Source checkpoint labels distinguish the earlier reliable checkpoint from the final source checkpoint. PPO follow-up steps begin at zero. Entropy loss is negative entropy measured on training rollouts and begins after the first PPO update.

Baseline and regular follow-up groups use seeds 7, 9, 11, 13, 15, 17 and 19. Delivery after demonstrations uses the four completed seeds shared by every group: 7, 9, 11 and 13.

For a matched comparison with delivery after demonstrations, use the four-seed `delivery_transfer_matched` figures. The regular `delivery_transfer` figures contain all seven seeds.

Beacon PPO after demonstrations requested 50,000 transitions; ordinary correction requested 200,000. Their entropy schedules therefore decay over different absolute step counts. Delivery runs requested 300,000 steps. PPO completes full rollouts, so saved final steps can exceed the requested budget.

Entropy loss is measured on training states, not a shared evaluation panel, so it reflects both visited states and policy uncertainty. These figures summarize PPO runs. The supervised imitation stage has different metrics and does not have a PPO entropy-loss curve. See the per-run artifacts for its metrics.

## Figures

| Experiment | Condition | Reliable seeds |
| --- | --- | --- |
| baselines | [genuine-fixed](baselines/genuine-fixed.png) | 7/7 |
| baselines | [genuine-variable](baselines/genuine-variable.png) | 7/7 |
| baselines | [hacky-variable](baselines/hacky-variable.png) | 7/7 |
| baselines | [pickup-and-deliver](baselines/pickup-and-deliver.png) | 5/7 |
| checker correction | [genuine-fixed-reliable](checker_correction/genuine-fixed-reliable.png) | 4/7 |
| checker correction | [genuine-fixed-final](checker_correction/genuine-fixed-final.png) | 0/7 |
| checker correction | [hacky-variable-reliable](checker_correction/hacky-variable-reliable.png) | 2/7 |
| checker correction | [hacky-variable-final](checker_correction/hacky-variable-final.png) | 0/7 |
| action head reset | [hacky-variable-reliable](action_head_reset/hacky-variable-reliable.png) | 7/7 |
| action head reset | [hacky-variable-final](action_head_reset/hacky-variable-final.png) | 6/7 |
| delivery transfer | [genuine-fixed-reliable](delivery_transfer/genuine-fixed-reliable.png) | 0/7 |
| delivery transfer | [genuine-fixed-final](delivery_transfer/genuine-fixed-final.png) | 0/7 |
| delivery transfer | [genuine-variable-reliable](delivery_transfer/genuine-variable-reliable.png) | 7/7 |
| delivery transfer | [genuine-variable-final](delivery_transfer/genuine-variable-final.png) | 6/7 |
| delivery transfer | [hacky-variable-reliable](delivery_transfer/hacky-variable-reliable.png) | 0/7 |
| delivery transfer | [hacky-variable-final](delivery_transfer/hacky-variable-final.png) | 0/7 |
| post demonstration correction | [genuine-fixed-reliable](post_demonstration_correction/genuine-fixed-reliable.png) | 7/7 |
| post demonstration correction | [genuine-fixed-final](post_demonstration_correction/genuine-fixed-final.png) | 7/7 |
| post demonstration correction | [hacky-variable-reliable](post_demonstration_correction/hacky-variable-reliable.png) | 7/7 |
| post demonstration correction | [hacky-variable-final](post_demonstration_correction/hacky-variable-final.png) | 7/7 |
| post demonstration correction | [fresh-initial](post_demonstration_correction/fresh-initial.png) | 7/7 |
| post demonstration delivery | [genuine-fixed-reliable](post_demonstration_delivery/genuine-fixed-reliable.png) | 3/4 |
| post demonstration delivery | [genuine-fixed-final](post_demonstration_delivery/genuine-fixed-final.png) | 2/4 |
| post demonstration delivery | [hacky-variable-reliable](post_demonstration_delivery/hacky-variable-reliable.png) | 1/4 |
| post demonstration delivery | [hacky-variable-final](post_demonstration_delivery/hacky-variable-final.png) | 1/4 |
| delivery transfer matched | [genuine-fixed-reliable](delivery_transfer_matched/genuine-fixed-reliable.png) | 0/4 |
| delivery transfer matched | [genuine-fixed-final](delivery_transfer_matched/genuine-fixed-final.png) | 0/4 |
| delivery transfer matched | [hacky-variable-reliable](delivery_transfer_matched/hacky-variable-reliable.png) | 0/4 |
| delivery transfer matched | [hacky-variable-final](delivery_transfer_matched/hacky-variable-final.png) | 0/4 |

## Reproduce

The bundled `data/seed_metrics.json.gz` contains the original per-seed measurements and provenance used for these figures. It includes follow-up seeds that are not in the smaller Hugging Face model upload.

From the project root, run `python -m experiments.runners.plot_final_results` to regenerate the figures, this index and `summary.csv` from the bundled data. Use `python -m experiments.runners.plot_final_results --from-runs` to rebuild the bundled data from the complete original `runs/` folders.

The CSV records seed counts, reliable seeds, conditional mean confirmation steps and both requested and actual final budgets. Repeated evaluation panels and the small number of seeds limit how broadly the results can be interpreted.
