# Keep the original budgets and schedules in one place for the experiment runners.
SOURCE_BUDGET = 75_000
CORRECTION_BUDGET = 200_000
DELIVERY_BUDGET = 300_000
ENTROPY_SCHEDULE = {'initial_coef': 0.1, 'final_coef': 0.01, 'decay_fraction': 0.35}
DELIVERY_CURRICULUM = {'initial_probability': 0.5, 'decay_fraction': 0.5, 'max_distance': 2}
