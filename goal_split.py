from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GoalSplit:
    """Training-eligible and withheld beacon locations for a development run."""

    seen: tuple[tuple[int, int], ...]
    withheld: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if len(self.seen) < 10 or len(self.withheld) != 10:
            raise ValueError('the development panel needs at least 10 seen and exactly 10 withheld goals')
        if len(set(self.seen)) != len(self.seen) or len(set(self.withheld)) != len(self.withheld):
            raise ValueError('goal lists must not contain duplicates')
        if set(self.seen) & set(self.withheld):
            raise ValueError('seen and withheld goals must be disjoint')

    def evaluation_panel(self, seed: int):
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(self.seen), size=10, replace=False)
        repeated = rng.choice(selected, size=5, replace=False)
        panel = [(self.seen[int(index)], 'seen') for index in selected]
        panel += [(self.seen[int(index)], 'seen') for index in repeated]
        panel += [(goal, 'withheld') for goal in self.withheld]
        rng.shuffle(panel)
        return panel


def development_goal_split() -> GoalSplit:
    # Coordinates are (x, y); the old fixed goal is (3, 0).
    withheld = ((0, 0), (6, 6), (0, 1), (6, 5), (0, 2),
                (6, 4), (0, 3), (6, 3), (3, 1), (3, 5))
    seen = tuple((x, y) for x in range(7) for y in range(7)
                 if (x, y) != (3, 0) and (x, y) not in withheld)
    return GoalSplit(seen=seen, withheld=withheld)
