"""Group-based experiment splits; never imported by the application."""

from sklearn.model_selection import GroupShuffleSplit


def split_dataset(dataset, test_size: float = 0.2, seed: int = 42):
    if not 0 < test_size < 1 or len(set(dataset.groups)) < 2:
        raise ValueError("A split requires test_size in (0, 1) and at least two text groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(dataset.scores, groups=dataset.groups))
