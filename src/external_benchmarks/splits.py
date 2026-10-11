import numpy as np


def assert_no_group_leakage(train_groups, test_groups, label=''):

    overlap = set(np.unique(train_groups)) & set(np.unique(test_groups))
    if overlap:
        raise AssertionError(
            f'{label}group leakage: {len(overlap)} group(s) in both train and test, '
            f'e.g. {sorted(overlap)[:5]}')
    return True


def chronological_split(groups, timestamps, test_fraction=0.3):

    groups = np.asarray(groups)
    order = {}
    for g, t in zip(groups, np.asarray(timestamps)):
        order.setdefault(g, []).append(t)
    ranked = sorted(order, key=lambda g: np.min(order[g]))
    cut = int(len(ranked) * (1 - test_fraction))
    train_groups, test_groups = set(ranked[:cut]), set(ranked[cut:])
    tr = np.array([g in train_groups for g in groups])
    assert_no_group_leakage(groups[tr], groups[~tr], 'chronological split: ')
    return tr, ~tr


def grouped_kfold(groups, n_splits=5, seed=0):

    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    idx = np.arange(len(groups))
    for tr, te in GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(idx, groups=groups):
        assert_no_group_leakage(groups[tr], groups[te], 'grouped k-fold: ')
        yield tr, te


def disjoint_severity_grids(dev_grid, test_grid):

    shared = set(np.round(dev_grid, 10)) & set(np.round(test_grid, 10))
    if shared:
        raise AssertionError(f'development and test severity grids overlap at {sorted(shared)}')
    return True
