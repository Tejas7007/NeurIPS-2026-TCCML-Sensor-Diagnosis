import numpy as np
from sklearn.model_selection import KFold


def inner_oof_indices(n, n_folds=5, seed=0):

    kf = KFold(n_folds, shuffle=True, random_state=seed)
    idx = np.arange(n)
    for fit, score in kf.split(idx):
        yield fit, score


def oof_map(frame, build_fn, apply_fn, n_folds=5, seed=0):


    n = len(frame)
    out = None
    for fit, score in inner_oof_indices(n, n_folds, seed):
        model = build_fn(frame.iloc[fit])
        vals = apply_fn(model, frame.iloc[score])
        if out is None:
            out = np.full((n, vals.shape[1]), np.nan)
        out[score] = vals
    return out


def split_dev(frame, dev_frac=0.3, seed=0):


    n = len(frame)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    cut = int((1 - dev_frac) * n)
    fit_idx, dev_idx = idx[:cut], idx[cut:]
    return frame.iloc[fit_idx].copy(), frame.iloc[dev_idx].copy()
