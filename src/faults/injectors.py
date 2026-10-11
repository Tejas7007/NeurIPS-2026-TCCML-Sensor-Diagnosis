import numpy as np


def inject_bias(x, std, severity, direction=+1):

    x = x.astype(float).copy()
    x = x + direction * severity * std
    mask = np.ones(len(x), dtype=bool)
    return x, mask


def inject_gain(x, std, gain_frac, direction=+1):

    x = x.astype(float).copy()
    x = x * (1 + direction * gain_frac)
    mask = np.ones(len(x), dtype=bool)
    return x, mask


def inject_saturation(x, std, cap_percentile=90, side='upper', cap_value=None):


    x = x.astype(float).copy()
    if side == 'upper':
        cap = cap_value if cap_value is not None else np.percentile(x, cap_percentile)
        mask = x > cap
        x = np.minimum(x, cap)
    else:
        cap = cap_value if cap_value is not None else np.percentile(x, 100 - cap_percentile)
        mask = x < cap
        x = np.maximum(x, cap)
    return x, mask


def inject_dropout(x, std, frac, rng, dropout_value=0.0):


    x = x.astype(float).copy()
    mask = rng.random(len(x)) < frac
    x[mask] = dropout_value
    return x, mask


def inject_partial_bias(x, std, severity, frac, rng, direction=+1):


    x = x.astype(float).copy()
    mask = rng.random(len(x)) < frac
    x[mask] = x[mask] + direction * severity * std
    return x, mask


def inject_noise(x, std, noise_mult, rng):

    x = x.astype(float).copy()
    x = x + rng.normal(0, noise_mult * std, len(x))
    mask = np.ones(len(x), dtype=bool)
    return x, mask


FAULT_LIBRARY = {
    'bias':           dict(fn=inject_bias,        param='severity_std'),
    'gain':           dict(fn=inject_gain,        param='gain_frac'),
    'saturation':     dict(fn=inject_saturation,  param='cap_percentile'),
    'partial_dropout':dict(fn=inject_dropout,     param='frac'),
    'partial_bias':   dict(fn=inject_partial_bias,param='severity_std+frac'),
    'noise':          dict(fn=inject_noise,       param='noise_mult'),
}


def apply(frame, channel, fault_type, std, rng, **kwargs):

    f = frame.copy()
    x = f[channel].values
    if fault_type == 'bias':
        xc, mask = inject_bias(x, std, kwargs['severity'], kwargs.get('direction', +1))
    elif fault_type == 'gain':
        xc, mask = inject_gain(x, std, kwargs['gain_frac'], kwargs.get('direction', +1))
    elif fault_type == 'saturation':
        xc, mask = inject_saturation(x, std, kwargs.get('cap_percentile', 90),
                                     kwargs.get('side', 'upper'), kwargs.get('cap_value', None))
    elif fault_type == 'partial_dropout':
        xc, mask = inject_dropout(x, std, kwargs['frac'], rng, kwargs.get('dropout_value', 0.0))
    elif fault_type == 'partial_bias':
        xc, mask = inject_partial_bias(x, std, kwargs['severity'], kwargs['frac'], rng,
                                       kwargs.get('direction', +1))
    elif fault_type == 'noise':
        xc, mask = inject_noise(x, std, kwargs['noise_mult'], rng)
    else:
        raise ValueError(f"unknown fault_type {fault_type}")
    f[channel] = xc
    return f, mask
