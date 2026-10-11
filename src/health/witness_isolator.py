import itertools
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

MONITORED = ['T_IM', 'P_IM', 'EGR_Rate', 'MF_IA', 'NOx_EO']
CONTEXT = ['Engine Speed', 'Torque']


def _pr():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(1.0))


def valid_witnesses(c, S):

    out = []
    for t in MONITORED:
        if t == c or t in S:
            continue
        for pair in itertools.combinations([x for x in MONITORED if x != t], 2):
            I = set(pair)
            if c not in I or (I - {c}) & S:
                continue
            out.append((t, tuple(sorted(I))))
    return out


def _fit_route(train_frame, c, S, wt, wi):
    recon = _pr().fit(train_frame[CONTEXT + list(S)].values, train_frame[c].values)
    wit = _pr().fit(train_frame[CONTEXT + list(wi)].values, train_frame[wt].values)
    hr = np.abs(train_frame[wt].values - wit.predict(train_frame[CONTEXT + list(wi)].values))
    scale = np.median(np.abs(hr - np.median(hr))) * 1.4826 + 1e-9
    return recon, wit, scale


def _selection_score(recon, wit, scale, val_frame, c, S, wt, wi, stds):
    def wr(frame):
        return np.abs(frame[wt].values - wit.predict(frame[CONTEXT + list(wi)].values))
    faulted = val_frame.copy(); faulted[c] = faulted[c] + 2.0 * stds[c]
    rep = faulted.copy(); rep[c] = recon.predict(rep[CONTEXT + list(S)].values)
    healthy_rep = val_frame.copy()
    healthy_rep[c] = recon.predict(healthy_rep[CONTEXT + list(S)].values)
    sig = (wr(faulted).mean() - wr(rep).mean()) / scale
    noise = abs(wr(healthy_rep).mean() - wr(val_frame).mean()) / scale
    return sig, sig - noise


class WitnessIsolator:


    def __init__(self, seed=0, n_inner=4):
        self.seed = seed
        self.n_inner = n_inner
        self.routes = {}
        self.usable = []

    def fit(self, train):
        stds = train[MONITORED].std()
        kf = KFold(self.n_inner, shuffle=True, random_state=self.seed)
        for c in MONITORED:
            candidates = [(S, wt, wi)
                          for S in [set([s]) for s in MONITORED if s != c]
                          for (wt, wi) in valid_witnesses(c, S)]
            best = None
            for (S, wt, wi) in candidates:
                scores, sigs = [], []
                for in_tr, in_val in kf.split(train):
                    itr, ival = train.iloc[in_tr], train.iloc[in_val]
                    if len(ival) < 3:
                        continue
                    recon, wit, scale = _fit_route(itr, c, S, wt, wi)
                    sig, sc = _selection_score(recon, wit, scale, ival, c, S, wt, wi, stds)
                    sigs.append(sig); scores.append(sc)
                if not scores or np.mean(sigs) < 1e-6:
                    continue
                m = np.mean(scores)
                if best is None or m > best['cv']:
                    best = dict(S=S, wt=wt, wi=wi, cv=m)
            if best is not None:
                recon, wit, scale = _fit_route(train, c, best['S'], best['wt'], best['wi'])
                best.update(dict(recon=recon, wit=wit, scale=scale, channel=c))
                self.routes[c] = best
        self.usable = [c for c in MONITORED if c in self.routes]
        return self

    def improvement(self, c, frame):

        r = self.routes[c]
        S, wt, wi = r['S'], r['wt'], r['wi']
        def wr(fr):
            return np.abs(fr[wt].values - r['wit'].predict(fr[CONTEXT + list(wi)].values))
        rep = frame.copy()
        rep[c] = r['recon'].predict(rep[CONTEXT + list(S)].values)
        return (wr(frame) - wr(rep)) / r['scale']

    def scores(self, frame):


        mat = np.column_stack([self.improvement(c, frame) for c in self.usable])
        return mat, list(self.usable)

    def isolate(self, frame):

        n = len(frame)
        imps = {c: self.improvement(c, frame) for c in self.usable}
        preds, margins = [], []
        for i in range(n):
            ranked = sorted(((c, imps[c][i]) for c in self.usable),
                            key=lambda kv: kv[1], reverse=True)
            preds.append(ranked[0][0])
            margins.append(ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0))
        return np.array(preds), np.array(margins)

    def reconstruct(self, c, frame):

        r = self.routes[c]
        return r['recon'].predict(frame[CONTEXT + list(r['S'])].values)
