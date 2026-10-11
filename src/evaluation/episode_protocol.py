from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, List
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from src.data.load_engine_map import CONTEXT
from src.health.shi import SHI, MONITORED
from src.health.envelope_gate import EnvelopeGate
from src.health.witness_isolator import WitnessIsolator
from src.faults import injectors

MECHANISMS = ['bias', 'gain', 'noise', 'saturation', 'partial_dropout']
STOCHASTIC = {'noise', 'partial_dropout'}


DEV_GRID = {'bias': [0.5, 1.0, 2.0], 'gain': [0.05, 0.12, 0.20],
            'noise': [1.0, 2.0], 'saturation': [80, 88], 'partial_dropout': [0.15, 0.30]}
TEST_GRID = {'bias': [0.75, 1.5, 3.0], 'gain': [0.08, 0.16],
             'noise': [1.5, 3.0], 'saturation': [84, 92], 'partial_dropout': [0.10, 0.40]}


@dataclass(frozen=True)
class EpisodeConfig:
    seed: int = 42
    episode_sizes: Tuple[int, ...] = (1, 5, 10, 20)
    modes: Tuple[str, ...] = ('diverse', 'local')
    n_realizations: int = 3
    n_episodes: int = 30
    n_healthy_episodes: int = 200
    target_healthy_fa: float = 0.05
    target_accepted_precision: float = 0.90
    min_supported: int = 1
    min_accepted: int = 25
    n_inner: int = 4
    n_boot: int = 400
    boot_seed: int = 12345
    alpha: float = 0.05
    shi_normalization: str = 'region'
    isolator: str = 'structured'
    mechanism_model: str = 'episode_rf'

    def as_dict(self):
        return asdict(self)


def derive_seed(*parts) -> int:

    s = '|'.join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def wilson_bounds(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:

    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def wilson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    return wilson_bounds(k, n, alpha)[1]


def wilson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    return wilson_bounds(k, n, alpha)[0]


def _poly_ridge():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(1.0))


class FoldModels:


    def __init__(self, seed: int, shi_normalization: str = 'region',
                 isolator: str = 'structured'):


        self.seed = seed
        self.shi_normalization = shi_normalization
        self.isolator = isolator

    def fit(self, train: pd.DataFrame):
        self.shi = SHI(system_fpr=0.05, seed=self.seed,
                       normalization=self.shi_normalization).fit(train)
        self.gate = EnvelopeGate(k=3, quantile=0.95).fit(train[CONTEXT].values)
        if self.isolator != 'structured':
            raise ValueError(f'unsupported isolator {self.isolator!r}')
        self.iso = WitnessIsolator(seed=self.seed, n_inner=3).fit(train)
        self.ch_models = {c: _poly_ridge().fit(train[CONTEXT].values, train[c].values)
                          for c in MONITORED}
        self.scales = {c: float(np.std(train[c].values -
                                       self.ch_models[c].predict(train[CONTEXT].values)) + 1e-9)
                       for c in MONITORED}
        self.level_refs = {c: float(np.mean(np.abs(train[c].values))) for c in MONITORED}
        self.stds = train[MONITORED].std()
        pcts = sorted(set(DEV_GRID['saturation']) | set(TEST_GRID['saturation']))
        self.sat_caps = {c: {p: float(np.percentile(train[c].values, p)) for p in pcts}
                         for c in MONITORED}
        self.coord_scaler = StandardScaler().fit(train[CONTEXT].values)
        self._train_index = set(train.index.tolist())
        return self


    def score(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        supported = self.gate.inside(frame[CONTEXT].values)
        suspicious = self.shi.detect(frame)
        cands = self.iso.usable
        smat = np.column_stack([self.iso.improvement(c, frame) for c in cands]) \
            if cands else np.zeros((len(frame), 0))
        resid = np.column_stack([frame[c].values - self.ch_models[c].predict(frame[CONTEXT].values)
                                 for c in MONITORED])
        vals = frame[MONITORED].values
        return dict(supported=supported, suspicious=suspicious, smat=smat,
                    cands=cands, resid=resid, vals=vals,
                    coords=self.coord_scaler.transform(frame[CONTEXT].values))


def inject(frame, ch, mech, val, direction, stds, sat_caps, seed):

    rng = np.random.RandomState(seed)
    if mech == 'bias':
        return injectors.apply(frame, ch, 'bias', stds[ch], rng, severity=val, direction=direction)
    if mech == 'gain':
        return injectors.apply(frame, ch, 'gain', stds[ch], rng, gain_frac=val, direction=direction)
    if mech == 'noise':
        return injectors.apply(frame, ch, 'noise', stds[ch], rng, noise_mult=val)
    if mech == 'saturation':
        return injectors.apply(frame, ch, 'saturation', stds[ch], rng,
                               cap_percentile=val, cap_value=sat_caps[ch][val])
    if mech == 'partial_dropout':
        return injectors.apply(frame, ch, 'partial_dropout', stds[ch], rng, frac=val)
    raise ValueError(mech)


CHAN_FEATS = ['abs_mean', 'signed_mean', 'std', 'sign_consistency', 'corr_level',
              'frac_distinct', 'frac_zero', 'frac_at_max', 'frac_at_min',
              'level_rel', 'max_abs', 'iqr']
EPI_FEATS = ['support_frac', 'susp_frac', 'n_susp', 'src_margin', 'src_top_norm',
             'src_ratio', 'src_spread',

             'src_win_freq', 'src_vote_entropy', 'src_obs_margin_mean',
             'src_obs_margin_min', 'src_loo_stability']
ALL_FEATS = CHAN_FEATS + EPI_FEATS
REPORTING_FEATURE_POSITIONS = EPI_FEATS + [
    'mechanism_probability_max',
    'mechanism_probability_entropy',
    'mechanism_probability_margin',
    'n_susp',
    'support_frac',
    'batch_size',
]


def channel_features(resid, values, scale, level_ref) -> List[float]:
    n = len(resid)
    r = np.asarray(resid, float) / (scale + 1e-9)
    v = np.asarray(values, float)
    absr = np.abs(r)
    std_r = r.std() if n > 1 else 0.0
    sign_consistency = abs(r.mean()) / (std_r + 1e-9) if n > 1 else abs(r.mean())
    if n > 2 and v.std() > 1e-9 and absr.std() > 1e-9:
        corr_level = float(np.corrcoef(absr, v)[0, 1])
    else:
        corr_level = 0.0
    frac_distinct = len(np.unique(np.round(v, 6))) / n
    frac_zero = float(np.mean(np.isclose(v, 0.0)))
    frac_at_max = float(np.mean(np.isclose(v, v.max(), rtol=1e-6)))
    frac_at_min = float(np.mean(np.isclose(v, v.min(), rtol=1e-6)))
    level_rel = float(v.mean() / (level_ref + 1e-9))
    iqr = float(np.subtract(*np.percentile(r, [75, 25]))) if n > 3 else 0.0
    return [float(absr.mean()), float(r.mean()), float(std_r), float(sign_consistency),
            corr_level, frac_distinct, frac_zero, frac_at_max, frac_at_min,
            level_rel, float(absr.max()), iqr]


def episode_level_features(support_frac, susp_frac, n_susp, agg_scores,
                           per_obs_scores=None) -> List[float]:


    base_missing = [support_frac, susp_frac, float(n_susp), 0.0, 0.0, 0.0, 0.0]
    stab_missing = [0.0, 0.0, 0.0, 0.0, 0.0]
    if agg_scores is None or len(agg_scores) == 0:
        return base_missing + stab_missing
    a_raw = np.asarray(agg_scores, float)
    a = np.sort(a_raw)[::-1]
    top = float(a[0]); second = float(a[1]) if len(a) > 1 else 0.0
    denom = float(np.abs(a).sum()) + 1e-9
    base = [support_frac, susp_frac, float(n_susp), top - second, top / denom,
            top / (abs(second) + 1e-9), float(np.std(a))]

    if per_obs_scores is None or len(per_obs_scores) == 0:
        return base + stab_missing
    P = np.asarray(per_obs_scores, float)
    winner = int(np.argmax(a_raw))
    per_obs_win = P.argmax(axis=1)
    win_freq = float(np.mean(per_obs_win == winner))

    counts = np.bincount(per_obs_win, minlength=P.shape[1]).astype(float)
    pvec = counts / max(counts.sum(), 1e-9)
    nz = pvec[pvec > 0]
    entropy = float(-(nz * np.log(nz)).sum())

    Ps = np.sort(P, axis=1)[:, ::-1]
    obs_margin = Ps[:, 0] - (Ps[:, 1] if Ps.shape[1] > 1 else 0.0)

    if len(P) > 1:
        tot = P.sum(axis=0)
        stable = np.mean([int(np.argmax(tot - P[i]) == winner) for i in range(len(P))])
    else:
        stable = 1.0
    return base + [win_freq, entropy, float(obs_margin.mean()),
                   float(obs_margin.min()), float(stable)]


def sample_episode_idx(coords, n_rows, N, mode, rng) -> Tuple[np.ndarray, Optional[int]]:

    N = min(N, n_rows)
    if mode == 'diverse' or N == 1:
        return rng.choice(n_rows, size=N, replace=False), None
    center = int(rng.randint(n_rows))
    d = np.linalg.norm(coords - coords[center], axis=1)
    idx = np.argsort(d)[:N]
    return idx, center


def score_scenarios(models: FoldModels, frame: pd.DataFrame, grid: Dict[str, List],
                    cfg: EpisodeConfig, *, regime: str, outer_fold: int, stage: str) -> List[dict]:


    out = [dict(kind='healthy', scen_id='healthy', mech='healthy', true_src=None,
                severity=np.nan, direction=0, realization_id=0,
                scored=models.score(frame), mask=np.zeros(len(frame), bool))]
    for ch in MONITORED:
        for mech in MECHANISMS:
            for val in grid[mech]:
                dirs = [+1, -1] if mech in ('bias', 'gain') else [+1]
                for d in dirs:
                    scen_id = f"{ch}|{mech}|{val}|{d}"
                    n_real = cfg.n_realizations if mech in STOCHASTIC else 1
                    for real in range(n_real):
                        fseed = derive_seed(cfg.seed, regime, outer_fold, stage, scen_id, real)
                        fframe, mask = inject(frame, ch, mech, val, d,
                                              models.stds, models.sat_caps, fseed)
                        out.append(dict(kind='fault', scen_id=scen_id, mech=mech, true_src=ch,
                                        severity=val, direction=d, realization_id=real,
                                        scored=models.score(fframe), mask=mask, fseed=fseed))
    return out


def episodes_from_scenarios(models: FoldModels, scenarios: List[dict], N: int, mode: str,
                            cfg: EpisodeConfig, *, regime: str, outer_fold: int, stage: str,
                            row_ids: np.ndarray) -> List[dict]:

    recs = []
    n_rows = len(row_ids)
    for sc in scenarios:
        scored = sc['scored']
        coords = scored['coords']
        cands = scored['cands']
        n_ep = cfg.n_healthy_episodes if sc['kind'] == 'healthy' else cfg.n_episodes
        base = sc.get('fseed', derive_seed(cfg.seed, regime, outer_fold, stage, 'healthy'))
        rng = np.random.RandomState(derive_seed(base, mode, N, 'episodes'))
        for e in range(n_ep):
            idx, center = sample_episode_idx(coords, n_rows, N, mode, rng)
            recs.append(_make_record(
                models, None, scored, idx, center, cands,
                mech=sc['mech'], true_src=sc['true_src'], severity=sc['severity'],
                direction=sc['direction'], realization_id=sc['realization_id'],
                episode_idx=e, manifest_mask=sc['mask'], regime=regime, mode=mode, N=N,
                outer_fold=outer_fold, stage=stage, row_ids=row_ids, scen_id=sc['scen_id']))
    return recs


def build_episode_records(models: FoldModels, frame: pd.DataFrame, grid: Dict[str, List],
                          N: int, mode: str, cfg: EpisodeConfig, *, regime: str,
                          outer_fold: int, stage: str, row_ids: np.ndarray) -> List[dict]:


    scen = score_scenarios(models, frame, grid, cfg, regime=regime,
                           outer_fold=outer_fold, stage=stage)
    return episodes_from_scenarios(models, scen, N, mode, cfg, regime=regime,
                                   outer_fold=outer_fold, stage=stage, row_ids=row_ids)


def _make_record(models, _frame, scored, idx, center, cands, *, mech, true_src, severity,
                 direction, realization_id, episode_idx, manifest_mask, regime, mode, N,
                 outer_fold, stage, row_ids, scen_id) -> dict:
    sup = scored['supported'][idx]
    sus = scored['suspicious'][idx] & sup
    n_sup, n_susp = int(sup.sum()), int(sus.sum())
    support_frac = n_sup / len(idx)
    susp_frac = n_susp / max(n_sup, 1)


    agg = None; pred_src = None
    if n_susp > 0 and len(cands) > 0:
        agg = scored['smat'][idx][sus].sum(axis=0)
        pred_src = cands[int(np.argmax(agg))]

    per_obs = scored['smat'][idx][sus] if n_susp > 0 else None
    epi = episode_level_features(support_frac, susp_frac, n_susp, agg, per_obs)


    if pred_src is not None:
        j = MONITORED.index(pred_src)
        f_pred = channel_features(scored['resid'][idx, j], scored['vals'][idx, j],
                                  models.scales[pred_src], models.level_refs[pred_src])
    else:
        f_pred = [0.0] * len(CHAN_FEATS)


    if true_src is not None:
        jt = MONITORED.index(true_src)
        f_or = channel_features(scored['resid'][idx, jt], scored['vals'][idx, jt],
                                models.scales[true_src], models.level_refs[true_src])
    else:
        f_or = [0.0] * len(CHAN_FEATS)

    man = manifest_mask[idx]
    return dict(
        regime=regime, mode=mode, N=N, outer_fold=outer_fold, stage=stage,
        scen_id=scen_id, realization_id=realization_id, episode_idx=episode_idx,
        center_id=(int(row_ids[center]) if center is not None else -1),
        mech=mech, true_src=(true_src or 'none'), severity=severity, direction=direction,
        n_total=len(idx), n_supported=n_sup, n_suspicious=n_susp,
        support_frac=support_frac, susp_frac=susp_frac,
        manifested_count=int(man.sum()), manifested_fraction=float(man.mean()),
        manifested=bool(man.any()),
        pred_src=(pred_src or 'none'),
        row_ids=';'.join(str(int(row_ids[i])) for i in idx),
        f_pred=f_pred, f_oracle=f_or, f_epi=epi,
    )


def calibrate_k_alarm(healthy_n_susp: np.ndarray, N: int, cfg: EpisodeConfig) -> dict:


    n = len(healthy_n_susp)
    for k in range(1, N + 1):
        hits = int((healthy_n_susp >= k).sum())
        fa = hits / max(n, 1)
        lo, up = wilson_bounds(hits, n, cfg.alpha)
        if up <= cfg.target_healthy_fa:
            return dict(k_alarm=k, status='attained', cal_fa=fa, cal_lo=lo, cal_hi=up,
                        n_healthy=n)
    hits = int((healthy_n_susp >= N).sum())
    lo, up = wilson_bounds(hits, n, cfg.alpha)
    return dict(k_alarm=N + 1, status='target_unattainable',
                cal_fa=hits / max(n, 1), cal_lo=lo, cal_hi=up, n_healthy=n)


def tune_confidence(conf: np.ndarray, complete_ok: np.ndarray, cfg: EpisodeConfig) -> dict:


    conf = np.asarray(conf, float); ok = np.asarray(complete_ok, bool)
    if len(conf) == 0:
        return dict(thr=np.inf, status='no_data', coverage=0.0, prec=np.nan, prec_lo=np.nan)
    best = None
    for thr in np.unique(np.quantile(conf, np.linspace(0, 0.99, 60))):
        acc = conf >= thr
        n_acc = int(acc.sum())
        if n_acc < cfg.min_accepted:
            continue
        hits = int(ok[acc].sum())
        prec = hits / n_acc
        lo, _ = wilson_bounds(hits, n_acc, cfg.alpha)
        if lo >= cfg.target_accepted_precision:
            cov = acc.mean()
            if best is None or cov > best['coverage']:
                best = dict(thr=float(thr), status='attained', coverage=float(cov),
                            prec=prec, prec_lo=lo, n_accepted=n_acc)
    if best is not None:
        return best

    return dict(thr=np.inf, status='target_unattainable', coverage=0.0,
                prec=np.nan, prec_lo=np.nan, n_accepted=0)


def cluster_bootstrap(values, clusters, cfg: EpisodeConfig) -> Tuple[float, float, float]:

    v = np.asarray(values, float); c = np.asarray(clusters)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    uniq = np.unique(c)
    groups = {u: v[c == u] for u in uniq}
    rng = np.random.RandomState(cfg.boot_seed)
    draws = np.empty(cfg.n_boot)
    for b in range(cfg.n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        draws[b] = np.concatenate([groups[u] for u in pick]).mean()
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cluster_key(rec: dict) -> str:

    return (f"{rec['outer_fold']}|{rec['true_src']}|{rec['mech']}|{rec['severity']}|"
            f"{rec['direction']}|{rec['realization_id']}")


def inner_splits(train: pd.DataFrame, regime: str, cfg: EpisodeConfig):

    if regime.startswith('grouped_RPM'):
        groups = train['rpm_level'].values
        n_groups = len(np.unique(groups))
        k = int(min(cfg.n_inner, n_groups))
        if k < 2:
            return []
        return list(GroupKFold(k).split(train, groups=groups))
    return list(KFold(cfg.n_inner, shuffle=True, random_state=cfg.seed).split(train))


def prepare_inner_oof(train: pd.DataFrame, regime: str, cfg: EpisodeConfig, outer_fold: int):


    prepared = []
    for i, (itr, iva) in enumerate(inner_splits(train, regime, cfg)):
        tr, va = train.iloc[itr], train.iloc[iva]
        if len(va) < 3:
            continue
        fm = FoldModels(seed=derive_seed(cfg.seed, outer_fold, i),
                        shi_normalization=cfg.shi_normalization,
                        isolator=cfg.isolator).fit(tr)
        scen = score_scenarios(fm, va, DEV_GRID, cfg, regime=regime,
                               outer_fold=outer_fold, stage='inner')

        disjoint = all(int(x) not in fm._train_index for x in va.index.values)
        prepared.append(dict(inner_fold=i, models=fm, scen=scen,
                             row_ids=va.index.values, audit_disjoint=disjoint))
    return prepared


def run_inner_oof(prepared, regime: str, cfg: EpisodeConfig, N: int, mode: str,
                  outer_fold: int) -> List[dict]:

    pooled = []
    for pr in prepared:
        recs = episodes_from_scenarios(pr['models'], pr['scen'], N, mode, cfg, regime=regime,
                                       outer_fold=outer_fold, stage='inner',
                                       row_ids=pr['row_ids'])
        for r in recs:
            r['inner_fold'] = pr['inner_fold']
            r['_audit_disjoint'] = pr['audit_disjoint']
        pooled.extend(recs)
    return pooled


def train_deployment_classifier(records: List[dict], cfg: EpisodeConfig):

    faulted = [r for r in records if r['mech'] != 'healthy']
    if not faulted:
        return None
    X = np.array([r['f_pred'] + r['f_epi'] for r in faulted])
    y = np.array([r['mech'] for r in faulted])
    return RandomForestClassifier(n_estimators=300, random_state=cfg.seed,
                                  class_weight='balanced').fit(X, y)


def oof_confidence_dataset(records: List[dict], k_alarm: int, cfg: EpisodeConfig):


    faulted = [r for r in records if r['mech'] != 'healthy']
    if not faulted:
        return np.array([]), np.array([])
    X = np.array([r['f_pred'] + r['f_epi'] for r in faulted])
    y = np.array([r['mech'] for r in faulted])
    groups = np.array([f"{r['inner_fold']}|{r['scen_id']}|{r['realization_id']}"
                       for r in faulted])
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return np.array([]), np.array([])
    conf = np.zeros(len(faulted)); pred = np.empty(len(faulted), dtype=object)
    gkf = GroupKFold(int(min(4, n_groups)))
    for tr, te in gkf.split(X, y, groups=groups):
        clf = RandomForestClassifier(n_estimators=200, random_state=cfg.seed,
                                     class_weight='balanced').fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        conf[te] = p.max(1)
        pred[te] = clf.classes_[p.argmax(1)]
    detected = np.array([r['n_suspicious'] >= k_alarm for r in faulted])
    supported = np.array([r['n_supported'] >= cfg.min_supported for r in faulted])
    src_ok = np.array([r['pred_src'] == r['true_src'] for r in faulted])
    type_ok = np.array([pred[i] == y[i] for i in range(len(faulted))])
    complete_ok = detected & supported & src_ok & type_ok
    return conf, complete_ok


def mech_features(record):


    return list(record['f_pred']) + list(record['f_epi'])


def episode_groups(records):

    return np.array([f"{r.get('outer_fold', 0)}_{r['true_src']}_{r['mech']}_"
                     f"{r.get('severity', 0)}_{r.get('direction', 0)}_"
                     f"{r.get('realization_id', 0)}" for r in records], dtype=object)


def oof_mechanism_probabilities(records, seed, n_splits=5):


    from sklearn.ensemble import RandomForestClassifier as _RF


    feats = np.array([mech_features(r) for r in records])
    ym = np.array([r['mech'] for r in records])
    groups = episode_groups(records)
    oof = np.zeros((len(records), len(MECHANISMS)))
    uniq = np.unique(groups)
    gkf = GroupKFold(n_splits=int(min(n_splits, max(2, len(uniq)))))
    for tr, va in gkf.split(feats, ym, groups=groups):
        if len(np.unique(ym[tr])) < 2:
            oof[va] = 1.0 / len(MECHANISMS)
            continue
        m = _RF(n_estimators=200, random_state=seed, class_weight='balanced').fit(feats[tr], ym[tr])
        cols = [list(m.classes_).index(c) if c in m.classes_ else None for c in MECHANISMS]
        pv = m.predict_proba(feats[va])
        for j, ci in enumerate(cols):
            if ci is not None:
                oof[va, j] = pv[:, ci]
    refit = _RF(n_estimators=300, random_state=seed, class_weight='balanced').fit(feats, ym)
    return oof, refit


def correctness_features(record, proba):

    p = np.clip(np.asarray(proba, float), 1e-9, 1.0)
    srt = np.sort(p)[::-1]
    return list(record['f_epi']) + [
        float(p.max()), mechanism_probability_dispersion(p),
        float(srt[0] - srt[1]) if len(srt) > 1 else 0.0,
        float(record['n_suspicious']), float(record['support_frac']), float(record['N'])]


def mechanism_probability_dispersion(proba):

    p = np.clip(np.asarray(proba, float), 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())


def complete_correct(record, proba, k_alarm):

    detected = record['n_suspicious'] >= k_alarm
    supported = record['support_frac'] > 0
    pred_mech = MECHANISMS[int(np.argmax(np.asarray(proba, float)))]
    return bool(detected and supported and (record['pred_src'] == record['true_src'])
                and (pred_mech == record['mech']))


def fit_nested_correctness_calibrator(records, k_alarm, seed, model='logistic', n_splits=5):


    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.ensemble import GradientBoostingClassifier as _GB

    if len(records) < 20:
        return None
    oof_mech, mech_refit = oof_mechanism_probabilities(records, seed, n_splits)
    X = np.array([correctness_features(r, oof_mech[i]) for i, r in enumerate(records)])
    y = np.array([int(complete_correct(r, oof_mech[i], k_alarm)) for i, r in enumerate(records)])
    baseline = oof_mech.max(axis=1)
    if len(np.unique(y)) < 2:
        return None

    def _ctor():
        if model == 'grad_boost':
            return _GB(random_state=seed)
        return make_pipeline(StandardScaler(), _LR(max_iter=2000))

    groups = episode_groups(records)
    uniq = np.unique(groups)
    gkf = GroupKFold(n_splits=int(min(n_splits, max(2, len(uniq)))))
    oof_cal = np.zeros(len(y))
    for tr, va in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            oof_cal[va] = float(y[tr].mean()) if len(tr) else 0.5
            continue
        oof_cal[va] = _ctor().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]

    return dict(calibrator=_ctor().fit(X, y), mech_clf=mech_refit,
                oof_calibrated=oof_cal, oof_baseline=baseline, correct=y,
                n=len(y))


class JointTyper:


    def __init__(self, seed=0, model='logistic'):
        self.seed = int(seed)
        self.model = model
        self.classes_ = np.array(MECHANISMS)

    def _ctor(self):
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.ensemble import RandomForestClassifier as _RF
        if self.model == 'rf':
            return _RF(n_estimators=300, random_state=self.seed, class_weight='balanced')
        return make_pipeline(StandardScaler(),
                             _LR(max_iter=3000, C=1.0, class_weight='balanced'))

    def fit(self, records):
        x = np.array([mech_features(r) for r in records])
        y = np.array([f"{r['true_src']}|{r['mech']}" for r in records])
        self._model = self._ctor().fit(x, y)
        self._pairs = list(self._model.classes_ if hasattr(self._model, 'classes_')
                           else self._model[-1].classes_)
        return self

    def _features(self, record):
        return np.array(mech_features(record)).reshape(1, -1)

    def predict_pair(self, record):

        proba = self._model.predict_proba(self._features(record))[0]
        best = self._pairs[int(np.argmax(proba))]
        sensor, mech = best.split('|')
        return sensor, mech

    def predict_proba_mech(self, record):

        proba = self._model.predict_proba(self._features(record))[0]
        out = np.zeros(len(MECHANISMS))
        for pair, p in zip(self._pairs, proba):
            mech = pair.split('|')[1]
            if mech in MECHANISMS:
                out[MECHANISMS.index(mech)] += p
        total = out.sum()
        return out / total if total > 0 else np.full(len(MECHANISMS), 1.0 / len(MECHANISMS))


def train_joint_typer(records, cfg, model='logistic'):

    usable = [r for r in records if r['true_src'] != 'none' and r['mech'] in MECHANISMS]
    if len(usable) < 30:
        return None
    return JointTyper(seed=derive_seed(cfg.seed, 'joint'), model=model).fit(usable)


def fit_nested_correctness_calibrator_joint(records, k_alarm, seed, model='logistic',
                                            n_splits=5):


    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.ensemble import GradientBoostingClassifier as _GB

    usable = [r for r in records if r['true_src'] != 'none' and r['mech'] in MECHANISMS]
    if len(usable) < 30:
        return None
    groups = episode_groups(usable)
    uniq = np.unique(groups)
    gkf = GroupKFold(n_splits=int(min(n_splits, max(2, len(uniq)))))
    x = np.array([mech_features(r) for r in usable])

    oof_pair_conf = np.zeros(len(usable))
    oof_mech_proba = np.zeros((len(usable), len(MECHANISMS)))
    oof_correct = np.zeros(len(usable), dtype=int)
    for tr, va in gkf.split(x, np.array([r['mech'] for r in usable]), groups=groups):
        sub = [usable[i] for i in tr]
        jt = JointTyper(seed=seed, model=model)
        try:
            jt.fit(sub)
        except Exception:
            continue
        for i in va:
            rec = usable[i]
            src, mech = jt.predict_pair(rec)
            proba = jt.predict_proba_mech(rec)
            oof_mech_proba[i] = proba
            oof_pair_conf[i] = float(proba.max())
            detected = rec['n_suspicious'] >= k_alarm
            oof_correct[i] = int(detected and rec['support_frac'] > 0
                                 and src == rec['true_src'] and mech == rec['mech'])
    if len(np.unique(oof_correct)) < 2:
        return None

    feats = np.array([correctness_features(r, oof_mech_proba[i])
                      for i, r in enumerate(usable)])
    def _ctor():
        if model == 'grad_boost':
            return _GB(random_state=seed)
        return make_pipeline(StandardScaler(), _LR(max_iter=2000))
    oof_cal = np.zeros(len(oof_correct))
    for tr, va in gkf.split(feats, oof_correct, groups=groups):
        if len(np.unique(oof_correct[tr])) < 2:
            oof_cal[va] = float(oof_correct[tr].mean()) if len(tr) else 0.5
            continue
        oof_cal[va] = _ctor().fit(feats[tr], oof_correct[tr]).predict_proba(feats[va])[:, 1]

    return dict(calibrator=_ctor().fit(feats, oof_correct),
                joint=JointTyper(seed=seed, model=model).fit(usable),
                oof_calibrated=oof_cal, oof_baseline=oof_pair_conf,
                correct=oof_correct, n=len(oof_correct))
