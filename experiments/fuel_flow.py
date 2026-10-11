import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor

from src.data.load_engine_map import load_engine_map, assign_speed_blocks, CONTEXT
from src.health.envelope_gate import EnvelopeGate
from src.faults import injectors
from src.evaluation.episode_protocol import DEV_GRID, TEST_GRID, MECHANISMS, wilson_lower, derive_seed

RNG = 42
TARGET = 'MF_FUEL'
BASE_MONITORED = ['T_IM', 'P_IM', 'EGR_Rate', 'MF_IA', 'NOx_EO']
MONITORED = BASE_MONITORED + [TARGET]
TARGET_SYSTEM_FA = 0.05
TARGET_SOURCE_PRECISION = 0.90
OUT_TABLE = ROOT / 'results'
OUT_RECORD = ROOT / 'results' / 'records'


def poly_ridge():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(1.0))


def split_inner(frame, regime, n_splits=4, seed=RNG):
    if regime.startswith('grouped_RPM'):
        groups = frame['rpm_level'].values
        k = min(n_splits, len(np.unique(groups)))
        if k < 2:
            return []
        return list(GroupKFold(k).split(frame, groups=groups))
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(frame))


class GenericConsistencyMonitor:

    def __init__(self, seed=0, n_estimators=120):
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)

    def _fit_models(self, frame):
        out = {}
        for ch in MONITORED:
            peers = [c for c in MONITORED if c != ch]
            cols = CONTEXT + peers
            out[ch] = (poly_ridge().fit(frame[cols].values, frame[ch].values), cols)
        return out

    @staticmethod
    def _residuals(models, frame):
        return np.column_stack([
            frame[ch].values - models[ch][0].predict(frame[models[ch][1]].values)
            for ch in MONITORED
        ])

    def fit(self, train, regime='within_map'):
        R = np.zeros((len(train), len(MONITORED)), float)
        splits = split_inner(train, regime, n_splits=min(5, max(2, len(train) // 20)),
                             seed=self.seed)
        if not splits:
            splits = list(KFold(3, shuffle=True, random_state=self.seed).split(train))
        for tr, va in splits:
            m = self._fit_models(train.iloc[tr])
            R[va] = self._residuals(m, train.iloc[va])
        self.global_scale = np.nanstd(R, axis=0) + 1e-9

        self.ensembles = {}
        for ch in MONITORED:
            peers = [c for c in MONITORED if c != ch]
            cols = CONTEXT + peers
            self.ensembles[ch] = (
                ExtraTreesRegressor(n_estimators=self.n_estimators,
                                    random_state=self.seed, n_jobs=-1).fit(
                                        train[cols].values, train[ch].values),
                cols)

        self.models = self._fit_models(train)
        Zcal = np.abs(R) / (self._scale_for(train) + 1e-9)
        self.A_cal = np.nanmax(Zcal, axis=1)
        self.threshold = float(np.quantile(self.A_cal, 1 - TARGET_SYSTEM_FA))
        self.stds = train[MONITORED].std()
        pcts = sorted(set(DEV_GRID['saturation']) | set(TEST_GRID['saturation']))
        self.sat_caps = {c: {p: float(np.percentile(train[c].values, p)) for p in pcts}
                         for c in MONITORED}
        return self

    def _scale_for(self, frame):
        cols = []
        for ch in MONITORED:
            ens, inputs = self.ensembles[ch]
            per = np.stack([t.predict(frame[inputs].values) for t in ens.estimators_])
            cols.append(per.std(axis=0))
        sigma_model = np.column_stack(cols)
        return np.sqrt(sigma_model ** 2 + self.global_scale ** 2)

    def z(self, frame):
        R = self._residuals(self.models, frame)
        return np.abs(R) / (self._scale_for(frame) + 1e-9)

    def detect(self, frame):
        return self.z(frame).max(axis=1) > self.threshold

    def reconstruct(self, frame, ch):
        model, cols = self.models[ch]
        return model.predict(frame[cols].values)

    def localize(self, frame):
        z0 = self.z(frame)
        a0 = z0.max(axis=1)
        scores = np.zeros((len(frame), len(MONITORED)), float)
        for j, ch in enumerate(MONITORED):
            fixed = frame.copy()
            fixed[ch] = self.reconstruct(frame, ch)
            a1 = self.z(fixed).max(axis=1)
            scores[:, j] = (a0 - a1) / (a0 + 1e-9)
        order = np.argsort(-scores, axis=1)
        pred = np.array([MONITORED[i] for i in order[:, 0]])
        top = scores[np.arange(len(frame)), order[:, 0]]
        second = scores[np.arange(len(frame)), order[:, 1]]
        return pred, top - second, scores


def inject_target(frame, model, mech, val, direction, seed):
    rng = np.random.RandomState(seed)
    kwargs = {}
    if mech == 'bias':
        kwargs = dict(severity=val, direction=direction)
    elif mech == 'gain':
        kwargs = dict(gain_frac=val, direction=direction)
    elif mech == 'noise':
        kwargs = dict(noise_mult=val)
    elif mech == 'saturation':
        kwargs = dict(cap_percentile=val, cap_value=model.sat_caps[TARGET][val])
    elif mech == 'partial_dropout':
        kwargs = dict(frac=val)
    else:
        raise ValueError(mech)
    return injectors.apply(frame, TARGET, mech, model.stds[TARGET], rng, **kwargs)


def select_margin_on_dev(outer_train, regime, seed):
    margins, correct = [], []
    for i, (tr, va) in enumerate(split_inner(outer_train, regime, n_splits=4, seed=seed)):
        itr, iva = outer_train.iloc[tr].copy(), outer_train.iloc[va].copy()
        if len(iva) < 3:
            continue
        model = GenericConsistencyMonitor(seed=seed + i).fit(itr, regime=regime)
        gate = EnvelopeGate(k=3, quantile=0.95).fit(itr[CONTEXT].values)
        supported = gate.inside(iva[CONTEXT].values)
        for mech in MECHANISMS:
            for val in DEV_GRID[mech]:
                dirs = (+1, -1) if mech in ('bias', 'gain') else (+1,)
                for direction in dirs:
                    corr, manifest = inject_target(
                        iva, model, mech, val, direction,
                        seed=derive_seed(seed, i, mech, val, direction, 'dev_fuel'))
                    det = model.detect(corr)
                    pred, margin, _ = model.localize(corr)
                    keep = det & supported & manifest
                    margins.extend(margin[keep].tolist())
                    correct.extend((pred[keep] == TARGET).astype(int).tolist())
    if len(margins) < 10:
        return -np.inf
    margins = np.asarray(margins, float)
    correct = np.asarray(correct, int)
    best = None
    for thr in np.unique(np.quantile(margins, np.linspace(0, 0.98, 80))):
        acc = margins >= thr
        n = int(acc.sum())
        if n < 10:
            continue
        hits = int(correct[acc].sum())
        if wilson_lower(hits, n) >= TARGET_SOURCE_PRECISION:
            cov = float(acc.mean())
            if best is None or cov > best[1]:
                best = (float(thr), cov)
    return best[0] if best is not None else float('inf')


def run_regime(df, regime, splits, quick=False):
    rows = []
    grid = TEST_GRID
    for fold_id, (tr_idx, te_idx) in enumerate(splits):
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        model = GenericConsistencyMonitor(seed=RNG + fold_id,
                                          n_estimators=60 if quick else 120).fit(train, regime)
        gate = EnvelopeGate(k=3, quantile=0.95).fit(train[CONTEXT].values)
        supported = gate.inside(test[CONTEXT].values)
        margin_thr = select_margin_on_dev(train, regime, seed=RNG + 100 + fold_id)
        true_fuel = test[TARGET].to_numpy(float)

        for mech in MECHANISMS:
            vals = grid[mech][:1] if quick else grid[mech]
            for val in vals:
                dirs = (+1, -1) if mech in ('bias', 'gain') else (+1,)
                for direction in dirs:
                    seed = derive_seed(RNG, fold_id, mech, val, direction, 'test_fuel')
                    corr, manifest = inject_target(test, model, mech, val, direction, seed)
                    det = model.detect(corr)
                    pred, margin, _ = model.localize(corr)
                    accepted = det & supported & manifest & (margin >= margin_thr)

                    before = np.abs(corr[TARGET].to_numpy(float) - true_fuel)
                    fuel_rec = model.reconstruct(corr, TARGET)
                    after = before.copy()
                    act_on_fuel = accepted & (pred == TARGET)
                    after[act_on_fuel] = np.abs(fuel_rec[act_on_fuel] - true_fuel[act_on_fuel])
                    oracle_after = np.abs(fuel_rec - true_fuel)

                    for i in range(len(test)):
                        rows.append(dict(
                            regime=regime, fold=fold_id, base_point_id=int(te_idx[i]),
                            rpm=float(test['Engine Speed'].iloc[i]), mechanism=mech,
                            severity=float(val), direction=int(direction),
                            manifested=int(bool(manifest[i])), supported=int(bool(supported[i])),
                            detected=int(bool(det[i])), predicted_source=str(pred[i]),
                            source_correct=int(pred[i] == TARGET), margin=float(margin[i]),
                            margin_threshold=float(margin_thr), accepted=int(bool(accepted[i])),
                            acted_on_fuel=int(bool(act_on_fuel[i])),
                            error_before=float(before[i]), error_after=float(after[i]),
                            oracle_error_after=float(oracle_after[i]),
                        ))
    return pd.DataFrame(rows)


def summarize(records):
    rows = []
    for regime, s0 in records.groupby('regime'):
        s = s0[s0.manifested == 1].copy()
        if len(s) == 0:
            continue
        det = s.detected.mean()
        sd = s[s.detected == 1]
        src_det = sd.source_correct.mean() if len(sd) else np.nan
        acc = s[s.accepted == 1]
        acc_prec = acc.source_correct.mean() if len(acc) else np.nan
        coverage = s.accepted.mean()
        total_before = s.error_before.sum()
        recovery = 1 - s.error_after.sum() / total_before if total_before > 1e-12 else np.nan
        oracle = 1 - s.oracle_error_after.sum() / total_before if total_before > 1e-12 else np.nan
        harm = (s.error_after > s.error_before + 1e-12).mean()
        rows.append(dict(
            regime=regime, n_manifested=len(s), detection=float(det),
            source_given_detected=float(src_det), selective_coverage=float(coverage),
            source_precision_accepted=float(acc_prec),
            fuel_error_recovered=float(recovery), oracle_fuel_error_recovered=float(oracle),
            reconstruction_harm_rate=float(harm),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default='data/engine_map.csv')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    df, units = load_engine_map(args.data_path)
    if TARGET not in df.columns:
        raise ValueError(f'{TARGET} is absent from {args.data_path}')
    leaked = [c for c in ('BSFC', 'q_MI') if c in MONITORED or c in CONTEXT]
    assert not leaked, f'fuel-derived/closely related variables leaked into predictors: {leaked}'

    df = assign_speed_blocks(df)
    df['rpm_level'] = df['Engine Speed'].round(0)
    regimes = {
        'within_map': list(KFold(5, shuffle=True, random_state=RNG).split(df)),
        'grouped_RPM_holdout': list(GroupKFold(7).split(
            df, groups=df.rpm_level)),
    }
    print('Fuel-flow diagnosis')
    print(f'  target={TARGET}  units={units.get(TARGET, "kg/h")}')
    print(f'  monitored={MONITORED}')

    all_records = []
    for name, splits in regimes.items():
        rec = run_regime(df, name, splits, quick=args.quick)
        all_records.append(rec)
        tmp = summarize(rec)
        if len(tmp):
            r = tmp.iloc[0]
            print(f"  {name:22s} det {r.detection:.1%}  src|det {r.source_given_detected:.1%}  "
                  f"coverage {r.selective_coverage:.1%}  accepted-src {r.source_precision_accepted:.1%}  "
                  f"fuel error recovered {r.fuel_error_recovered:.1%}")

    records = pd.concat(all_records, ignore_index=True)
    summary = summarize(records)
    OUT_RECORD.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    records.to_csv(OUT_RECORD / 'fuel_flow.csv', index=False)
    summary.to_csv(OUT_TABLE / 'fuel_flow.csv', index=False)
    print('\n[saved] results/records/fuel_flow.csv')
    print('[saved] results/fuel_flow.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

