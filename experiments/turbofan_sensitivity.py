import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings('ignore')

from src.external_benchmarks.dlr import DLRTurbofanAdapter, CONTEXT, obs_col, parse_fault_targets, RAW
from src.external_benchmarks.metrics import detection_metrics, wilson_lower

sys.path.insert(0, str(ROOT / 'experiments'))
from importlib import import_module
_e30 = import_module('turbofan')
ConsistencyModel, truth_map = _e30.ConsistencyModel, _e30.truth_map

TARGET_FALSE_ALARM = 0.05
OUT = ROOT / 'results'
BINS = [0, 0.5, 1, 2, 3, 5, 8, 12, 20, np.inf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='DS03', choices=['DS03', 'DS04'])
    ap.add_argument('--max-engines', type=int, default=0)
    args = ap.parse_args()

    ref = DLRTurbofanAdapter(suite='DS02', split='train')
    ref_val = DLRTurbofanAdapter(suite='DS02', split='val')
    tgt = DLRTurbofanAdapter(suite=args.suite, split='test')
    for a, label in ((ref, 'DS02/train'), (ref_val, 'DS02/val'), (tgt, f'{args.suite}/test')):
        if not a.available():
            print(f'missing {label}')
            return

    print(f'Turbofan detection sensitivity ({args.suite})')

    train, val, test = ref.load(), ref_val.load(), tgt.load()
    oracle = tgt.oracle_values().reset_index(drop=True)
    monitored = train.roles.monitored_sensors
    model = ConsistencyModel(CONTEXT, monitored).fit(train.data)
    threshold = float(np.quantile(model.health(val.data), 1 - TARGET_FALSE_ALARM))

    td = test.data.reset_index(drop=True)
    if args.max_engines:
        keep = sorted(td.unit_id.unique())[:args.max_engines]
        m = td.unit_id.isin(keep).values
        td, oracle = td[m].reset_index(drop=True), oracle[m].reset_index(drop=True)

    flags = [c for c in td.columns if c.startswith('fault_') and c.endswith('_on')]
    is_faulted = (td[flags].sum(axis=1) > 0).values
    health = model.health(td)
    alarm = health >= threshold


    tmap = truth_map(args.suite)
    scale = dict(zip([c[:-4] for c in monitored], model.scale))
    magnitude = np.zeros(len(td))
    for i in range(len(td)):
        u = td.unit_id.iloc[i]
        if not is_faulted[i] or u not in tmap:
            continue
        worst = 0.0
        for s in tmap[u]['faulted_sensors']:
            if obs_col(s) in td.columns and s in oracle.columns:
                d = abs(float(td[obs_col(s)].iloc[i]) - float(oracle[s].iloc[i]))
                worst = max(worst, d / (scale.get(s, 1.0) + 1e-12))
        magnitude[i] = worst

    faulted = np.flatnonzero(is_faulted)
    mags = magnitude[faulted]
    print(f'\ntrue corruption magnitude on faulted rows (healthy-residual units)')
    for q in (10, 25, 50, 75, 90, 99):
        print(f'  p{q:<3d} {np.percentile(mags, q):8.2f}')
    print(f'  faulted rows whose corruption is under 1 residual sigma: '
          f'{(mags < 1).mean():.1%}')

    print('\nRecall by fault magnitude')
    print(f"{'magnitude (sigma)':>22s} {'rows':>8s} {'recall':>9s} {'wilson lo':>11s}")
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = faulted[(mags >= lo) & (mags < hi)]
        if len(sel) < 5:
            continue
        rec = float(alarm[sel].mean())
        lower = wilson_lower(int(alarm[sel].sum()), len(sel))
        label = f'{lo:g} to {hi:g}' if np.isfinite(hi) else f'{lo:g}+'
        print(f'{label:>22s} {len(sel):8d} {rec:9.1%} {lower:11.3f}')
        rows.append(dict(suite=args.suite, mag_low=lo, mag_high=(None if np.isinf(hi) else hi),
                         n=len(sel), recall=rec, wilson_lower=lower))


    mdf = next((r['mag_low'] for r in rows if r['wilson_lower'] >= 0.90), None)
    print(f"\n  minimum detectable fault at 90% recall (Wilson lower bound): "
          f"{'%.1f sigma' % mdf if mdf is not None else 'not attained in any bin'}")

    print('\nRecall above and below one residual standard deviation')
    manifested = mags >= 1.0
    if manifested.any():
        print(f'  recall on rows corrupted by at least 1 sigma : '
              f'{alarm[faulted[manifested]].mean():.1%}')
    if (~manifested).any():
        print(f'  recall on rows corrupted by less than 1 sigma: '
              f'{alarm[faulted[~manifested]].mean():.1%}')
    print(f'  pooled recall over all faulted rows          : {alarm[faulted].mean():.1%}')

    print('\nRanking metrics by minimum fault magnitude')
    healthy_idx = np.flatnonzero(~is_faulted)
    for cut in (0.0, 1.0, 3.0):
        sel = faulted[mags >= cut]
        if len(sel) < 20:
            continue
        y = np.r_[np.ones(len(sel)), np.zeros(len(healthy_idx))]
        s = np.r_[health[sel], health[healthy_idx]]
        d = detection_metrics(s, y.astype(int))
        print(f'  faults of at least {cut:>4.1f} sigma: AUROC {d["auroc"]:.3f} '
              f'AUPRC {d["auprc"]:.3f}  (n={len(sel)})')

    if rows:
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / 'turbofan_sensitivity.csv', index=False)
        print('\n[saved] results/turbofan_sensitivity.csv')


if __name__ == '__main__':
    main()

