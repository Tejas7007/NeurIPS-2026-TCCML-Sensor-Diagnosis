import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings('ignore')

from src.external_benchmarks.dlr import DLRTurbofanAdapter, CONTEXT, obs_col
from src.external_benchmarks.metrics import detection_metrics
from src.external_benchmarks.splits import assert_no_group_leakage

sys.path.insert(0, str(ROOT / 'experiments'))
from importlib import import_module
_e30 = import_module('turbofan')
ConsistencyModel, truth_map = _e30.ConsistencyModel, _e30.truth_map

TARGET_FALSE_ALARM = 0.05
N_BOOT = 400
OUT = ROOT / 'results'


def metrics_on(units, per_engine):

    heal_alarm = np.concatenate([per_engine[u]['heal_alarm'] for u in units]) \
        if any(len(per_engine[u]['heal_alarm']) for u in units) else np.zeros(0)
    f_alarm = np.concatenate([per_engine[u]['f_alarm'] for u in units]) \
        if any(len(per_engine[u]['f_alarm']) for u in units) else np.zeros(0)
    health = np.concatenate([per_engine[u]['health'] for u in units])
    label = np.concatenate([per_engine[u]['label'] for u in units])
    src = np.concatenate([per_engine[u]['src_ok'] for u in units]) \
        if any(len(per_engine[u]['src_ok']) for u in units) else np.zeros(0)
    eb = np.concatenate([per_engine[u]['err_before'] for u in units]) \
        if any(len(per_engine[u]['err_before']) for u in units) else np.zeros(0)
    ea = np.concatenate([per_engine[u]['err_after'] for u in units]) \
        if any(len(per_engine[u]['err_after']) for u in units) else np.zeros(0)

    out = {}
    if len(np.unique(label)) == 2:
        out['detection_auroc'] = detection_metrics(health, label)['auroc']
    else:
        out['detection_auroc'] = np.nan
    out['detection_recall'] = float(f_alarm.mean()) if len(f_alarm) else np.nan
    out['healthy_false_alarm'] = float(heal_alarm.mean()) if len(heal_alarm) else np.nan
    out['source_given_detected'] = float(src.mean()) if len(src) else np.nan
    if len(eb):
        tb, ta = float(eb.sum()), float(ea.sum())
        out['repair_recovered'] = 1 - ta / tb if tb > 1e-9 else np.nan
        out['repair_harm'] = float(np.mean(ea > eb))
    else:
        out['repair_recovered'] = out['repair_harm'] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='DS03', choices=['DS03', 'DS04'])
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    args = ap.parse_args()

    ref, ref_val = DLRTurbofanAdapter('DS02', 'train'), DLRTurbofanAdapter('DS02', 'val')
    tgt = DLRTurbofanAdapter(args.suite, 'test')
    for a, label in ((ref, 'DS02/train'), (ref_val, 'DS02/val'), (tgt, f'{args.suite}/test')):
        if not a.available():
            print(f'missing {label}')
            return

    print(f'Turbofan engine sampling intervals ({args.suite})')

    train, val, test = ref.load(), ref_val.load(), tgt.load()
    oracle = tgt.oracle_values().reset_index(drop=True)
    monitored = train.roles.monitored_sensors
    model = ConsistencyModel(CONTEXT, monitored).fit(train.data)
    threshold = float(np.quantile(model.health(val.data), 1 - TARGET_FALSE_ALARM))

    td = test.data.reset_index(drop=True)
    assert_no_group_leakage(train.data.unit_id.values, td.unit_id.values, 'DS02/target: ')
    health = model.health(td)
    alarm = health >= threshold
    z = model.z(td)
    flags = [c for c in td.columns if c.startswith('fault_') and c.endswith('_on')]
    faulted = (td[flags].sum(axis=1) > 0).values
    tmap = truth_map(args.suite)

    per_engine = {}
    for unit, sub in td.groupby('unit_id', sort=False):
        ix = sub.index.values
        f = faulted[ix]
        truth = set(tmap.get(int(unit), {}).get('faulted_sensors', []))
        acc = dict(heal_alarm=alarm[ix][~f].astype(float),
                   f_alarm=alarm[ix][f].astype(float),
                   health=health[ix], label=f.astype(int),
                   src_ok=[], err_before=[], err_after=[])
        for i in ix[f & alarm[ix]]:
            pick = monitored[int(z[i].argmax())][:-4]
            acc['src_ok'].append(float(pick in truth))
            if pick in oracle.columns:
                clean = float(oracle[pick].iloc[i])
                acc['err_before'].append(abs(float(td[obs_col(pick)].iloc[i]) - clean))
                acc['err_after'].append(abs(float(
                    model.reconstruct(td.iloc[[i]], obs_col(pick))[0]) - clean))
        for k in ('src_ok', 'err_before', 'err_after'):
            acc[k] = np.array(acc[k], float)
        per_engine[unit] = acc

    units = np.array(list(per_engine))
    point = metrics_on(units, per_engine)

    rs = np.random.RandomState(0)
    draws = {k: [] for k in point}
    for _ in range(args.n_boot):
        sample = rs.choice(units, len(units), replace=True)
        m = metrics_on(sample, per_engine)
        for k, v in m.items():
            draws[k].append(v)

    print(f'\n{len(units)} test engines, {args.n_boot} engine-level resamples\n')
    print(f"{'metric':26s} {'estimate':>10s} {'95% interval':>22s} {'width':>8s}")
    rows = []
    for k in ['detection_auroc', 'detection_recall', 'healthy_false_alarm',
              'source_given_detected', 'repair_recovered', 'repair_harm']:
        arr = np.array([d for d in draws[k] if np.isfinite(d)])
        if len(arr) < 10:
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5])
        fmt = (lambda v: f'{v:.3f}') if 'auroc' in k else (lambda v: f'{v:.1%}')
        print(f'{k:26s} {fmt(point[k]):>10s}   [{fmt(lo)}, {fmt(hi)}] {fmt(hi - lo):>8s}')
        rows.append(dict(suite=args.suite, metric=k, estimate=point[k],
                         ci_low=float(lo), ci_high=float(hi), n_engines=len(units),
                         n_boot=args.n_boot))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'turbofan_intervals.csv', index=False)
    print('\n[saved] results/turbofan_intervals.csv')


if __name__ == '__main__':
    main()

