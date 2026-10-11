import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.covariance import EmpiricalCovariance
import warnings; warnings.filterwarnings('ignore')

from src.external_benchmarks.dlr import (
    DLRTurbofanAdapter, SENSORS, CONTEXT, obs_col, parse_fault_targets, RAW)
from src.external_benchmarks.metrics import (
    detection_metrics, repair_metrics, counterfactual_evidence, restoration_score)
from src.external_benchmarks.splits import assert_no_group_leakage

TARGET_FALSE_ALARM = 0.05
OUT = ROOT / 'results'


def poly_ridge():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(1.0))


def truth_map(suite='DS03'):

    mapping = {}
    for split in ('train', 'val', 'test'):
        folder = RAW / suite / 'raw_csv' / split
        if not folder.exists():
            continue
        for f in folder.glob('*.csv'):
            try:
                info = parse_fault_targets(f.name)
            except ValueError:
                continue
            mapping[info['engine_id']] = info
    return mapping


class ConsistencyModel:


    def __init__(self, context, monitored):
        self.context = list(context)
        self.monitored = list(monitored)

    def fit(self, df):
        self.models = {}
        for ch in self.monitored:
            others = [c for c in self.monitored if c != ch]
            self.models[ch] = poly_ridge().fit(df[self.context + others].values, df[ch].values)
        R = self.residuals(df)
        self.scale = R.std(axis=0) + 1e-9
        self.precision = EmpiricalCovariance().fit(R / self.scale).get_precision()
        return self

    def residuals(self, df):
        return np.column_stack([
            df[ch].values - self.models[ch].predict(
                df[self.context + [c for c in self.monitored if c != ch]].values)
            for ch in self.monitored])

    def z(self, df):
        return np.abs(self.residuals(df)) / self.scale

    def health(self, df):

        return self.z(df).max(axis=1)

    def joint_loglik(self, df):
        Z = self.residuals(df) / self.scale
        return -np.einsum('ij,jk,ik->i', Z, self.precision, Z)

    def reconstruct(self, df, ch):
        others = [c for c in self.monitored if c != ch]
        return self.models[ch].predict(df[self.context + others].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='DS03', choices=['DS03', 'DS04'])
    ap.add_argument('--max-engines', type=int, default=0,
                    help='limit the number of test engines')
    args = ap.parse_args()

    ref = DLRTurbofanAdapter(suite='DS02', split='train')
    ref_val = DLRTurbofanAdapter(suite='DS02', split='val')
    tgt = DLRTurbofanAdapter(suite=args.suite, split='test')
    for a, label in ((ref, 'DS02/train'), (ref_val, 'DS02/val'), (tgt, f'{args.suite}/test')):
        if not a.available():
            print(f'missing {label}; see scripts/download_turbofan_data.py')
            return

    print(f'Turbofan sensor diagnosis ({args.suite})')

    train = ref.load()
    val = ref_val.load()
    test = tgt.load()
    oracle = tgt.oracle_values()

    assert_no_group_leakage(train.data.unit_id.values, test.data.unit_id.values, 'DS02/DS03: ')
    assert_no_group_leakage(train.data.unit_id.values, val.data.unit_id.values, 'DS02 train/val: ')

    monitored = train.roles.monitored_sensors
    model = ConsistencyModel(CONTEXT, monitored).fit(train.data)


    val_health = model.health(val.data)
    threshold = float(np.quantile(val_health, 1 - TARGET_FALSE_ALARM))
    print(f'\nthreshold from DS02 val at {TARGET_FALSE_ALARM:.0%} target false alarm: '
          f'{threshold:.3f}')

    td = test.data.reset_index(drop=True)
    oracle = oracle.reset_index(drop=True)
    if args.max_engines:
        keep_units = sorted(td.unit_id.unique())[:args.max_engines]
        mask = td.unit_id.isin(keep_units).values
        td, oracle = td[mask].reset_index(drop=True), oracle[mask].reset_index(drop=True)
    is_faulted = (test.fault_scope.reset_index(drop=True)[td.index] == 'sensor').values \
        if len(td) == len(test.data) else \
        (td[[c for c in td.columns if c.startswith('fault_') and c.endswith('_on')]].sum(axis=1) > 0).values

    health = model.health(td)
    alarm = health >= threshold

    print('\nDetection on held-out engines')
    det = detection_metrics(health, is_faulted.astype(int), threshold)
    print(f"  AUROC {det['auroc']:.3f} | AUPRC {det['auprc']:.3f}")
    print(f"  recall {det['recall']:.1%} | healthy false alarm {det['healthy_false_alarm']:.1%}")


    tmap = truth_map(args.suite)
    covered = td.unit_id.map(lambda u: u in tmap)
    loc_end_to_end = float('nan')
    loc_given_det = float('nan')
    print(f"\nSensor identification (fault labels available for {covered.mean():.0%} of rows)")
    if covered.any():
        Z = model.z(td)
        pred_idx = Z.argmax(axis=1)
        pred = np.array([monitored[i] for i in pred_idx])
        pred_sensor = np.array([p[:-4] for p in pred])
        hit = np.array([
            (u in tmap) and (s in tmap[u]['faulted_sensors'])
            for u, s in zip(td.unit_id.values, pred_sensor)])
        scored = is_faulted & covered.values
        loc_end_to_end = float(hit[scored].mean()) if scored.any() else float('nan')
        loc_given_det = (float(hit[scored & alarm].mean())
                         if (scored & alarm).any() else float('nan'))
        print(f"  top-1 among faulted rows, any documented target: {loc_end_to_end:.1%}")
        print(f"  conditional on detection: {loc_given_det:.1%}")


    print('\nReconstruction against clean reference readings')
    rows_before, rows_after, repaired_flag = [], [], []
    for i in np.flatnonzero(is_faulted & alarm):
        u = td.unit_id.iloc[i]
        if u not in tmap:
            continue
        ch = pred_sensor[i]
        if ch not in SENSORS:
            continue
        row = td.iloc[[i]]
        obs_val = float(row[obs_col(ch)].iloc[0])
        clean = float(oracle[ch].iloc[i])
        rec = float(model.reconstruct(row, obs_col(ch))[0])
        rows_before.append(abs(obs_val - clean))
        rows_after.append(abs(rec - clean))
        repaired_flag.append(True)
    if rows_before:
        rep = repair_metrics(rows_before, rows_after,
                             error_oracle=np.zeros(len(rows_before)),
                             repaired=repaired_flag)


        tot_before, tot_after = float(np.sum(rows_before)), float(np.sum(rows_after))
        weighted = 1.0 - (tot_after / tot_before) if tot_before > 1e-9 else float('nan')
        print(f"  repairs attempted            {len(rows_before)}")
        print(f"  mean |error| before repair   {rep['mean_error_before']:.4f}")
        print(f"  mean |error| after repair    {rep['mean_error_after']:.4f}")
        print(f"  error-weighted reduction     {weighted:.1%}")
        print(f"  mean per-row error reduction {rep['excess_error_recovered']:.1%}")
        print(f"  corrections increasing error {rep['actual_harm_rate']:.1%}")
    else:
        print('  no repairs attempted')

    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(dataset=f'dlr_{args.suite.lower()}', fault_origin='native',
                   source_given_detected=loc_given_det,
                   end_to_end_source=loc_end_to_end,
                   fault_scope='sensor', n_test_rows=int(len(td)),
                   n_test_engines=int(td.unit_id.nunique()),
                   threshold=threshold, **{k: v for k, v in det.items()})
    if rows_before:
        summary.update({
            'repair_coverage': rep['repair_coverage'],
            'repair_mean_error_before': rep['mean_error_before'],
            'repair_mean_error_after': rep['mean_error_after'],
            'repair_actual_harm_rate': rep['actual_harm_rate'],
            'repair_excess_error_recovered': rep['excess_error_recovered'],
        })
        summary['repair_recovered_error_weighted'] = weighted
    pd.DataFrame([summary]).to_csv(OUT / 'turbofan_detection.csv', index=False)
    print("\n[saved] results/turbofan_detection.csv")


if __name__ == '__main__':
    main()

