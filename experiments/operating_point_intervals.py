import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / 'results' / 'records' / 'risk_coverage.csv'
OUT = ROOT / 'results' / 'operating_point_intervals.csv'


def parse_ids(value):
    if pd.isna(value) or str(value).strip() == '':
        return set()
    return {int(x) for x in str(value).split(';') if str(x).strip() != ''}


def metric_value(frame, metric):
    if len(frame) == 0:
        return np.nan
    if metric == 'detection':
        return float(frame.detected.mean())
    if metric == 'source_given_detected':
        s = frame[frame.detected == 1]
        return float(s.source_correct.mean()) if len(s) else np.nan
    if metric == 'complete_correct_rate':
        return float(frame.complete_correct.mean())
    if metric == 'selective_coverage':
        return float(frame.accepted_frozen.mean())
    if metric == 'accepted_precision':
        s = frame[frame.accepted_frozen == 1]
        return float(s.complete_correct.mean()) if len(s) else np.nan
    raise ValueError(metric)


def jackknife(frame, metric):
    memberships = [parse_ids(v) for v in frame.row_ids.values]
    points = sorted(set().union(*memberships)) if memberships else []
    full = metric_value(frame, metric)
    if len(points) < 3 or not np.isfinite(full):
        return dict(estimate=full, se=np.nan, ci_low=np.nan, ci_high=np.nan,
                    n_points=len(points), n_episodes=len(frame), min_leave_one=np.nan,
                    max_leave_one=np.nan)

    vals = []
    for point in points:
        keep = np.array([point not in ids for ids in memberships], dtype=bool)
        theta = metric_value(frame.loc[keep], metric)
        if np.isfinite(theta):
            vals.append(theta)
    vals = np.asarray(vals, float)
    if len(vals) < 3:
        return dict(estimate=full, se=np.nan, ci_low=np.nan, ci_high=np.nan,
                    n_points=len(points), n_episodes=len(frame), min_leave_one=np.nan,
                    max_leave_one=np.nan)

    mean_j = float(vals.mean())
    m = len(vals)
    se = float(np.sqrt((m - 1) / m * np.sum((vals - mean_j) ** 2)))
    lo = max(0.0, full - 1.96 * se)
    hi = min(1.0, full + 1.96 * se)
    return dict(estimate=full, se=se, ci_low=lo, ci_high=hi,
                n_points=len(points), n_episodes=len(frame),
                min_leave_one=float(vals.min()), max_leave_one=float(vals.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(INFILE))
    ap.add_argument('--quick', action='store_true',
                    help='only grouped-RPM N=20')
    args = ap.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f'{path} not found; run experiments/risk_coverage.py first')
    df = pd.read_csv(path)
    required = {'regime', 'N', 'row_ids', 'detected', 'source_correct',
                'complete_correct', 'accepted_frozen'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Risk coverage record file missing columns: {sorted(missing)}')

    if args.quick:
        df = df[(df.regime == 'grouped_RPM_holdout') & (df.N == 20)]

    metrics = ['detection', 'source_given_detected', 'complete_correct_rate',
               'selective_coverage', 'accepted_precision']
    rows = []
    print('Operating-point sensitivity')
    for (regime, N), s in df.groupby(['regime', 'N']):
        print(f'\n[{regime}, N={N}]')
        for metric in metrics:
            res = jackknife(s, metric)
            rows.append(dict(regime=regime, N=int(N), metric=metric, **res))
            print(f"  {metric:24s} {res['estimate']:.3f}  interval "
                  f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  "
                  f"points={res['n_points']}")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f'\n[saved] {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

