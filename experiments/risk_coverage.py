import sys
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
import warnings; warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, GroupKFold
from src.data.load_engine_map import load_engine_map, assign_speed_blocks
from src.evaluation.episode_protocol import (
    fit_nested_correctness_calibrator, correctness_features, mech_features,
    EpisodeConfig, FoldModels, prepare_inner_oof, run_inner_oof, calibrate_k_alarm,
    score_scenarios, episodes_from_scenarios, train_deployment_classifier,
    derive_seed, TEST_GRID, MECHANISMS)

ablation = import_module('primary_diagnosis')
RNG = 42
MODE = 'diverse'
VARIANT = 'selected_reporting_model'
OUT_RECORD = ROOT / 'results' / 'records'
OUT_TABLE = ROOT / 'results'
OUT_FIG = ROOT / 'results' / 'figures'


def run_one(df, regime, splits, N, cfg):
    records = []
    flags = ablation.variant_flags(VARIANT)
    for fold_id, (tr_idx, te_idx) in enumerate(splits):
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        prepared = prepare_inner_oof(train, regime, cfg, fold_id)
        if not prepared:
            print(f'  fold {fold_id}: no inner folds, skipped')
            continue
        inner = run_inner_oof(prepared, regime, cfg, N, MODE, fold_id)
        inner_f = [r for r in inner if r['true_src'] != 'none']
        if len(inner_f) < 40:
            print(f'  fold {fold_id}: only {len(inner_f)} inner fault episodes, skipped')
            continue
        healthy_inner = np.array([r['n_suspicious'] for r in inner if r['mech'] == 'healthy'])
        k_alarm = calibrate_k_alarm(healthy_inner, N, cfg)['k_alarm']
        clf = train_deployment_classifier(inner_f, cfg)
        nested = fit_nested_correctness_calibrator(
            inner_f, k_alarm, seed=derive_seed(cfg.seed, fold_id, 'cal'), model='logistic')
        if nested is None:
            print(f'  fold {fold_id}: correctness calibrator unavailable, skipped')
            continue
        cal = nested['calibrator']
        frozen_thr = ablation.select_threshold_on_dev(
            nested['oof_calibrated'], nested['correct'], ablation.TARGET_PRECISION)

        final = FoldModels(seed=derive_seed(cfg.seed, fold_id, 'final'),
                           shi_normalization=flags['shi_norm'],
                           isolator=flags['isolator']).fit(train)
        te_scen = score_scenarios(final, test, TEST_GRID, cfg, regime=regime,
                                  outer_fold=fold_id, stage='outer')
        te_recs = episodes_from_scenarios(
            final, te_scen, N, MODE, cfg, regime=regime, outer_fold=fold_id,
            stage='outer', row_ids=test.index.values)

        for r in te_recs:
            if r['mech'] == 'healthy' or r['true_src'] == 'none':
                continue
            fp = np.asarray(mech_features(r), float).reshape(1, -1)
            detected = bool(r['n_suspicious'] >= k_alarm)
            supported = bool(r['support_frac'] > 0)
            proba = clf.predict_proba(fp)[0]
            pred_mech = clf.classes_[int(np.argmax(proba))] if detected else 'none'
            src_ok = bool(detected and r['pred_src'] == r['true_src'])
            type_ok = bool(detected and pred_mech == r['mech'])
            complete = bool(detected and supported and src_ok and type_ok)

            pal = np.zeros(len(MECHANISMS), float)
            for j, mech in enumerate(MECHANISMS):
                if mech in clf.classes_:
                    pal[j] = proba[list(clf.classes_).index(mech)]
            score = float(cal.predict_proba(
                np.asarray(correctness_features(r, pal), float).reshape(1, -1))[0, 1])
            records.append(dict(
                regime=regime, N=N, fold=fold_id, scen_id=r['scen_id'],
                true_src=r['true_src'], mechanism=r['mech'], severity=r['severity'],
                direction=r['direction'], realization_id=r['realization_id'],
                episode_idx=r['episode_idx'], row_ids=r['row_ids'],
                detected=int(detected), supported=int(supported),
                source_correct=int(src_ok), type_correct=int(type_ok),
                complete_correct=int(complete), correctness_score=score,
                frozen_threshold=float(frozen_thr),
                accepted_frozen=int(score >= frozen_thr),
            ))
    return pd.DataFrame(records)


def make_curve(records, n_grid=151):
    rows = []
    for (regime, N), s in records.groupby(['regime', 'N']):
        scores = s.correctness_score.to_numpy(float)
        correct = s.complete_correct.to_numpy(int)
        if len(scores) == 0:
            continue
        thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, n_grid)))
        for thr in thresholds:
            acc = scores >= thr
            n = int(acc.sum())
            rows.append(dict(
                regime=regime, N=int(N), threshold=float(thr),
                coverage=float(acc.mean()),
                precision=float(correct[acc].mean()) if n else np.nan,
                risk=float(1 - correct[acc].mean()) if n else np.nan,
                n_accepted=n, frozen_operating_point=False,
            ))
        frozen = s.accepted_frozen.astype(bool).to_numpy()
        n = int(frozen.sum())
        rows.append(dict(
            regime=regime, N=int(N), threshold=np.nan,
            coverage=float(frozen.mean()),
            precision=float(correct[frozen].mean()) if n else np.nan,
            risk=float(1 - correct[frozen].mean()) if n else np.nan,
            n_accepted=n, frozen_operating_point=True,
        ))
    return pd.DataFrame(rows)


def summarize(records):
    rows = []
    for (regime, N), frame in records.groupby(['regime', 'N']):
        detected = frame.detected.astype(bool)
        accepted = frame.accepted_frozen.astype(bool)
        rows.append(dict(
            regime=regime,
            N=int(N),
            detection=float(frame.detected.mean()),
            source_given_detected=float(frame.loc[detected, 'source_correct'].mean()),
            reporting_rate=float(frame.accepted_frozen.mean()),
            complete_correct_given_report=float(
                frame.loc[accepted, 'complete_correct'].mean()),
            n=int(len(frame)),
        ))
    return pd.DataFrame(rows)


def summarize_mechanisms(records):
    rows = []
    selected = records[
        (records.regime == 'grouped_RPM_holdout') & (records.N == 20)
    ]
    for mechanism, frame in selected.groupby('mechanism'):
        detected = frame.detected.astype(bool)
        accepted = frame.accepted_frozen.astype(bool)
        rows.append(dict(
            mechanism=mechanism,
            n=int(len(frame)),
            detection=float(frame.detected.mean()),
            source_given_detected=float(frame.loc[detected, 'source_correct'].mean()),
            reporting_rate=float(frame.accepted_frozen.mean()),
            complete_correct_given_report=float(
                frame.loc[accepted, 'complete_correct'].mean()),
        ))
    table = pd.DataFrame(rows)
    pooled_detected = selected.detected.astype(bool)
    pooled_accepted = selected.accepted_frozen.astype(bool)
    pooled = pd.DataFrame([dict(
        mechanism='pooled',
        n=int(len(selected)),
        detection=float(selected.detected.mean()),
        source_given_detected=float(
            selected.loc[pooled_detected, 'source_correct'].mean()),
        reporting_rate=float(selected.accepted_frozen.mean()),
        complete_correct_given_report=float(
            selected.loc[pooled_accepted, 'complete_correct'].mean()),
    )])
    equal = pd.DataFrame([dict(
        mechanism='equal_weighted',
        n=int(len(selected)),
        detection=float(table.detection.mean()),
        source_given_detected=float(table.source_given_detected.mean()),
        reporting_rate=float(table.reporting_rate.mean()),
        complete_correct_given_report=float(
            table.complete_correct_given_report.mean()),
    )])
    return pd.concat([table, pooled, equal], ignore_index=True)


def save_figure(curve):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'[figure skipped] matplotlib unavailable: {exc}')
        return
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    for regime in sorted(curve.regime.unique()):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        sub = curve[(curve.regime == regime) & (~curve.frozen_operating_point)]
        for N in sorted(sub.N.unique()):
            d = sub[sub.N == N].sort_values('coverage')
            ax.plot(d.coverage, d.precision, label=f'N={N}')
        frozen = curve[(curve.regime == regime) & (curve.frozen_operating_point)]
        for _, r in frozen.iterrows():
            ax.scatter([r.coverage], [r.precision], s=55,
                       label=f'frozen N={int(r.N)}')
        ax.set_xlabel('selective coverage')
        ax.set_ylabel('complete-diagnosis precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = OUT_FIG / f'41_e0_risk_coverage_{regime}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f'[saved] {path.relative_to(ROOT)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default='data/engine_map.csv')
    ap.add_argument('--quick', action='store_true',
                    help='run held out speed at twenty points only')
    args = ap.parse_args()

    df, _ = load_engine_map(args.data_path)
    df = assign_speed_blocks(df)
    df['rpm_level'] = df['Engine Speed'].round(0)
    all_regimes = {
        'within_map': list(KFold(5, shuffle=True, random_state=RNG).split(df)),
        'grouped_RPM_holdout': list(GroupKFold(7).split(
            df, groups=df.rpm_level)),
    }
    regimes = ({'grouped_RPM_holdout': all_regimes['grouped_RPM_holdout']}
               if args.quick else all_regimes)
    sizes = [20] if args.quick else [5, 20]
    cfg = EpisodeConfig(episode_sizes=tuple(sizes), modes=(MODE,),
                        n_episodes=15, n_healthy_episodes=120, n_realizations=2,
                        n_inner=4, n_boot=200, shi_normalization='ensemble',
                        isolator='structured', mechanism_model='episode_rf')

    print('Reporting threshold curves')
    chunks = []
    for regime, splits in regimes.items():
        for N in sizes:
            print(f'\n[{regime}, N={N}]')
            rec = run_one(df, regime, splits, N, cfg)
            if len(rec):
                chunks.append(rec)
                frozen = rec.accepted_frozen.astype(bool)
                print(f"  frozen coverage {frozen.mean():.1%}; precision "
                      f"{rec.loc[frozen, 'complete_correct'].mean():.1%}" if frozen.any()
                      else '  frozen policy accepted no episodes')
    if not chunks:
        print('no records produced')
        return 0

    records = pd.concat(chunks, ignore_index=True)
    curve = make_curve(records)
    summary = summarize(records)
    mechanisms = summarize_mechanisms(records)
    OUT_RECORD.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    records.to_csv(OUT_RECORD / 'risk_coverage.csv', index=False)
    curve.to_csv(OUT_TABLE / 'risk_coverage.csv', index=False)
    summary.to_csv(OUT_TABLE / 'primary_results.csv', index=False)
    mechanisms.to_csv(OUT_TABLE / 'mechanism_breakdown.csv', index=False)
    save_figure(curve)
    print('\n[saved] results/records/risk_coverage.csv')
    print('[saved] results/risk_coverage.csv')
    print('[saved] results/primary_results.csv')
    print('[saved] results/mechanism_breakdown.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

