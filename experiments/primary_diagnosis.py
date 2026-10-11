import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import warnings; warnings.filterwarnings('ignore')

from src.data.load_engine_map import load_engine_map, assign_speed_blocks
from src.evaluation.episode_protocol import (
    fit_nested_correctness_calibrator, fit_nested_correctness_calibrator_joint,
    correctness_features, complete_correct,
    mech_features,
    train_joint_typer,
    EpisodeConfig, FoldModels, prepare_inner_oof, run_inner_oof, calibrate_k_alarm,
    score_scenarios, episodes_from_scenarios, train_deployment_classifier,
    oof_confidence_dataset, tune_confidence, derive_seed, wilson_lower, TEST_GRID, MECHANISMS)

RNG = 42
MODE = 'diverse'
TARGET_PRECISION = 0.90


VARIANTS = [
    'original_residual_adjustment',
    'uncertainty_residual_adjustment',
    'selected_reporting_model',
]


def variant_flags(v):

    return dict(
        shi_norm='region' if v == 'original_residual_adjustment' else 'ensemble',


        isolator='structured',
        joint_typing=False,
        correctness_cal=v == 'selected_reporting_model',
    )


def select_threshold_on_dev(scores, correct, target=0.90, min_acc=20):


    scores = np.asarray(scores, float)
    correct = np.asarray(correct, int)
    best = None
    if len(scores) == 0:
        return float('inf')
    for thr in np.unique(np.quantile(scores, np.linspace(0, 0.99, 60))):
        acc = scores >= thr
        n = int(acc.sum())
        if n < min_acc:
            continue
        if wilson_lower(int(correct[acc].sum()), n) >= target:
            cov = float(acc.mean())
            if best is None or cov > best[1]:
                best = (float(thr), cov)
    return best[0] if best else float('inf')


def evaluate_variant(df, regime, splits, N, cfg, flags, variant):

    cfg_v = replace(cfg, shi_normalization=flags['shi_norm'],
                    isolator=flags['isolator'],
                    mechanism_model='joint' if flags['joint_typing'] else 'episode_rf')
    skipped_folds = []
    det_, src_, typ_, comp_, fa_ = [], [], [], [], []
    src_given_det, typ_given_det = [], []
    acc_flags, acc_correct = [], []

    for fold_id, (tr_idx, te_idx) in enumerate(splits):
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        prepared = prepare_inner_oof(train, regime, cfg_v, fold_id)
        if not prepared:
            continue
        inner = run_inner_oof(prepared, regime, cfg_v, N, MODE, fold_id)
        inner_f = [r for r in inner if r['true_src'] != 'none']
        if len(inner_f) < 40:
            skipped_folds.append(f'fold {fold_id}: only {len(inner_f)} inner episodes')
            continue
        healthy_inner = np.array([r['n_suspicious'] for r in inner if r['mech'] == 'healthy'])
        k_alarm = calibrate_k_alarm(healthy_inner, N, cfg_v)['k_alarm']
        clf = train_deployment_classifier(inner_f, cfg_v)


        if flags['joint_typing']:
            nested = fit_nested_correctness_calibrator_joint(
                inner_f, k_alarm, seed=derive_seed(cfg_v.seed, fold_id, 'cal'))
        else:
            nested = fit_nested_correctness_calibrator(
                inner_f, k_alarm, seed=derive_seed(cfg_v.seed, fold_id, 'cal'),
                model='logistic')
        if nested is None:
            skipped_folds.append(f'fold {fold_id}: calibration routine returned nothing '
                                 f'({len(inner_f)} inner episodes, possibly single-class)')
            continue


        joint = nested.get('joint') if flags['joint_typing'] else None
        cal = nested['calibrator'] if flags['correctness_cal'] else None
        dev_scores = nested['oof_calibrated'] if cal is not None else nested['oof_baseline']
        thr = select_threshold_on_dev(dev_scores, nested['correct'], TARGET_PRECISION)


        final = FoldModels(seed=derive_seed(cfg_v.seed, fold_id, 'final'),
                           shi_normalization=flags['shi_norm'],
                           isolator=flags['isolator']).fit(train)
        te_scen = score_scenarios(final, test, TEST_GRID, cfg_v, regime=regime,
                                  outer_fold=fold_id, stage='outer')
        te_recs = episodes_from_scenarios(final, te_scen, N, MODE, cfg_v, regime=regime,
                                          outer_fold=fold_id, stage='outer',
                                          row_ids=test.index.values)
        faulted = [r for r in te_recs if r['mech'] != 'healthy' and r['true_src'] != 'none']
        if not faulted:
            continue
        for r in faulted:
            fp = np.array(mech_features(r)).reshape(1, -1)
            detected = r['n_suspicious'] >= k_alarm
            supported = r['support_frac'] > 0
            if joint is not None:
                j_src, j_mech = joint.predict_pair(r)
                pal_joint = joint.predict_proba_mech(r)
                pm = j_mech if detected else 'none'
                src_ok = detected and (j_src == r['true_src'])
                proba = None
            else:
                proba = clf.predict_proba(fp)[0]
                pm = clf.classes_[int(np.argmax(proba))] if detected else 'none'
                src_ok = detected and (r['pred_src'] == r['true_src'])
            typ_ok = detected and (pm == r['mech'])
            comp = bool(detected and supported and src_ok and typ_ok)
            if proba is None:
                pal = pal_joint
            else:
                cols = [list(clf.classes_).index(c) if c in clf.classes_ else None
                        for c in MECHANISMS]
                pal = np.zeros(len(MECHANISMS))
                for j, ci in enumerate(cols):
                    if ci is not None:
                        pal[j] = proba[ci]
            if cal is not None:
                score = float(cal.predict_proba(
                    np.array(correctness_features(r, pal)).reshape(1, -1))[0, 1])
            else:
                score = float(pal.max()) if detected else 0.0
            det_.append(int(detected)); src_.append(int(src_ok))
            typ_.append(int(typ_ok)); comp_.append(int(comp))
            if detected:
                src_given_det.append(int(src_ok)); typ_given_det.append(int(typ_ok))
            acc_flags.append(int(score >= thr)); acc_correct.append(int(comp))
        for r in [x for x in te_recs if x['mech'] == 'healthy']:
            fa_.append(int(r['n_suspicious'] >= k_alarm))

    if not det_:

        reason = '; '.join(skipped_folds[:3]) if skipped_folds else 'no scorable test episodes'
        print(f"    [{variant}] produced no rows -> {reason}")
        return None
    acc = np.array(acc_flags, bool); cr = np.array(acc_correct)
    sel_cov = float(acc.mean())
    sel_prec = float(cr[acc].mean()) if acc.any() else float('nan')
    sel_lo = wilson_lower(int(cr[acc].sum()), int(acc.sum())) if acc.any() else float('nan')
    return dict(
        variant=variant, regime=regime, N=N,
        healthy_FA=float(np.mean(fa_)) if fa_ else float('nan'),
        detection=float(np.mean(det_)),


        end_to_end_source=float(np.mean(src_)), end_to_end_type=float(np.mean(typ_)),
        source_given_detected=float(np.mean(src_given_det)) if src_given_det else float('nan'),
        type_given_detected=float(np.mean(typ_given_det)) if typ_given_det else float('nan'),
        complete=float(np.mean(comp_)),
        selective_coverage=sel_cov, selective_precision=sel_prec, selective_lower=sel_lo,
        n=len(det_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='run twenty point batches only')
    args = ap.parse_args()

    df, _ = load_engine_map()
    df = assign_speed_blocks(df)
    df['rpm_level'] = df['Engine Speed'].round(0)
    regimes = {
        'within_map': list(KFold(5, shuffle=True, random_state=RNG).split(df)),
        'grouped_RPM_holdout': list(GroupKFold(7).split(df, groups=df['rpm_level'])),
    }
    cfg = EpisodeConfig(episode_sizes=(5, 20), modes=('diverse',),
                        n_episodes=15, n_healthy_episodes=120, n_realizations=2,
                        n_inner=4, n_boot=200)

    runnable = VARIANTS
    sizes = [20] if args.quick else [5, 20]

    print("Primary diagnosis comparison" + ("  (--quick)" if args.quick else ""))
    rows = []
    for v in runnable:
        flags = variant_flags(v)
        for regime, splits in regimes.items():
            for N in sizes:
                res = evaluate_variant(df, regime, splits, N, cfg, flags, v)
                if res is None:
                    continue
                rows.append(res)
                print(f"  {v:20s} {regime:22s} N={N:<3d} "
                      f"FA {res['healthy_FA']:.1%}  det {res['detection']:.1%}  "
                      f"src|det {res['source_given_detected']:.1%}  "
                      f"complete {res['complete']:.1%}  "
                      f"sel-cov {res['selective_coverage']:.1%} "
                      f"sel-prec {res['selective_precision']:.3f}")

    if not rows:
        print("\nNo variant produced results.")
        return
    out = pd.DataFrame(rows)
    Path('results').mkdir(parents=True, exist_ok=True)
    out.to_csv('results/primary_diagnosis.csv', index=False)

    print("\nWithin-map comparison (N=20)")
    hdr = (f"{'variant':22s} {'healthyFA':>10s} {'detect':>8s} {'src|det':>8s} {'mech|det':>9s} "
           f"{'complete':>9s} {'sel cov':>8s} {'sel prec':>9s}")
    print(hdr)
    for v in runnable:
        d = out[(out.variant == v) & (out.regime == 'within_map') & (out.N == 20)]
        if not len(d):
            continue
        r = d.iloc[0]
        print(f"{v:22s} {r['healthy_FA']:10.1%} {r['detection']:8.1%} "
              f"{r['source_given_detected']:8.1%} {r['type_given_detected']:9.1%} "
              f"{r['complete']:9.1%} {r['selective_coverage']:8.1%} "
              f"{r['selective_precision']:9.3f}")
    print("\n[saved] results/primary_diagnosis.csv")


if __name__ == '__main__':
    main()

