import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import KFold
import warnings; warnings.filterwarnings('ignore')

from src.data.load_engine_map import load_engine_map, assign_speed_blocks
from src.health.shi import SHI, MONITORED
from src.faults import injectors

RNG = 42
DIRECTIONS = [+1, -1]
UNITS = {'T_IM': 'degC', 'P_IM': 'bar', 'EGR_Rate': '%', 'MF_IA': 'kg/h', 'NOx_EO': 'ppm'}
BIAS_SEV = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
GAIN_FRAC = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]
DROPOUT_FRAC = [0.05, 0.1, 0.2, 0.4]
NOISE_MULT = [0.5, 1.0, 2.0, 3.0]


def detection_rate_for(shi, test, ch, ftype, stds, rng, **kw):

    corr, mask = injectors.apply(test, ch, ftype, stds[ch], rng, **kw)
    det = shi.detect(corr)

    if mask.any():
        return det[mask].mean()
    return det.mean()


def main():
    df, _ = load_engine_map()
    df = assign_speed_blocks(df)
    within = list(KFold(5, shuffle=True, random_state=RNG).split(df))

    print("Bias detection sensitivity")


    print("\nDetection by bias magnitude and sensor")
    header = "  bias(std) " + " ".join(f"{c:>9s}" for c in MONITORED)
    print(header)

    bias_hits = {ch: {s: 0 for s in BIAS_SEV} for ch in MONITORED}
    bias_total = {ch: {s: 0 for s in BIAS_SEV} for ch in MONITORED}
    fold_phys = {ch: [] for ch in MONITORED}
    for tr_idx, te_idx in within:
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        shi = SHI(system_fpr=0.05, seed=RNG).fit(train)
        stds = train[MONITORED].std()
        rng = np.random.RandomState(0)
        for ch in MONITORED:
            fold_phys[ch].append(stds[ch])
            for s in BIAS_SEV:
                for d in DIRECTIONS:
                    corr, mask = injectors.apply(test, ch, 'bias', stds[ch], rng,
                                                 severity=s, direction=d)
                    det = shi.detect(corr)
                    sel = mask if mask.any() else np.ones(len(det), bool)
                    bias_hits[ch][s] += int(det[sel].sum())
                    bias_total[ch][s] += int(sel.sum())
    def rate(ch, s):
        return bias_hits[ch][s] / max(bias_total[ch][s], 1)
    for s in BIAS_SEV:
        row = [f"{rate(ch, s):9.2f}" for ch in MONITORED]
        print(f"  {s:8.2f}  " + " ".join(row))


    print("\nMinimum tested bias at 90% pooled detection (5% alarm target)")
    print(f"    {'channel':9s} {'std':>6s} {'phys (median)':>16s} {'phys range across folds':>28s}")
    mdf_rows = []
    for ch in MONITORED:
        hit = next((s for s in BIAS_SEV if rate(ch, s) >= 0.90), None)
        phys_arr = np.array(fold_phys[ch])
        if hit is not None:
            pmed = hit * np.median(phys_arr)
            plo, phi = hit * phys_arr.min(), hit * phys_arr.max()
            print(f"    {ch:9s} {hit:6.2f} {pmed:12.3f} {UNITS[ch]:3s} "
                  f"[{plo:.3f}, {phi:.3f}] {UNITS[ch]}")
            mdf_rows.append({'channel': ch, 'min_bias_std': hit,
                             'phys_median': pmed, 'phys_min': plo, 'phys_max': phi,
                             'unit': UNITS[ch], 'pooled_detection': rate(ch, hit)})
        else:
            print(f"    {ch:9s}   >{BIAS_SEV[-1]:.2f} std (not >=90% pooled in tested range)")
            mdf_rows.append({'channel': ch, 'min_bias_std': np.nan, 'phys_median': np.nan,
                             'phys_min': np.nan, 'phys_max': np.nan, 'unit': UNITS[ch],
                             'pooled_detection': rate(ch, BIAS_SEV[-1])})


    print("\nDetection by fault type")
    print(f"{'fault type':16s} {'setting':>14s} {'mean detection':>16s}")
    ft_rows = []
    for tr_idx, te_idx in within:
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        shi = SHI(system_fpr=0.05, seed=RNG).fit(train)
        stds = train[MONITORED].std()
        rng = np.random.RandomState(0)

        sat_cap_85 = {ch: np.percentile(train[ch].values, 85) for ch in MONITORED}

        for ftype, settings in [
            ('bias', [('severity', 1.0)]),
            ('gain', [('gain_frac', 0.12)]),
            ('saturation', [('cap_percentile', 85)]),
            ('partial_dropout', [('frac', 0.2)]),
            ('noise', [('noise_mult', 2.0)]),
        ]:
            for pname, pval in settings:
                dets = []
                for ch in MONITORED:
                    kw = {pname: pval}
                    if ftype == 'bias':
                        kw['direction'] = +1
                    if ftype == 'saturation':
                        kw['cap_value'] = sat_cap_85[ch]
                    dets.append(detection_rate_for(shi, test, ch, ftype, stds, rng, **kw))
                ft_rows.append({'fault_type': ftype, 'setting': f'{pname}={pval}',
                                'detection': np.mean(dets)})
    ftdf = pd.DataFrame(ft_rows).groupby(['fault_type', 'setting'])['detection'].mean().reset_index()
    for _, r in ftdf.iterrows():
        print(f"{r.fault_type:16s} {r.setting:>14s} {r.detection:16.2f}")

    Path('results').mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'channel': ch, 'severity_std': s, 'detection': rate(ch, s),
                   'n_detected': bias_hits[ch][s], 'n_tested': bias_total[ch][s]}
                  for ch in MONITORED for s in BIAS_SEV]).to_csv(
        'results/bias_sensitivity.csv', index=False)
    pd.DataFrame(mdf_rows).to_csv('results/minimum_detectable_bias.csv', index=False)
    ftdf.to_csv('results/fault_type_detection.csv', index=False)
    print("\n[saved] results/bias_sensitivity.csv")


if __name__ == '__main__':
    main()

