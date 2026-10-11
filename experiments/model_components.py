import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, KernelDensity
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial import ConvexHull, Delaunay
import warnings; warnings.filterwarnings('ignore')

from src.data.load_engine_map import load_engine_map, assign_speed_blocks, CONTEXT
from src.health.shi import SHI, MONITORED
from src.faults import injectors

RNG = 42


def make_models():
    return {
        'ridge': lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
        'poly_ridge': lambda: make_pipeline(StandardScaler(),
                                            PolynomialFeatures(2, include_bias=False), Ridge(1.0)),
        'pls': lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=3)),
        'extratrees': lambda: ExtraTreesRegressor(n_estimators=200, random_state=0),
        'rf': lambda: RandomForestRegressor(n_estimators=200, random_state=0),
        'knn': lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    }


class Gates:
    def __init__(self, train_xy):
        self.X = train_xy
        self.sc = StandardScaler().fit(train_xy)
        Z = self.sc.transform(train_xy)
        self.nn = NearestNeighbors(n_neighbors=min(4, len(Z))).fit(Z)
        d, _ = self.nn.kneighbors(Z)
        self.knn_thr = float(np.quantile(d[:, 1:].mean(1), 0.95))
        self.cov = EmpiricalCovariance().fit(Z)
        self.maha_thr = float(np.quantile(self.cov.mahalanobis(Z), 0.95))
        self.iforest = IsolationForest(random_state=0, contamination=0.05).fit(Z)
        self.kde = KernelDensity(bandwidth=0.5).fit(Z)
        self.kde_thr = float(np.quantile(self.kde.score_samples(Z), 0.05))
        try:
            self.hull = Delaunay(train_xy)
        except Exception:
            self.hull = None

    def accept(self, xy):
        Z = self.sc.transform(xy)
        d, _ = self.nn.kneighbors(Z)
        knn = d[:, 1:].mean(1) <= self.knn_thr
        maha = self.cov.mahalanobis(Z) <= self.maha_thr
        ifo = self.iforest.predict(Z) == 1
        kde = self.kde.score_samples(Z) >= self.kde_thr
        hull = (self.hull.find_simplex(xy) >= 0) if self.hull is not None else np.ones(len(xy), bool)
        return {'knn': knn, 'mahalanobis': maha, 'iforest': ifo, 'kde': kde,
                'hull_and_density': hull & knn}


def main():
    df, _ = load_engine_map()
    df = assign_speed_blocks(df)
    df['rpm_level'] = df['Engine Speed'].round(0)
    within = list(KFold(5, shuffle=True, random_state=RNG).split(df))
    grouped = list(GroupKFold(7).split(df, groups=df['rpm_level']))

    print("Model and operating-condition comparisons")


    print("\nHealthy prediction and sensor identification")
    rec_rows = []
    for name, ctor in make_models().items():
        r2s, iso_ok = [], []
        for tr_idx, te_idx in within:
            train, test = df.iloc[tr_idx], df.iloc[te_idx]
            stds = train[MONITORED].std()
            models = {}
            for ch in MONITORED:
                others = [c for c in MONITORED if c != ch]
                models[ch] = ctor().fit(train[CONTEXT + others].values, train[ch].values)

            for ch in MONITORED:
                others = [c for c in MONITORED if c != ch]
                p = np.asarray(models[ch].predict(test[CONTEXT + others].values)).ravel()
                ss = 1 - np.sum((test[ch].values - p) ** 2) / \
                     (np.sum((test[ch].values - test[ch].values.mean()) ** 2) + 1e-9)
                r2s.append(ss)

            for true_c in MONITORED:
                f, _ = injectors.apply(test, true_c, 'bias', stds[true_c],
                                       np.random.RandomState(0), severity=2.0, direction=1)
                errs = []
                for cand in MONITORED:
                    others = [c for c in MONITORED if c != cand]
                    rep = f.copy()
                    rep[cand] = np.asarray(models[cand].predict(f[CONTEXT + others].values)).ravel()
                    resid = np.column_stack([
                        rep[c].values - np.asarray(
                            models[c].predict(rep[CONTEXT + [o for o in MONITORED if o != c]].values)
                        ).ravel() for c in MONITORED])
                    errs.append(np.abs(resid).sum(1))
                E = np.column_stack(errs)
                iso_ok.extend((np.array(MONITORED)[E.argmin(1)] == true_c).tolist())
        rec_rows.append(dict(model=name, healthy_R2=float(np.mean(r2s)),
                             isolation_acc=float(np.mean(iso_ok))))
        print(f"  {name:12s} healthy R2 {np.mean(r2s):6.3f} | isolation acc {np.mean(iso_ok):6.1%}")
    RM = pd.DataFrame(rec_rows)
    best_r2 = RM.loc[RM.healthy_R2.idxmax(), 'model']
    best_iso = RM.loc[RM.isolation_acc.idxmax(), 'model']
    print(f"  best healthy R2 model {best_r2} | best sensor identification model {best_iso}")


    print("\nHealthy false-alarm rates by error scaling")
    norm_rows = []
    for regime, splits in [('within_map', within), ('grouped_RPM_holdout', grouped)]:
        acc = {k: [] for k in ['global', 'region', 'ensemble', 'knn_local']}
        for tr_idx, te_idx in splits:
            train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            shi = SHI(system_fpr=0.05, seed=RNG).fit(train)
            Ztest = shi.channel_z(test)
            acc['region'].append(float((Ztest.max(1) > shi.sys_thresh).mean()))
            R = shi._residuals(shi.parity, test)
            gs = shi.global_scale + 1e-9
            zg = np.abs(R) / gs
            thr_g = float(np.quantile(
                (np.abs(shi._residuals(shi.parity, train)) / gs).max(1), 0.95))
            acc['global'].append(float((zg.max(1) > thr_g).mean()))


            Rtr = shi._residuals(shi.parity, train)
            ens_te, ens_tr = {}, {}
            for ch in MONITORED:
                others = [c for c in MONITORED if c != ch]
                m = ExtraTreesRegressor(n_estimators=120, random_state=0).fit(
                    train[CONTEXT + others].values, train[ch].values)
                ens_te[ch] = np.stack([t.predict(test[CONTEXT + others].values)
                                       for t in m.estimators_]).std(0)
                ens_tr[ch] = np.stack([t.predict(train[CONTEXT + others].values)
                                       for t in m.estimators_]).std(0)
            sm_te = np.column_stack([ens_te[c] for c in MONITORED])
            sm_tr = np.column_stack([ens_tr[c] for c in MONITORED])
            thr_e = float(np.quantile((np.abs(Rtr) / np.sqrt(sm_tr ** 2 + gs ** 2)).max(1), 0.95))
            ze = np.abs(R) / np.sqrt(sm_te ** 2 + gs ** 2)
            acc['ensemble'].append(float((ze.max(1) > thr_e).mean()))


            sc_tr = StandardScaler().fit(train[CONTEXT].values)
            Ztr_ctx = sc_tr.transform(train[CONTEXT].values)
            nn = NearestNeighbors(n_neighbors=min(6, len(train))).fit(Ztr_ctx)
            floor = 0.25 * gs
            _, idx_tr = nn.kneighbors(Ztr_ctx)
            loc_tr = np.maximum(np.stack([np.abs(Rtr[ii[1:]]).mean(0) for ii in idx_tr]), floor)
            thr_l = float(np.quantile((np.abs(Rtr) / loc_tr).max(1), 0.95))
            _, idx = nn.kneighbors(sc_tr.transform(test[CONTEXT].values))
            loc = np.maximum(np.stack([np.abs(Rtr[ii]).mean(0) for ii in idx]), floor)
            acc['knn_local'].append(float(((np.abs(R) / loc).max(1) > thr_l).mean()))
        for k, v in acc.items():
            norm_rows.append(dict(regime=regime, normalization=k, healthy_FA=float(np.mean(v))))
        print(f"  {regime:22s} " + "  ".join(f"{k}={np.mean(v):.1%}" for k, v in acc.items()))


    print("\nOperating-condition checks")
    gate_rows = []
    for tr_idx, te_idx in grouped:
        train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        shi = SHI(system_fpr=0.05, seed=RNG).fit(train)
        det = shi.detect(test)
        g = Gates(train[CONTEXT].values)
        acc = g.accept(test[CONTEXT].values)
        for name, m in acc.items():
            gate_rows.append(dict(gate=name, coverage=float(m.mean()),
                                  fa_accepted=float(det[m].mean()) if m.any() else np.nan,
                                  fa_all=float(det.mean())))
    G = pd.DataFrame(gate_rows).groupby('gate').mean().reset_index()
    print(f"{'gate':20s} {'coverage':>10s} {'FA among accepted':>19s} {'FA all':>9s}")
    for _, r in G.iterrows():
        print(f"{r['gate']:20s} {r['coverage']:10.1%} {r['fa_accepted']:19.1%} {r['fa_all']:9.1%}")

    Path('results').mkdir(parents=True, exist_ok=True)
    RM.to_csv('results/model_comparison.csv', index=False)
    pd.DataFrame(norm_rows).to_csv('results/residual_adjustment.csv', index=False)
    G.to_csv('results/support_comparison.csv', index=False)
    print("\n[saved] results/model_comparison.csv")


if __name__ == '__main__':
    main()

