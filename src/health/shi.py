import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from src.evaluation.crossfit import oof_map

CONTEXT = ['Engine Speed', 'Torque']
MONITORED = ['T_IM', 'P_IM', 'EGR_Rate', 'MF_IA', 'NOx_EO']


def _poly_ridge():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                         Ridge(alpha=1.0))


class SHI:


    def __init__(self, n_folds=5, seed=0, system_fpr=0.05, normalization='region', k_local=6):


        self.normalization = normalization
        self.k_local = k_local
        self._ctx_train = None
        self._absR_train = None
        self._ctx_scaler = None
        self._ens = {}
        self._init_rest(n_folds, seed, system_fpr)

    def _init_rest(self, n_folds=5, seed=0, system_fpr=0.05):
        self.n_folds = n_folds
        self.seed = seed
        self.system_fpr = system_fpr
        self.parity = {}
        self.region_scale = {}
        self.global_scale = None
        self.healthy_z = {}
        self.A_cal = None
        self.sys_thresh = None


    def _fit_parity(self, frame):
        models = {}
        for ch in MONITORED:
            in_cols = CONTEXT + [c for c in MONITORED if c != ch]
            models[ch] = (_poly_ridge().fit(frame[in_cols].values, frame[ch].values), in_cols)
        return models

    def _residuals(self, models, frame):
        R = np.zeros((len(frame), len(MONITORED)))
        for j, ch in enumerate(MONITORED):
            m, in_cols = models[ch]
            R[:, j] = frame[ch].values - m.predict(frame[in_cols].values)
        return R

    def fit(self, train, region_col='region_block'):

        R_oof = oof_map(train, self._fit_parity, self._residuals,
                        n_folds=self.n_folds, seed=self.seed)
        regions = train[region_col].values


        self.global_scale = np.nanstd(R_oof, axis=0)
        self.region_scale = {}
        for reg in np.unique(regions):
            m = regions == reg
            self.region_scale[reg] = (np.nanstd(R_oof[m], axis=0)
                                      if m.sum() >= 4 else self.global_scale)


        from sklearn.preprocessing import StandardScaler as _SS
        self._ctx_scaler = _SS().fit(train[CONTEXT].values)
        self._ctx_train = self._ctx_scaler.transform(train[CONTEXT].values)
        self._absR_train = np.abs(R_oof)

        if self.normalization == 'ensemble':
            from sklearn.ensemble import ExtraTreesRegressor
            for ch in MONITORED:
                others = [c for c in MONITORED if c != ch]
                self._ens[ch] = ExtraTreesRegressor(
                    n_estimators=120, random_state=self.seed).fit(
                        train[CONTEXT + others].values, train[ch].values)

        sc = self._scale_for(train, region_col)
        Z = np.abs(R_oof) / (sc + 1e-9)


        self.healthy_z = {j: Z[~np.isnan(Z[:, j]), j] for j in range(len(MONITORED))}


        A = np.nanmax(Z, axis=1)
        self.A_cal = A[~np.isnan(A)]
        self.sys_thresh = np.quantile(self.A_cal, 1 - self.system_fpr)


        self.parity = self._fit_parity(train)
        return self

    def _scale_for(self, frame, region_col='region_block'):
        if self.normalization == 'ensemble' and self._ens:
            return self._model_uncertainty_adjusted_scale(frame)
        if self.normalization == 'local' and self._ctx_train is not None:
            from sklearn.neighbors import NearestNeighbors
            k = int(min(self.k_local, len(self._ctx_train)))
            nn = NearestNeighbors(n_neighbors=k).fit(self._ctx_train)
            _, idx = nn.kneighbors(self._ctx_scaler.transform(frame[CONTEXT].values))
            loc = np.stack([np.nanmean(self._absR_train[ii], axis=0) for ii in idx])


            return np.maximum(loc, 0.25 * self.global_scale)
        return np.array([self.region_scale.get(r, self.global_scale)
                         for r in frame[region_col].values])

    def _model_uncertainty_adjusted_scale(self, frame):
        cols = []
        for ch in MONITORED:
            others = [c for c in MONITORED if c != ch]
            per = np.stack([t.predict(frame[CONTEXT + others].values)
                            for t in self._ens[ch].estimators_])
            cols.append(per.std(0))
        sigma_model = np.column_stack(cols)
        return np.sqrt(sigma_model ** 2 + self.global_scale ** 2)

    def channel_z(self, frame, region_col='region_block'):
        R = self._residuals(self.parity, frame)
        sc = self._scale_for(frame, region_col)
        return np.abs(R) / (sc + 1e-9)

    def channel_shi(self, frame, region_col='region_block'):

        Z = self.channel_z(frame, region_col)
        out = np.zeros_like(Z)
        for j in range(len(MONITORED)):
            hz = self.healthy_z[j]

            out[:, j] = (1 + (hz[None, :] >= Z[:, j][:, None]).sum(1)) / (len(hz) + 1)
        return out

    def system_shi(self, frame, region_col='region_block'):


        Z = self.channel_z(frame, region_col)
        A = Z.max(axis=1)
        p = (1 + (self.A_cal[None, :] >= A[:, None]).sum(1)) / (len(self.A_cal) + 1)
        return np.clip(p / self.system_fpr, 0, 1)

    def channel_severity(self, frame, region_col='region_block'):


        return self.channel_z(frame, region_col)

    def system_severity(self, frame, region_col='region_block'):

        return self.channel_z(frame, region_col).max(axis=1)

    def channel_pvalue(self, frame, region_col='region_block'):

        return self.channel_shi(frame, region_col)

    def detect(self, frame, region_col='region_block'):

        Z = self.channel_z(frame, region_col)
        return Z.max(axis=1) > self.sys_thresh
