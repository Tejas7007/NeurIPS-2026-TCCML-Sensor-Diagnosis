import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class EnvelopeGate:
    def __init__(self, k=3, quantile=0.95):


        self.k = k
        self.quantile = quantile
        self.scaler = None
        self.nn = None
        self.threshold = None

    def fit(self, context_train):

        self.scaler = StandardScaler().fit(context_train)
        Z = self.scaler.transform(context_train)
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Z)

        d, _ = self.nn.kneighbors(Z)
        healthy_knn = d[:, 1:].mean(axis=1)
        self.threshold = np.quantile(healthy_knn, self.quantile)
        return self

    def score(self, context):

        Z = self.scaler.transform(np.atleast_2d(context))
        d, _ = self.nn.kneighbors(Z)

        return d[:, :self.k].mean(axis=1)

    def inside(self, context):

        return self.score(context) <= self.threshold
