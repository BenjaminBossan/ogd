"""According to: McMahan et al. 2013:

Ad click prediction: a view from the trenches

"""

# -*- coding: utf-8 -*-

from __future__ import division
from collections import defaultdict

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = (sum(act * sp.log(pred) +
              sp.subtract(1,act) *
              sp.log(sp.subtract(1,pred)))
          )
    ll = ll * -1.0/len(act)
    return ll


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FTRLprox(BaseEstimator):
    def __init__(self, lambda1, lambda2, alpha, beta):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.beta = beta

    # def predict_proba(self, X, W=None):
    #     if W is None:
    #         W = self.z
    #     result = sigmoid(np.dot(X, W.T))
    #     return result
            
    # def predict(self, X, W=None):
    #     if W is None:
    #         W = self.z
    #     result = self.predict_proba(X, W) > 0.5
    #     return result

    def predict_proba(self, X, W=None):
        if W is None:
            W = []
            for x in X:
                W.append(self.z[xi] for xi in x)
        y_prob = []
        for w in W:
            y_prob.append(sigmoid(sum(w)))
        return y_prob
            
    def predict(self, X, W=None):
        y_prob = self.predict_proba(X, W)
        y_pred = [1 if yp > 0.5 else 0 for yp in y_prob]
        return y_pred

    # def _get_weights(self, zi, etai):
    #     wi = 0.
    #     if np.abs(zi) > self.lambda1:
    #         temp0 = (self.beta + np.sqrt(etai)) / self.alpha
    #         temp1 = -1 / (temp0 + self.lambda2)
    #         wi = temp1 * (zi - np.sign(zi) * self.lambda1)
    #     return wi

    def _get_weights(self, x):
        mask = [(xi in self.z) and (self.z[xi] > self.lambda1) for xi in x]
        eta = [self.eta[xi] for i, xi in enumerate(x) if mask[i]]
        wt = [self.z[xi] for i, xi in enumerate(x) if mask[i]]
        wt = np.sign(wt) * self.lambda1 - wt
        wt /= (self.beta + np.sqrt(eta)) / self.alpha + self.lambda2
        return wt
    
    def fit(self, X, y):
        n, m = X.shape
        if not hasattr(self, 'learn_rate'):
            self.eta = defaultdict(float)
        if not hasattr(self, 'z'):
            self.z = defaultdict(float)
        if not hasattr(self, 'cols'):
            self.cols = X.columns

        trange = np.arange(n)
        for t in trange:
            x = ['_'.join((col, str(val))) for col, val
                 in zip(self.cols, X.iloc[t])]
            wt = self._get_weights(x)
            y_prob = self.predict_proba(wt)
            y_err = y_prob - y[t]
            grad = [y_err] * len(x)
            grad_sq = [y_err * y_err] * len(x)
            eta = [self.eta[xi] for xi in x]
            sigma = (
                (np.sqrt(eta + grad_sq) - np.sqrt(eta))
                / self.alpha
            )
            for xi in x:
                self.z[xi] += grad - sigma
            self.z[index] += grad - sigma * wt
            self.eta[index] += grad_sq

    # def fit(self, X, y):
    #     n, m = X.shape
    #     if not hasattr(self, 'learn_rate'):
    #         self.eta = np.zeros(m)
    #     if not hasattr(self, 'z'):
    #         self.z = np.zeros(m)

    #     trange = np.arange(n)
    #     for t in trange:
    #         index = np.nonzero(X[t])[0]
    #         wt = np.zeros(index.shape)
    #         for j, i in enumerate(index):
    #             wi = self._get_weights(self.z[i],
    #                                    self.eta[i])
    #             wt[j] = wi
    #         y_pred = self.predict(X[t, index], wt)
    #         y_err = y_pred - y[t]
    #         grad = y_err * X[t, index]
    #         grad_sq = grad * grad
    #         sigma = (
    #             (
    #                 np.sqrt(self.eta[index] + grad_sq) -
    #                 np.sqrt(self.eta[index])
    #             )
    #             / self.alpha
    #         )
    #         import pdb; pdb.set_trace()
    #         self.z[index] += grad - sigma * wt
    #         self.eta[index] += grad_sq

    def score(self, X, y, scoring='logloss'):
        y_pred = self.predict_proba(X)
        if scoring == 'logloss':
            return logloss(y, y_pred)
        elif scoring == 'accuracy':
            return np.mean(y == (y_pred > 0.5))
        elif scoring == 'MSE':
            return np.mean((y - y_pred) ** 2)
        