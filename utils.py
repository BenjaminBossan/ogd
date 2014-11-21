# -*- coding: utf-8 -*-

from __future__ import division

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


def logloss(y_true, y_pred):
    """ As provided by kaggle:
    https://www.kaggle.com/wiki/LogarithmicLoss
    """
    epsilon = 1e-18
    y_pred = sp.maximum(epsilon, y_pred)
    y_pred = sp.minimum(1 - epsilon, y_pred)
    ll = (sum(y_true * sp.log(y_pred) +
              sp.subtract(1, y_true) *
              sp.log(sp.subtract(1, y_pred)))
          )
    ll = ll * -1.0 / len(y_true)
    return ll


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GetList(list):
    """Allows lists to be interfaced with get as dicts would be.

    Unfortunatey, this workaround is slower.

    'default' is not used since lists are initalized with defaults
    already.

    """
    def get(self, key, default):
        return self[key]


class PlotLogloss(object):
    def __init__(self, windowsize=1000, figsize=(8, 5), title=None):
        self.windowsize = windowsize
        self.figsize = figsize
        self.title = title
        self._fig_is_init = False
        plt.ion()

    def _init_fig(self):
        fig, ax = plt.subplots(figsize=self.figsize)

        plt.xlabel("time")
        plt.ylabel("logloss")
        plt.grid()

        plt.ylim((0, 1))
        if self.title:
            plt.title(self.title)
        else:
            plt.title('validation logloss')
        return fig, ax

    def plot(self, clf):
        if not self._fig_is_init:
            fig, ax = self._init_fig()

        y, y_prob = zip(*clf.valid_history[:-1])
        t = len(y)
        ll = []
        x_ax = np.arange(0, t, self.windowsize)
        for x in x_ax:
            ll.append(logloss(y[x:x + self.windowsize],
                              y_prob[x:x + self.windowsize]))

        # Plot validation loss curve
        plt.plot(x_ax, ll, 'k.-')

        if not self._fig_is_init:
            plt.legend(loc='best')
            self._fig_is_init = True


class PlotWeightChange(object):
    def __init__(self, figsize=(8, 5), title=None):
        self.figsize = figsize
        self.title = title
        self._fig_is_init = False
        plt.ion()

    def _init_fig(self):
        self.sum_weights = [0.]
        self.x_ax = []
        fig, ax = plt.subplots(figsize=self.figsize)

        plt.xlabel("time")
        plt.ylabel("change in weights")
        plt.grid()

        if self.title:
            plt.title(self.title)
        else:
            plt.title('change in weights over time')
        return fig, ax

    def plot(self, clf):
        if not self._fig_is_init:
            fig, ax = self._init_fig()

        self.x_ax.append(len(clf.valid_history))
        self.sum_weights.append(np.mean(clf.w.values()))

        # Plot change in weights and 0 line
        plt.plot(self.x_ax, np.diff(self.sum_weights), 'k.-',
                 label='weight change')
        plt.plot([self.x_ax[0], self.x_ax[-1]], [0, 0], 'k--', label='zero')

        if not self._fig_is_init:
            plt.legend(loc='best')
            self._fig_is_init = True
