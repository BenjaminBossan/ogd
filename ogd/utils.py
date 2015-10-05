# -*- coding: utf-8 -*-

from __future__ import division
from itertools import combinations

import scipy as sp
import numpy as np


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


def add_interactions(lst, depth=1):
    if depth <= 1:
        return
    n = len(lst)
    interactions = []
    for d in range(2, depth + 1):
        for combi in combinations(range(n), r=d):
            interactions.append(' '.join(lst[c] for c in combi))
    lst.extend(interactions)


class GetList(list):
    """Allows lists to be interfaced with `get` as dicts would be.

    Unfortunatey, this workaround is slower.

    'default' is not used since lists are initalized with defaults
    already.

    """
    def get(self, key, default):
        return self[key]
