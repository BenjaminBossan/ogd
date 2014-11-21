# -*- coding: utf-8 -*-

from __future__ import division

import string

import numpy as np

from FTRLprox.models import FTRLprox
from FTRLprox.models import OGDLR
from FTRLprox.models import SEED
from FTRLprox.utils import logloss


def create_training_data(n=10000):
    """Create random 'golf' data.

    Probability of golfing is higher if weather is sunny and
    temperature is warm. Add a random noise feature.

    """
    weather = np.random.choice(['rainy', 'sunny'], n, p=[0.3, 0.7])
    temperature = np.random.choice(['warm', 'cold'], n, p=[0.4, 0.6])
    letters = [c for c in string.ascii_lowercase]
    noise_features = [''.join(np.random.choice(letters, 3))
                      for __ in range(100)]
    noise = np.random.choice(noise_features, n)
    X = np.array([weather, temperature, noise]).T
    y = np.zeros(n)
    for i in range(n):
        pr1 = 0.5 * np.random.rand() if X[i, 0] == 'sunny' else 0
        pr2 = 0.5 * np.random.rand() if X[i, 1] == 'warm' else 0
        pr3 = np.random.rand()
        y[i] = 1 if pr1 + pr2 + pr3 > 0.95 else 0
    return X, y


# generate data and various models
N = 10000
X, y = create_training_data(n=N)
COLS = ['weather', 'temperature', 'noise']
LAMBDA1 = 0
LAMBDA2 = 0
ALPHA = 0.1
NDIMS = 2 ** 20

ogdlr_before = OGDLR(lambda1=LAMBDA1, alpha=ALPHA, alr_schedule='constant')
ogdlr_before.fit(X[:10], y[:10], COLS)

ogdlr_after = OGDLR(lambda1=LAMBDA1, alpha=ALPHA, alr_schedule='constant')
ogdlr_after.fit(X, y, COLS)

ftrl_before = FTRLprox(lambda1=LAMBDA1, lambda2=LAMBDA2, alpha=ALPHA, beta=1,
                       alr_schedule='constant')
ftrl_before.fit(X[:10], y[:10], COLS)

ftrl_after = FTRLprox(lambda1=LAMBDA1, lambda2=LAMBDA2, alpha=ALPHA, beta=1,
                      alr_schedule='constant')
ftrl_after.fit(X, y, COLS)

hash_before = OGDLR(lambda1=LAMBDA1, alpha=ALPHA,
                    alr_schedule='constant', ndims=NDIMS)
hash_before.fit(X[:10], y[:10], COLS)

hash_after = OGDLR(lambda1=LAMBDA1, alpha=ALPHA,
                   alr_schedule='constant', ndims=NDIMS)
hash_after.fit(X, y, COLS)