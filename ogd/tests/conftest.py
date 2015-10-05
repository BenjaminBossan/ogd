# -*- coding: utf-8 -*-

from __future__ import division
from copy import deepcopy
import string

import numpy as np
import pytest

from ogd.models import FTRLprox
from ogd.models import OGDLR
from ogd.neuralnet import Neuralnet


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")


# generate data and various models
N = 10000
COLS = ['weather', 'temperature', 'noise']
LAMBDA1 = 0
LAMBDA2 = 0
ALPHA = 0.1
NDIMS = 2 ** 20


@pytest.fixture(scope='session')
def training_data():
    """Create random 'golf' data.

    Probability of golfing is higher if weather is sunny and
    temperature is warm. Add a random noise feature.

    """
    n = N
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


@pytest.fixture(scope='session')
def ogdlr_before(training_data):
    X, y = training_data
    clf = OGDLR(lambda1=LAMBDA1, alpha=ALPHA, alr_schedule='constant')
    clf.fit(X[:10], y[:10], COLS)
    return clf


@pytest.fixture(scope='session')
def ogdlr_after(ogdlr_before, training_data):
    X, y = training_data
    clf = deepcopy(ogdlr_before)
    clf.fit(X, y, COLS)
    return clf


@pytest.fixture(scope='session')
def ftrl_before(training_data):
    X, y = training_data
    clf = FTRLprox(lambda1=LAMBDA1, lambda2=LAMBDA2, alpha=ALPHA, beta=1,
                   alr_schedule='constant')
    clf.fit(X[:10], y[:10], COLS)
    return clf


@pytest.fixture(scope='session')
def ftrl_after(ftrl_before, training_data):
    X, y = training_data
    clf = deepcopy(ftrl_before)
    clf.fit(X, y, COLS)
    return clf


@pytest.fixture(scope='session')
def hash_before(training_data):
    X, y = training_data
    clf = OGDLR(lambda1=LAMBDA1, alpha=ALPHA, alr_schedule='constant',
                ndims=NDIMS)
    clf.fit(X[:10], y[:10], COLS)
    return clf


@pytest.fixture(scope='session')
def hash_after(hash_before, training_data):
    X, y = training_data
    clf = deepcopy(hash_before)
    clf.fit(X, y, COLS)
    return clf


@pytest.fixture(scope='session')
def manycols_before(training_data):
    X, y = training_data
    X = np.hstack((X for __ in range(30)))
    clf = FTRLprox(lambda1=LAMBDA1, lambda2=LAMBDA2, alpha=ALPHA, beta=1,
                   alr_schedule='constant')
    clf.fit(X[:10], y[:10], COLS)
    return clf


@pytest.fixture(scope='session')
def manycols_after(manycols_before, training_data):
    X, y = training_data
    X = np.hstack((X for __ in range(30)))
    clf = deepcopy(manycols_before)
    clf.fit(X, y, COLS)
    return clf


@pytest.fixture(scope='session')
def nn_before(training_data):
    X, y = training_data
    clf = Neuralnet(lambda1=LAMBDA1, lambda2=LAMBDA2, alpha=ALPHA, beta=1,
                    alr_schedule='constant', num_units=16)
    clf.fit(X[:10], y[:10], COLS)
    return clf


@pytest.fixture(scope='session')
def nn_after(nn_before, training_data):
    X, y = training_data
    clf = deepcopy(nn_before)
    clf.fit(X, y, COLS)
    return clf
