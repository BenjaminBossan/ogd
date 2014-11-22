# -*- coding: utf-8 -*-

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pytest

from FTRLprox.models import FTRLprox
from FTRLprox.models import OGDLR
from FTRLprox.models import SEED
from FTRLprox.utils import logloss
from tutils import ogdlr_before
from tutils import ogdlr_after
from tutils import ftrl_before
from tutils import ftrl_after
from tutils import hash_before
from tutils import hash_after
from tutils import X
from tutils import y


def test_FTRL_learns():
    # test that this model learns something
    y_pred = ftrl_before.predict_proba(X)
    ll_before = logloss(y, y_pred)

    y_pred = ftrl_after.predict_proba(X)
    ll_after = logloss(y, y_pred)

    assert ll_before > ll_after


def test_ogdlr_learns():
    # test that this model learns something
    y_pred = ogdlr_before.predict_proba(X)
    ll_before = logloss(y, y_pred)

    y_pred = ogdlr_after.predict_proba(X)
    ll_after = logloss(y, y_pred)

    assert ll_before > ll_after


def test_ogdlr_with_hash_learns():
    # test that this model learns something
    y_pred = hash_before.predict_proba(X)
    ll_before = logloss(y, y_pred)

    y_pred = hash_after.predict_proba(X)
    ll_after = logloss(y, y_pred)

    assert ll_before > ll_after


def test_models_same_predictions():
    # for lambda1, lambda2 = 0, OGDLR and FTRLprox should generate the
    # same result. The same goes if hashing is used (except for the
    # rare case of hash collisions.
    y_f = ftrl_after.predict(X)
    y_o = ogdlr_after.predict(X)
    y_h = hash_after.predict(X)
    assert sum(y_f != y_o) == 0
    assert sum(y_f != y_h) == 0


def test_effect_of_alpha():
    # higher alpha should lead to faster learning, thus higher weights
    mean_abs_weights = []
    for alpha in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]:
        clf = OGDLR(alpha=alpha)
        clf.fit(X[:100], y[:100])
        mean_abs_weights.append(np.abs(clf.weights()).mean())
    assert mean_abs_weights[0] == 0.
    assert all(np.diff(mean_abs_weights) > 0)


def test_effect_of_beta():
    # higher beta should lead to slower learning, thus lower weights
    mean_abs_weights = []
    for beta in [10 ** n for n in range(7)]:
        clf = OGDLR(beta=beta)
        clf.fit(X[:100], y[:100])
        mean_abs_weights.append(np.abs(clf.weights()).mean())
    assert np.allclose(mean_abs_weights[-1], 0, atol=1e-6)
    assert all(np.diff(mean_abs_weights) < 0)


@pytest.mark.parametrize('lambda1', [1 / lam for lam in range(1, 6)])
def test_effect_of_lambda1(lambda1):
    # gradients should be the same regardless of magnitude of weights
    weights = range(-5, 0) + range(1, 6)
    clf = OGDLR(lambda1=lambda1)
    grads = clf._get_grads(0, 0, weights)
    # gradient only depends on sign of weight
    abso = [gr * np.sign(w) for gr, w in zip(grads, weights)]
    assert np.allclose(abso[0], abso)

    # contingency test: should fail
    frac = [gr / w for gr, w in zip(grads, weights)]
    with pytest.raises(AssertionError):
        assert np.allclose(frac[0], frac)


@pytest.mark.parametrize('lambda2', [1 / lam for lam in range(1, 6)])
def test_effect_of_lambda2(lambda2):
    # relative difference in gradients should be the same
    weights = range(-5, 0) + range(1, 6)
    clf = OGDLR(lambda2=lambda2)
    grads = clf._get_grads(0, 0, weights)
    # gradient only depends on sign of weight
    frac = [gr / w for gr, w in zip(grads, weights)]
    assert np.allclose(frac[0], frac)

    # contingencey test: should fail
    abso = [gr * np.sign(w) for gr, w in zip(grads, weights)]
    with pytest.raises(AssertionError):
        assert np.allclose(abso[0], abso)
