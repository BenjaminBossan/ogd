# -*- coding: utf-8 -*-

from __future__ import division

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
    # higher alpha should lead to slower learning
    mean_abs_weights = []
    for alpha in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]:
        clf = OGDLR(alpha=alpha)
        clf.fit(X[:100], y[:100])
        mean_abs_weights.append(np.abs(clf.weights()).mean())
    assert all(np.diff(mean_abs_weights) > 0)


@pytest.mark.parametrize('lambda1_0, lambda1_1', [
    (0, 1e-3),
    (1e-3, 1e-2),
    (1e-2, 1e-1),
    (1e-1, 1e0),
    (1e0, 1e1),
    (1e1, 1e2),
    (1e2, 1e3),
])
def test_effect_lambda1(lambda1_0, lambda1_1):
    # higher lambda1 should lead to absolutely lower weights
    clf0 = OGDLR(lambda1=lambda1_0)
    clf0.fit(X[:100], y[:100])
    w0 = clf0.weights()
    clf1 = OGDLR(lambda1=lambda1_1)
    clf1.fit(X[:100], y[:100])
    w1 = clf1.weights()
    diff_weights = []
    for key in clf0.keys():
        diff_weights.append(clf0.w[key] - clf1.w[key])
    # import pdb; pdb.set_trace()
        