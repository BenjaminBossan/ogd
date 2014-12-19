# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import pytest

from FTRLprox.neuralnet import Neuralnet
from tutils import X
from tutils import y
from tutils import nn_before
from tutils import nn_after


def test_init_args():
    clf = Neuralnet(num_units=12)
    clf.lambda1 == 0.
    clf.alr_schedule == 'gradient'
    clf.num_units == 12

    clf = Neuralnet(3, num_units=13)
    clf.lambda1 == 3
    clf.alr_schedule == 'gradient'
    clf.num_units == 13

    clf = Neuralnet(num_units=1, alr_schedule='count')
    clf.lambda1 == 0.
    clf.alr_schedule == 'count'
    clf.num_units == 1

    clf = Neuralnet(alr_schedule='count', num_units=12)
    clf.lambda1 == 0.
    clf.alr_schedule == 'count'
    clf.num_units == 12

    clf = Neuralnet(0.5, alr_schedule='count', num_units=12)
    clf.lambda1 == 0.5
    clf.alr_schedule == 'count'
    clf.num_units == 12


def test_init_exceptions():
    with pytest.raises(TypeError):
        Neuralnet()
    with pytest.raises(TypeError):
        Neuralnet(0, 0.3, 0.1, 1., alr_schedule='constant')
    with pytest.raises(NotImplementedError):
        Neuralnet(num_units=10, ndims=2 ** 20)


@pytest.mark.parametrize('num_units', [1, 16, 64])
def test_effect_of_alpha(num_units):
    # Higher alpha should lead to faster learning, thus higher
    # weights. However, this is not as true for neural nets as for
    # logistic regressions, since the weights of neural nets are not
    # initialized at 0. Therefore, for small alpha, it may happen that
    # the weights would actually be higher.
    # loop through layers:
    for l in [0, 1]:
        mean_abs_weights = []
        for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
            clf = Neuralnet(num_units=num_units, alpha=alpha)
            clf.fit(X[:5000], y[:5000])
            mean_abs_weights.append(np.abs(clf.weights()[l]).mean())
        assert all(np.diff(mean_abs_weights) > 0)


@pytest.mark.parametrize('num_units', [1, 16, 64])
def test_effect_of_beta(num_units):
    # Lower beta should lead to faster learning, thus higher
    # weights. However, this is not as true for neural nets as for
    # logistic regressions, since the weights of neural nets are not
    # initialized at 0. Therefore, for high alpha, it may happen that
    # the weights would actually be higher.
    # loop through layers:
    for l in [0, 1]:
        mean_abs_weights = []
        for beta in [10 ** n for n in range(5)]:
            clf = Neuralnet(num_units=num_units, alpha=1, beta=beta)
            clf.fit(X[:5000], y[:5000])
            mean_abs_weights.append(np.abs(clf.weights()[l]).mean())
        assert all(np.diff(mean_abs_weights) < 0)


@pytest.mark.parametrize('lambda1', [1 / lam for lam in range(1, 6)])
def test_effect_of_lambda1(lambda1):
    # gradients should be the same regardless of magnitude of weights
    clf = Neuralnet(lambda1=lambda1, num_units=8)
    clf.fit(X[:10], y[:10])
    activities = clf._get_p(X[0])
    activities[-1] = 0  # set prediction to outcome, so that y_err = 0
    weights = clf._get_w(X[0])
    grads = clf._get_grads(0, activities, weights)
    # gradient only depends on sign of weight
    for l in [0, 1]:
        # loop through layers
        abso = [gr * np.sign(w) for gr, w
                in zip(grads[l].flatten(), weights[l].flatten())]
        assert np.allclose(abso[0], abso)

        # contingency test: should fail
        frac = [gr / w for gr, w in
                zip(grads[l].flatten(), weights[l].flatten())]
        with pytest.raises(AssertionError):
            assert np.allclose(frac[0], frac)


@pytest.mark.parametrize('lambda2', [1 / lam for lam in range(1, 6)])
def test_effect_of_lambda2(lambda2):
    # gradients should be the same regardless of magnitude of weights
    clf = Neuralnet(lambda2=lambda2, num_units=8)
    clf.fit(X[:10], y[:10])
    activities = clf._get_p(X[0])
    activities[-1] = 0  # set prediction to outcome, so that y_err = 0
    weights = clf._get_w(X[0])
    grads = clf._get_grads(0, activities, weights)
    # gradient only depends on sign of weight
    for l in [0, 1]:
        # loop through layers
        frac = [gr / w for gr, w in
                zip(grads[l].flatten(), weights[l].flatten())]
        assert np.allclose(frac[0], frac)

        abso = [gr * np.sign(w) for gr, w
                in zip(grads[l].flatten(), weights[l].flatten())]
        # contingency test: should fail
        with pytest.raises(AssertionError):
            assert np.allclose(abso[0], abso)


# def test_nn_gradients():
    # # The gradients are not right yet, they are off by a constant
    # # factor. This should not be a problem in itself -- just increaes
    # # the learning rate alpha -- but could suggest some deeper
    # # underlying bug.
    # clf = Neuralnet(num_units=8, lambda1=0.001)
    # for __ in range(10): clf.fit(X, y)
    # grads, grads_num = clf.numerical_grad(X[2], y[2], 1e-3)
    # assert [np.allclose(grad, grad_num) for grad, grad_num
            # in grads, grads_num]
