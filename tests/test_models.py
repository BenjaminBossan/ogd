# -*- coding: utf-8 -*-

from __future__ import division

from itertools import permutations
from mock import MagicMock
from mock import patch
import string

import numpy as np
import pytest
from sklearn.utils import murmurhash3_32 as mmh

from FTRLprox.models import FTRLprox
from FTRLprox.models import OGDLR
from FTRLprox.models import SEED
from FTRLprox.utils import logloss
from FTRLprox.utils import GetList


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


def test_keys_dictionary():
    ogdlr_keys = ogdlr_after.keys()
    ftrl_keys = ftrl_after.keys()
    assert ogdlr_keys == ftrl_keys
    known_keys = set(['BIAS', 'weather__rainy', 'weather__sunny',
                      'temperature__cold', 'temperature__warm'])
    assert known_keys <= set(ftrl_keys)


def test_keys_list():
    assert hash_after.keys() is None


def test_weights():
    ogdlr_weights = ogdlr_after.weights()
    ftrl_weights = ftrl_after.weights()
    hash_keys = [mmh(key, seed=SEED) % NDIMS for key in ftrl_after.keys()]
    hash_weights = hash_after._get_w(hash_keys)

    assert np.allclose(ogdlr_weights, ftrl_weights)
    assert np.allclose(hash_weights, ftrl_weights)


def test_weights_list():
    assert hash_after.weights() is None


@pytest.mark.parametrize('alr', [
    'gradient',
    'count',
    'constant',
])
def test_ogdlr_grad_numerically(alr):
    epsilon = 1e-6
    clf = OGDLR(alr_schedule=alr)
    clf.fit(X[:100], y[:100])
    for xx, yy in zip(X[:10], y[:10]):
        grad, grad_num = ogdlr_after.numerical_grad(xx, yy, epsilon)
        assert np.allclose(grad, grad_num, atol=epsilon)


@pytest.mark.parametrize('lambda1', [0, 0.1, 1, 10, 100])
def test_ogdlr_grad_numerically(lambda1):
    epsilon = 1e-6
    clf = OGDLR(lambda1=lambda1)
    clf.fit(X[:100], y[:100])
    for xx, yy in zip(X[:10], y[:10]):
        grad, grad_num = ogdlr_after.numerical_grad(xx, yy, epsilon)
        assert np.allclose(grad, grad_num, atol=epsilon)


# @pytest.mark.parametrize(
#     'args',
#     list(set(permutations(3 * [0] + 3 * [0.5] + 3 * [2], 3)))
# )
# def test_ftrl_grad_numerically_alpha002(args):
#     # test all combinations of lambda1, lambda2, beta values of 0,
#     # 0.5, and 2 and for alpha = 0.02. alpha is not part of the list
#     # since it is not allowed to be 0.
#     epsilon = 1e-6
#     clf = FTRLprox(args[0], args[1], 0.02, args[2])
#     clf.fit(X[:100], y[:100])
#     close = []
#     for xx, yy in zip(X[:10], y[:10]):
#         grad, grad_num = clf.numerical_grad(xx, yy, epsilon)
#         close.append(np.allclose(grad, grad_num, atol=epsilon))
#     assert all(close)


# @pytest.mark.parametrize(
#     'args',
#     list(set(permutations(3 * [0] + 3 * [0.5] + 3 * [2], 3)))
# )
# def test_ftrl_grad_numerically_alpha1(args):
#     # test all combinations of lambda1, lambda2, beta values of 0,
#     # 0.5, and 2 and for alpha = 1.
#     epsilon = 1e-6
#     clf = FTRLprox(args[0], args[1], 1, args[2])
#     clf.fit(X[:100], y[:100])
#     close = []
#     for xx, yy in zip(X[:10], y[:10]):
#         grad, grad_num = clf.numerical_grad(xx, yy, epsilon)
#         close.append(np.allclose(grad, grad_num, atol=epsilon))
#     assert all(close)


@pytest.mark.parametrize('arg, expected', [
    (None, dict),
    (123, list),
    (123, GetList),
])
def test_ogdlr_init_ndims(arg, expected):
    clf = OGDLR(ndims=arg)
    clf.fit(X[:10], y[:10])
    assert isinstance(clf.w, expected)
    assert isinstance(clf.num, expected)


def test_ogdlr_init_cols():
    # case where cols are given
    assert ogdlr_after.cols == ['weather', 'temperature', 'noise']

    clf = OGDLR()
    clf.fit(X[:10], y[:10])
    # creates default cols 0 ... 2
    assert clf.cols == ['col0', 'col1', 'col2']

    clf.fit(X[:10], y[:10], cols=['a', 'b', 'c'])
    # cols do not change
    assert clf.cols == ['col0', 'col1', 'col2']

    clf = OGDLR()
    clf.fit(np.random.random((10, 25)), y[:10])
    # creates cols 01 ... 24
    assert clf.cols[0] == 'col00'
    assert clf.cols[13] == 'col13'
    assert clf.cols[-1] == 'col24'

    # column names must be unique
    with pytest.raises(ValueError):
        clf = OGDLR()
        clf.fit(X[:10], y[:10], cols=['1', '2', '1'])


def test_get_x():
    # if using list, get_x should get ints
    clf = OGDLR(ndims=100)
    clf.fit(X[:100], y[:100])
    xt = ['sunny', 'cold', X[0, 2]]
    result = clf._get_x(xt)
    assert all([isinstance(r, int) for r in result])

    # if using dict, get_x should give dictionary keys
    xt = ['sunny', 'cold', X[0, 2]]
    result = ogdlr_after._get_x(xt)
    expected = ['BIAS',
                'weather__sunny',
                'temperature__cold',
                'noise__' + X[0, 2]]
    assert result == expected


@pytest.mark.parametrize('clf, key, expected', [
    (ogdlr_after, ogdlr_after.w.keys()[0],
     ogdlr_after.w.values()[0]),
    (ogdlr_after, ogdlr_after.w.keys()[-1],
     ogdlr_after.w.values()[-1]),
    (ogdlr_after, 'key-not-present', 0.)
])
def test_get_w(clf, key, expected):
    result = clf._get_w([key])[0]
    assert result == expected


def test_get_p():
    # probabilities are between 0 and 1
    Xs = [ogdlr_after._get_x(x) for x in X]
    prob = [ogdlr_after._get_p(xt) for xt in Xs]
    assert all([0 < pr < 1 for pr in prob])


def test_get_num_count():
    # if adaptive learning rate is constant, all nums should be 0
    assert all(np.array(ogdlr_after.num.values()) == 0)

    # if adaptive learning rate by counting, all nums should be
    # integers from 0 to number of examples + 1 (from bias)
    clf = OGDLR(alr_schedule='count')
    clf.fit(X[:100], y[:100])
    assert clf.num['BIAS'] == 100  # bias term
    result = set(clf.num.values()) - set(range(N + 1))
    assert result == set([])

    clf = OGDLR(alr_schedule='gradient')
    clf.fit(X[:100], y[:100])
    # if adaptive learning rate by gradient, all nums should be floats
    result = clf.num.values()
    assert all([isinstance(ni, float) for ni in result])


def test_w_and_num_keys():
    assert ogdlr_after.w.keys() == ogdlr_after.num.keys()


@pytest.mark.parametrize('alr, expected', [
    ('gradient', 1.23),
    ('count', 1),
    ('constant', 0),
])
def test_get_delta_num(alr, expected):
    clf = OGDLR(alr_schedule=alr)
    clf.fit(X[:100], y[:100])
    result = clf._get_delta_num(1.23)
    assert result == expected


def test_alr_schedule_error():
    clf = OGDLR(alr_schedule='nonsense')
    with pytest.raises(TypeError):
        clf.fit(X[:10], y[:10])


@pytest.mark.parametrize('class_weight, y, expected', [
    ('auto', [0, 0, 0, 1], [True, True, False, False]),
    ('auto', [0, 0, 1, 1], [False, False, False, False]),
    (1 / 4, [0, 0, 0, 1], [True, True, False, False]),
    (1 / 2, [0, 0, 1, 1], [True, False, False, False]),
    (4 / 5, [0, 0, 1, 1], [False, False, False, False]),
    (1 / 100, [1, 0, 1, 0], [False, True, False, True]),
    (1., [0, 0, 0, 0], [False, False, False, False]),
    (1., [0, 1, 0, 1], [False, False, False, False]),
])
def test_get_skip_sample(class_weight, y, expected):
    with patch('FTRLprox.models.np.random.random') as rand:
        rand.return_value = np.array([3/4, 1/2, 1/4, 1/2])
        skip_sample = ogdlr_after._get_skip_sample(class_weight, y)
        assert (np.array(skip_sample) == expected).all()


@pytest.mark.parametrize('callback_period', [
    (1000),
    (2345),
    (10000),
])
def test_callback_count(callback_period):
    mock_cb = MagicMock()
    mock_cb.plot = MagicMock(return_value=0)
    clf = OGDLR(callback=mock_cb, callback_period=callback_period)
    clf.fit(X, y)
    expected = (N - 1) // callback_period
    result = mock_cb.plot.call_count
    assert result == expected


@pytest.mark.parametrize('n_samples', [
    1,
    123,
    456,
])
def test_valid_history(n_samples):
    clf = OGDLR()
    clf.fit(X[:n_samples], y[:n_samples])

    # validation history as long as training
    assert len(clf.valid_history) == n_samples
    y_true, y_prob = zip(*clf.valid_history)
    # all true values 0 or 1
    assert set(y_true) <= set([0, 1])
    # all y_probs are probabilities
    assert all([isinstance(pr, float) for pr in y_prob])


@pytest.mark.parametrize('n_samples', [
    1,
    123,
    456,
])
def test_predict_proba(n_samples):
    y_prob = ogdlr_after.predict_proba(X[:n_samples])
    assert len(y_prob) == n_samples
    assert all([isinstance(pr, float) for pr in y_prob])


@pytest.mark.parametrize('n_samples', [
    1,
    123,
    456,
])
def test_predict(n_samples):
    y_pred = ogdlr_after.predict(X[:n_samples])
    assert len(y_pred) == n_samples
    assert set(y_pred) <= set([0, 1])


@pytest.mark.parametrize('skip_list, count', [
    ([True, False] * 50, 50),
    ([True, True, False] * 20, 20),
    ([True] * 123, 0),
    ([False] * 456, 456),
])
def test_fit_skip_sample(skip_list, count):
    with patch('FTRLprox.models.OGDLR._get_skip_sample') as skip:
        skip.return_value = skip_list
        with patch('FTRLprox.models.OGDLR._update') as update:
            n_samples = len(skip_list)
            clf = OGDLR()
            clf.fit(X[:n_samples], y[:n_samples])
            assert update.call_count == count


def test_fit_dont_skip_1():
    # if all is skipped, the call count should equal the number of 1's
    # in y
    with patch('FTRLprox.models.OGDLR._update') as update:
        clf = OGDLR()
        clf.fit(X, y, class_weight=0.)
        assert update.call_count == sum(y)