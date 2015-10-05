# -*- coding: utf-8 -*-

from __future__ import division

from itertools import permutations
from mock import Mock
from mock import patch

import numpy as np
import pytest
from sklearn.utils import murmurhash3_32 as mmh

from ogd.models import OGDLR
from ogd.models import SEED
from ogd.nn import Neuralnet
from ogd.utils import GetList
from conftest import N
from conftest import NDIMS


class TestKeys:
    def test_keys_dictionary(self, ogdlr_after, ftrl_after):
        ogdlr_keys = ogdlr_after.keys()
        ftrl_keys = ftrl_after.keys()
        assert ogdlr_keys == ftrl_keys
        known_keys = set(['BIAS', 'weather__rainy', 'weather__sunny',
                          'temperature__cold', 'temperature__warm'])
        assert known_keys <= set(ftrl_keys)

    def test_keys_list(self, hash_after):
        assert hash_after.keys() is None

    def test_w_and_num_keys(self, ogdlr_after):
        assert ogdlr_after.w.keys() == ogdlr_after.num.keys()


class TestWeights:
    def test_weights(self, ogdlr_after, ftrl_after, hash_after):
        ogdlr_weights = ogdlr_after.weights()
        ftrl_weights = ftrl_after.weights()
        hash_keys = [mmh(key, seed=SEED) % NDIMS for key in ftrl_after.keys()]
        hash_weights = hash_after._get_w(hash_keys)

        assert np.allclose(ogdlr_weights, ftrl_weights)
        assert np.allclose(hash_weights, ftrl_weights)

    def test_weights_list(self, hash_after):
        assert hash_after.weights() is None


class TestGradientNumerically:
    @pytest.mark.parametrize('alr', [
        'gradient',
        'count',
        'constant',
    ])
    def test_ogdlr_grad_numerically(self, alr, training_data):
        X, y = training_data
        epsilon = 1e-6
        clf = OGDLR(alr_schedule=alr)
        clf.fit(X[:100], y[:100])
        for xx, yy in zip(X[:10], y[:10]):
            grad, grad_num = clf.numerical_grad(xx, yy, epsilon)
            assert np.allclose(grad, grad_num, atol=epsilon)

    @pytest.mark.parametrize('lambda1', [0, 0.1, 1, 10, 100])
    def test_ogdlr_grad_numerically_l1(self, lambda1, training_data):
        X, y = training_data
        epsilon = 1e-6
        clf = OGDLR(lambda1=lambda1)
        clf.fit(X[:100], y[:100])
        for xx, yy in zip(X[:10], y[:10]):
            grad, grad_num = clf.numerical_grad(xx, yy, epsilon)
            assert np.allclose(grad, grad_num, atol=epsilon)

    @pytest.mark.parametrize('lambda2', [0, 0.1, 1, 10, 100])
    def test_ogdlr_grad_numerically_l2(self, lambda2, training_data):
        X, y = training_data
        epsilon = 1e-6
        clf = OGDLR(lambda2=lambda2)
        clf.fit(X[:100], y[:100])
        for xx, yy in zip(X[:10], y[:10]):
            grad, grad_num = clf.numerical_grad(xx, yy, epsilon)
            assert np.allclose(grad, grad_num, atol=epsilon)

    @pytest.mark.parametrize(
        'args',
        list(set(permutations(2 * [0] + 2 * [1e-4] + 2 * [1e-2] + 2 * [1], 2)))
    )
    def test_ogdlr_grad_numerically_l1_l2(self, args, training_data):
        X, y = training_data

        # test all combinations of lambda1 and lambda2 values of 0,
        # 0.5, and 2 and for alpha = 0.02 and beta = 1.
        epsilon = 1e-6
        clf = OGDLR(*args)
        clf.fit(X[:100], y[:100])
        for xx, yy in zip(X[:10], y[:10]):
            grad, grad_num = clf.numerical_grad(xx, yy, epsilon)
            assert np.allclose(grad, grad_num, atol=epsilon)


class TestInit:
    @pytest.mark.parametrize('arg, expected', [
        (None, dict),
        (123, list),
        (123, GetList),
    ])
    def test_ogdlr_init_ndims(self, arg, expected, training_data):
        X, y = training_data

        clf = OGDLR(ndims=arg)
        clf.fit(X[:10], y[:10])
        assert isinstance(clf.w, expected)
        assert isinstance(clf.num, expected)

    def test_ogdlr_init_cols(self, ogdlr_after, training_data):
        X, y = training_data

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

    def test_alr_schedule_error(self, training_data):
        X, y = training_data
        clf = OGDLR(alr_schedule='nonsense')
        with pytest.raises(TypeError):
            clf.fit(X[:10], y[:10])

    def test_neuralnet_init_args(self):
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

    def test_neuralnet_init_exceptions(self):
        with pytest.raises(NotImplementedError):
            Neuralnet(num_units=10, ndims=2 ** 20)


class TestGetMethods:
    def test_get_x(self, ogdlr_after, training_data):
        X, y = training_data

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

    def test_get_x_interactions(self):
        clf = OGDLR(interactions=2)
        clf.cols = ['c1', 'c2', 'c3']
        result = clf._get_x(['sunny', 'cold', '123'])
        expected = ['BIAS', 'c1__sunny', 'c2__cold', 'c3__123',
                    'c1__sunny c2__cold', 'c1__sunny c3__123',
                    'c2__cold c3__123']
        assert result == expected

    def test_get_w(self, ogdlr_after):
        for clf, key, expected in [
            (ogdlr_after, ogdlr_after.w.keys()[0], ogdlr_after.w.values()[0]),
            (
                ogdlr_after,
                ogdlr_after.w.keys()[-1],
                ogdlr_after.w.values()[-1]
            ),
            (ogdlr_after, 'key-not-present', 0.)
        ]:
            result = clf._get_w([key])[0]
            assert result == expected

    def test_get_p(self, ogdlr_after, training_data):
        X = training_data[0]
        # probabilities are between 0 and 1
        Xs = [ogdlr_after._get_x(x) for x in X]
        prob = [ogdlr_after._get_p(xt) for xt in Xs]
        assert all([0 < pr < 1 for pr in prob])

    def test_get_num_count(self, ogdlr_after, training_data):
        X, y = training_data

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

    @pytest.mark.parametrize('alr, expected', [
        ('gradient', 1.23),
        ('count', 1),
        ('constant', 0),
    ])
    def test_get_delta_num(self, alr, expected, training_data):
        X, y = training_data
        clf = OGDLR(alr_schedule=alr)
        clf.fit(X[:100], y[:100])
        result = clf._get_delta_num(1.23)
        assert result == expected ** 2


class TestSkipSample:
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
    def test_get_skip_sample(self, class_weight, y, expected, ogdlr_after):
        with patch('ogd.models.np.random.random') as rand:
            rand.return_value = np.array([3 / 4, 1 / 2, 1 / 4, 1 / 2])
            skip_sample = ogdlr_after._get_skip_sample(class_weight, y)
            assert (np.array(skip_sample) == expected).all()

    @pytest.mark.parametrize('skip_list, count', [
        ([True, False] * 50, 50),
        ([True, True, False] * 20, 20),
        ([True] * 123, 0),
        ([False] * 456, 456),
    ])
    def test_fit_skip_sample(self, skip_list, count, training_data):
        X, y = training_data
        with patch('ogd.models.OGDLR._get_skip_sample') as skip:
            skip.return_value = skip_list
            with patch('ogd.models.OGDLR._update') as update:
                n_samples = len(skip_list)
                clf = OGDLR()
                clf.fit(X[:n_samples], y[:n_samples])
                assert update.call_count == count

    def test_fit_dont_skip_1(self, training_data):
        X, y = training_data
        # if all is skipped, the call count should equal the number of 1's
        # in y
        with patch('ogd.models.OGDLR._update') as update:
            clf = OGDLR()
            clf.fit(X, y, class_weight=0.)
            assert update.call_count == sum(y)


class TestCallback:
    @pytest.mark.parametrize('callback_period', [
        (1000),
        (2345),
        (10000),
    ])
    def test_callback_count(self, callback_period, training_data):
        X, y = training_data
        mock_cb = Mock()
        clf = OGDLR(callbacks=[Mock(return_value=mock_cb)],
                    callback_period=callback_period)
        clf.fit(X, y)
        expected = (N - 1) // callback_period
        result = mock_cb.call_count
        assert result == expected

    @pytest.mark.parametrize('n_samples', [
        1,
        123,
        456,
    ])
    def test_valid_history(self, n_samples, training_data):
        X, y = training_data

        clf = OGDLR()
        clf.fit(X[:n_samples], y[:n_samples])

        # validation history as long as training
        assert len(clf.valid_history) == n_samples
        y_true, y_prob = zip(*clf.valid_history)
        # all true values 0 or 1
        assert set(y_true) <= set([0, 1])
        # all y_probs are probabilities
        assert all([isinstance(pr, float) for pr in y_prob])


class TestPredictProba:
    @pytest.mark.parametrize('n_samples', [
        1,
        123,
        456,
    ])
    def test_predict_proba(self, n_samples, ogdlr_after, training_data):
        X, y = training_data
        y_prob = ogdlr_after.predict_proba(X[:n_samples])
        assert len(y_prob) == n_samples
        assert all([isinstance(pr, float) for pr in y_prob])


class TestPredict:
    @pytest.mark.parametrize('n_samples', [
        1,
        123,
        456,
    ])
    def test_predict(self, n_samples, ogdlr_after, training_data):
        X = training_data[0]
        y_pred = ogdlr_after.predict(X[:n_samples])
        assert len(y_pred) == n_samples
        assert set(y_pred) <= set([0, 1])


class TestFTRLManyCols:
    @pytest.fixture
    def many_cols(self):
        from ogd.models import MANYCOLS
        return np.random.randint(0, 50, size=(500, MANYCOLS))

    @pytest.fixture
    def few_cols(self, many_cols):
        return many_cols[:, :-1]

    @pytest.fixture
    def y(self, many_cols):
        y = (many_cols.mean(1) > 1.01 * np.mean(many_cols)).astype(int)
        return y

    @pytest.fixture
    def ftrl(self):
        from ogd.models import FTRLprox

        mock_few = Mock(side_effect=lambda x: [1 for __ in x])
        mock_many = Mock(side_effect=lambda x: [1 for __ in x])
        ftrl = FTRLprox()
        ftrl._get_w_few_cols = mock_few
        ftrl._get_w_many_cols = mock_many
        return ftrl

    @pytest.fixture
    def ftrl_few_cols(self, ftrl, few_cols, y):
        ftrl.fit(few_cols, y)
        return ftrl

    @pytest.fixture
    def ftrl_many_cols(self, ftrl, many_cols, y):
        ftrl.fit(many_cols, y)
        return ftrl

    def test_few_cols_called_when_few_cols(self, ftrl_few_cols, y):
        assert ftrl_few_cols._get_w_few_cols.call_count == len(y)
        assert ftrl_few_cols._get_w_many_cols.call_count == 0

    def test_many_cols_called_when_many_cols(self, ftrl_many_cols, y):
        assert ftrl_many_cols._get_w_few_cols.call_count == 0
        assert ftrl_many_cols._get_w_many_cols.call_count == len(y)

    def test_new_and_many_similar_predictions(self, few_cols, many_cols, y):
        from ogd.models import FTRLprox
        ftrl_few_cols = FTRLprox()
        ftrl_many_cols = FTRLprox()

        pred_few = ftrl_few_cols.fit(few_cols, y).predict_proba(few_cols)
        pred_many = ftrl_many_cols.fit(many_cols, y).predict_proba(many_cols)
        assert np.allclose(pred_few, pred_many, rtol=0.05)

        # exclude trivial case where all predictions are just 0 or 1
        assert not np.allclose(pred_few[0], pred_few)

        # exclude that predictions are absolutely the same
        assert not (pred_few == pred_many).all()
