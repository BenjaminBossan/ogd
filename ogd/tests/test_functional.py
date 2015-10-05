# -*- coding: utf-8 -*-
from __future__ import division
from copy import deepcopy

import numpy as np
import pytest

from ogd.models import OGDLR
from ogd.models import FTRLprox


class TestModelsLearn:
    def test_FTRL_learns(self, ftrl_before, ftrl_after, training_data):
        X, y = training_data

        # test that this model learns something
        score_before = ftrl_before.score(X, y)
        score_after = ftrl_after.score(X, y)
        assert score_before > score_after

    def test_ogdlr_learns(self, ogdlr_before, ogdlr_after, training_data):
        X, y = training_data

        # test that this model learns something
        score_before = ogdlr_before.score(X, y)
        score_after = ogdlr_after.score(X, y)
        assert score_before > score_after

    def test_ogdlr_with_hash_learns(self, hash_before, hash_after,
                                    training_data):
        X, y = training_data

        # test that this model learns something
        score_before = hash_before.score(X, y)
        score_after = hash_after.score(X, y)
        assert score_before > score_after

    def test_ftrl_with_many_cols_learns(self, manycols_before, manycols_after,
                                        training_data):
        X, y = training_data

        # test that this model learns something
        score_before = manycols_before.score(X, y)
        score_after = manycols_after.score(X, y)
        assert score_before > score_after

    def test_nn_learns(self, nn_before, nn_after, training_data):
        X, y = training_data

        # test that this model learns something
        score_before = nn_before.score(X, y)
        score_after = nn_after.score(X, y)
        assert score_before > score_after

    def test_ogd_with_interactions_learns(self, training_data):
        X, y = training_data
        clf = OGDLR(interactions=3)
        clf.fit(X[:5], y[:5])
        score_before = clf.score(X, y)

        # test that this model learns something
        clf.fit(X[5:], y[5:])
        score_after = clf.score(X, y)
        assert score_before > score_after

    def test_ftrl_with_interactions_and_ndims_learns(self, training_data):
        X, y = training_data
        clf = FTRLprox(interactions=3, ndims=2**20)
        clf.fit(X[:5], y[:5])
        score_before = clf.score(X, y)

        # test that this model learns something
        clf.fit(X[5:], y[5:])
        score_after = clf.score(X, y)
        assert score_before > score_after


class TestSimilarPredictions:
    def test_models_same_predictions(self, ftrl_after, ogdlr_after, hash_after,
                                     training_data):
        X, y = training_data

        # for lambda1, lambda2 = 0, OGDLR and FTRLprox should generate the
        # same result. The same goes if hashing is used (except for the
        # rare case of hash collisions. A neural net does not necessarily
        # predict exactly the same outcome.
        y_f = ftrl_after.predict_proba(X)
        y_o = ogdlr_after.predict_proba(X)
        y_h = hash_after.predict_proba(X)
        assert np.allclose(y_f, y_o, atol=1e-15)
        assert np.allclose(y_f, y_h, atol=1e-15)


class TestParamterEffects:
    def test_effect_of_alpha(self, training_data):
        X, y = training_data

        # higher alpha should lead to faster learning, thus higher weights
        mean_abs_weights = []
        for alpha in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]:
            clf = OGDLR(alpha=alpha)
            clf.fit(X[:100], y[:100])
            mean_abs_weights.append(np.abs(clf.weights()).mean())
        assert mean_abs_weights[0] == 0.
        assert all(np.diff(mean_abs_weights) > 0)

    def test_effect_of_beta(self, training_data):
        X, y = training_data

        # higher beta should lead to slower learning, thus lower weights
        mean_abs_weights = []
        for beta in [10 ** n for n in range(7)]:
            clf = OGDLR(beta=beta)
            clf.fit(X[:100], y[:100])
            mean_abs_weights.append(np.abs(clf.weights()).mean())
        assert np.allclose(mean_abs_weights[-1], 0, atol=1e-6)
        assert all(np.diff(mean_abs_weights) < 0)

    @pytest.mark.parametrize('lambda1', [1 / lam for lam in range(1, 6)])
    def test_effect_of_lambda1(self, lambda1):
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
    def test_effect_of_lambda2(self, lambda2):
        # relative difference in gradients should be the same
        weights = range(-5, 0) + range(1, 6)
        clf = OGDLR(lambda2=lambda2)
        grads = clf._get_grads(0, 0, weights)
        # gradient is proportional to weights
        frac = [gr / w for gr, w in zip(grads, weights)]
        assert np.allclose(frac[0], frac)

        # contingencey test: should fail
        abso = [gr * np.sign(w) for gr, w in zip(grads, weights)]
        with pytest.raises(AssertionError):
            assert np.allclose(abso[0], abso)


class TestGridSearch:
    @pytest.mark.slow
    @pytest.mark.parametrize('model', [OGDLR, FTRLprox])
    def test_gridsearch_functional(self, model, training_data):
        X, y = training_data

        # We can perform a grid search
        from sklearn.grid_search import GridSearchCV
        gs_params = {'lambda1': [0, 0.1],
                     'lambda2': [0, 0.01]}
        clf = model()
        gs = GridSearchCV(clf, gs_params)
        gs.fit(X, y)
        gs.best_params_
        gs.best_score_


class TestVerbose:
    def test_verbose_output(self, ogdlr_after, training_data, capsys):
        X, y = training_data
        clf = deepcopy(ogdlr_after)

        clf.callback_period = 100
        clf.verbose = 1
        clf.fit(X[:1000], y[:1000])
        printed = capsys.readouterr()[0].split('\n')

        # Expected number of prints:
        # 1 for start print
        # 2 for header
        # num samples / callback - 1 for results.
        # 1 for end print
        assert len(printed) == 2 + 1 + (10 - 1) + 1

        assert printed[1] == '    period|     valid loss|  duration'
        assert printed[2] == '----------|---------------|----------'
        for i in range(1, 10):
            assert printed[i + 2].startswith('       ' + str(i))
            assert printed[i + 2].endswith('s')
