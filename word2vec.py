# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine


def cosine_dist(u, v):
    return np.dot(u, v.T) / norm(u) / norm(v)


def mse(u, v):
    return np.mean((u - v) ** 2)


def centered_cosine(u, v):
    u = u / u.sum()
    v = v / v.sum()
    return np.dot(u, v.T)


def logloss(y_true, y_pred):
    """ As provided by kaggle:
    https://www.kaggle.com/wiki/LogarithmicLoss
    """
    epsilon = 1e-15
    y_pred = np.maximum(epsilon, y_pred)
    y_pred = np.minimum(1-epsilon, y_pred)
    ll = (sum(y_true * np.log(y_pred) +
              np.subtract(1, y_true) *
              np.log(np.subtract(1, y_pred)))
          )
    ll = ll * -1.0/len(y_true)
    return ll


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def spit_sentences(X, sep='\n'):
    sentences = X.split(sep)
    for sentence in sentences:
        yield sentence


def spit_words(sentence, n, sep=' '):
    sentence = sentence.rstrip().lstrip()
    while '  ' in sentence:
        sentence = sentence.replace('  ', ' ')
    words = sentence.split(sep)
    m = len(words)
    for i in range(m - n):
        yield words[i: i + n], words[i + n]


class Word2vec(object):
    """ Using negative sampling """

    def __init__(self, vocab, ndims=100, alpha=0.01, neg_samples=5,
                 n_iter=None, verbose=False):
        assert isinstance(vocab, list)
        self.vocab = vocab
        self.ndims = ndims
        self.alpha = alpha
        self.neg_samples = neg_samples
        self.n_iter = n_iter
        self.verbose = verbose

    def _init_vecs(self):
        scale = 0.001
        self.vocab_size = len(self.vocab)
        self.word_index = dict(zip(self.vocab, range(self.vocab_size)))
        self.vecs = np.random.randn(self.vocab_size, self.ndims) * scale
        self.bias = np.random.randn(1, self.ndims) * scale
        self.nodevecs = np.zeros((self.vocab_size, self.ndims))
        # self.nodevecs = np.random.randn(self.vocab_size, self.ndims) * scale
        self.valid_history = []
        self._y_true = np.zeros((self.neg_samples + 1, 1))
        self._y_true[0] = 1.
        self._is_init = True

    def _get_word_index(self, word):
        return self.word_index[word]

    def _get_neg_samples(self):
        # draw with replacement and no comparison with actual word
        # maybe add more sophisticated approach later
        idx_neg_samples = np.random.randint(0, self.vocab_size,
                                            self.neg_samples)
        return idx_neg_samples

    def _feed_forward(self, vecs):
        vecs = np.vstack((vecs, self.bias))  # add bias
        return np.array(vecs).sum(axis=0)

    def _get_preds(self, idx_words, idx_targets):
        # return predictions, with the prediction of the target word
        # being first and the negative samples coming afterwords, as
        # well as the activation

        vecs = [self.vecs[idx] for idx in idx_words]
        activations = self._feed_forward(vecs)
        weights = np.array(self.nodevecs[idx_targets])

        preds = sigmoid(np.dot(activations, weights.T)).reshape(-1, 1)
        return preds, activations, weights

    def _update(self, y_err, activations, weights, idx_words, idx_targets):
        grad_vecs = np.dot(y_err.T, weights)
        self.vecs[idx_words] -= self.alpha * grad_vecs
        self.bias -= self.alpha * grad_vecs

        grad_nodevecs = np.outer(y_err.T, activations)
        self.nodevecs[idx_targets] -= self.alpha * grad_nodevecs

    def fit(self, X, y):
        if not hasattr(self, '_is_init'):
            self._init_vecs()

        if self.n_iter is None:
            self._fit(X, y)
        else:
            alpha_steps = self.alpha / self.n_iter
            for i in range(self.n_iter):
                self._fit(X, y)
                self.alpha -= alpha_steps
                if self.verbose:
                    print("{} of {}, validation score: {:0.4f}".format(
                        i + 1, self.n_iter, np.mean(self.valid_history)))

    def _fit(self, X, y):
        n = X.shape[0]
        for t in range(n):
            words = X[t]
            target = y[t]
            try:
                idx_words = [self._get_word_index(word) for word in words]
                idx_target = self._get_word_index(target)
                idx_neg_samples = self._get_neg_samples()
            except KeyError:
                continue

            idx_targets = np.concatenate(([idx_target], idx_neg_samples))
            preds, activations, weights = self._get_preds(idx_words,
                                                          idx_targets)

            y_err = preds - self._y_true
            self._update(y_err, activations, weights, idx_words, idx_targets)
            # accuracy
            self.valid_history.append(logloss(self._y_true, preds))

    def predict_proba(self, X):
        n = X.shape[0]
        y_prob = []
        for t in range(n):
            words = X[t]
            preds = self._get_preds_activations(words)[0]
            y_prob.append(preds)
        return np.array(y_prob)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        amax = np.argmax(y_prob, axis=1)
        y_pred = [self.vocab[idx] for idx in amax]
        return np.array(y_pred)

    def most_similar(self, word, topn=5, func=mse):
        # similarity based on word vectors (embeddings)
        vec = self.vecs[self._get_word_index(word)]
        dists = [(i, func(vec, target_vec))
                 for i, target_vec in enumerate(self.vecs)]
        dists = sorted(dists, key=lambda x: x[1])
        return [self.vocab[i] for i in zip(*dists[1:1 + topn])[0]]

    def most_similar2(self, word, topn=5, func=mse):
        # similarity based on weights of layer
        vec = self.nodevecs[self._get_word_index(word)]
        dists = [(i, func(vec, target_vec))
                 for i, target_vec in enumerate(self.nodevecs)]
        dists = sorted(dists, key=lambda x: x[1])
        return [self.vocab[i] for i in zip(*dists[1:1 + topn])[0]]
