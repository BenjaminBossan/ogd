""" Attempt at implementing Word2vec without hierarchical softmax layer.
"""

# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from utils import logloss
from utils import sigmoid


class Word2vec(object):
    def __init__(self, vocab, ndims=100, alpha=0.01):
        self.vocab = vocab
        self.ndims = ndims
        self.alpha = alpha

    def get_v(self, word):
        return self.vecs[word]

    def _init_vecs(self):
        self.vecs = {}
        for word in self.vocab:
            self.vecs[word] = np.random.randn(self.ndims) * 0.001
        self._is_init = True

    def fit(self, X, y):
        if not hasattr(self, '_is_init'):
            self._init_vecs()
        n = X.shape[0]
        for t in range(n):
            vecs = [self.get_v(word) for word in X[t]]
            vecsum = np.array(vecs).sum(axis=0)
            grads = {}
            for word in self.vocab:
                target_vec = self.get_v(word)
                pred = sigmoid(np.dot(vecsum, target_vec.T))
                grad = (pred - (word == y[t])) * vecsum
                grads[word] = grad
            for word in X[t]:
                self.vecs[word] -= self.alpha * grad

    def predict(self, X):
        n = X.shape[0]
        y_pred = []
        for t in range(n):
            vecs = [self.get_v(word) for word in X[t]]
            vecsum = np.array(vecs).sum(axis=0)
            preds = {}
            for word in self.vocab:
                target_vec = self.get_v(word)
                pred = sigmoid(np.dot(vecsum, target_vec.T))
                preds[word] = pred
            highest_prob = sorted(preds, lambda x: -x[1])
            y_pred.append(highest_prob[0][0])


# some utilities for streaming words as required by Word2vec
def spit_sentences(X, sep='\n'):
    sentences = X.split(sep)
    for sentence in sentences:
        yield sentence


def spit_words(sentence, n, sep=' '):
    words = sentence.split(sep)
    m = len(words)
    for i in range(m - n - 1):
        yield words[i: i + n], words[i + n + 1]
