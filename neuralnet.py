# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from utils import logloss
from utils import sigmoid

from models import OGDLR


SCALING = 1e-4


class Neuralnet(OGDLR):
    def __init__(self, *args, **kwargs):
        try:
            num_units = kwargs.pop('num_units')
        except KeyError:
            raise TypeError('Please specify the "num_units" argument.')
        if kwargs.get('ndims', None) is not None:
            raise NotImplementedError('Neuralnet does not yet support'
                                      'lists for storing weights.')
        self.num_units = num_units
        super(Neuralnet, self).__init__(*args, **kwargs)

    def _initialize_dicts(self):
        self.w = [{}, self._rand_weights().T]
        self.num = [{}, np.zeros((self.num_units, 1))]

    def _initialize(self, X, cols):
        super(Neuralnet, self)._initialize(X, cols)

    def _rand_weights(self):
        return SCALING * np.random.randn(1, self.num_units)

    def _get_w(self, xt):
        # # first weights are safed sparsely:
        # wt = np.zeros((0, self.num_units))
        # for xi in xt:
        #     if xi not in self.w[0]:
        #         rand_weight = self._rand_weights()
        #         wt = np.vstack((wt, rand_weight))
        #         # set w[0][xi] if not present
        #         self.w[0][xi] = rand_weight
        #     else:
        #         wt = np.vstack((wt, self.w[0][xi]))
        wt = np.zeros((len(xt), self.num_units))
        for i, xi in enumerate(xt):
            if xi not in self.w[0]:
                rand_weight = self._rand_weights()
                wt[i] = rand_weight
                # set w[0][xi] if not present
                self.w[0][xi] = rand_weight
            else:
                wt[i] = self.w[0][xi]
        return [wt] + self.w[1:]

    def _get_p(self, xt, weights=None):
        if weights is None:
            weights = self._get_w(xt)
        # first layer is condensed weight matrix, which is why it is
        # sufficient to multiply it by 1
        activities = [np.ones((1, len(xt)))]
        # feed forward
        for weight in weights:
            activities.append(sigmoid(np.dot(activities[-1], weight)))
        return activities

    def _get_deltas(self, y_err, weights, activities):
        deltas = [y_err * activities[-1] * (1 - activities[-1])]
        # backpropagation
        for weight, act in zip(weights[::-1][:-1], activities[::-1][1:-1]):
            delta = np.dot(deltas[0], weight.T) * act * (1 - act)
            deltas.insert(0, delta)
        return deltas

    def _get_grads(self, yt, activities, weights):
        # Get the gradient as a function of true value and predicted
        # probability, as well as the regularization terms. The
        # gradient is the derivative of the cost function with
        # respect to each wi.
        y_err = activities[-1] - yt
        deltas = self._get_deltas(y_err, weights, activities)
        grads = []
        for weight, delta, act in zip(weights, deltas, activities[:-1]):
            cost = self._get_regularization(weight)
            grad = np.dot(act.T, delta) + cost
            grads.append(grad)
        return grads

    def _get_num(self, xt):
        # first nums are safed sparsely:
        num0 = np.array([self.num[0].get(xi, 0.) for xi in xt])
        return [num0] + self.num[1:]

    def _vget_delta_num(self, gradt):
        # vectorization of _get_delta_num
        if self.alr_schedule == 'gradient':
            return gradt ** 2
        elif self.alr_schedule == 'count':
            return np.ones_like(gradt)
        elif self.alr_schedule == 'constant':
            return np.zeros_like(gradt)
        else:
            raise TypeError("Do not know adaptive learning"
                            "rate schedule %s" % self.alr_schedule)

    def _update(self, wt, xt, gradt, sample_weight):
        numt = self._get_num(xt)
        # update first layer
        grad = gradt[0]
        delta_numt = self._vget_delta_num(gradt[0].sum(1))
        delta_w = grad * self.alpha
        delta_w /= self.beta + np.sqrt(numt[0].reshape(-1, 1))
        for dw, xi, dnum in zip(delta_w, xt, delta_numt):
            self.w[0][xi] -= dw
            self.num[0][xi] = self.num[0].get(xi, 0.) + dnum

        # update other layers
        for l in range(1, len(self.w)):
            delta_numt = self._vget_delta_num(gradt[l])
            # update weights
            grad = gradt[l]
            delta_w = grad * self.alpha / (self.beta + np.sqrt(numt[l]))
            self.w[l] -= delta_w
            # update num counts
            self.num[l] += delta_numt
        return self

    def _update_valid(self, yt, pt):
        self.valid_history.append((yt, pt[-1][0][0]))

    def predict_proba(self, X):
        """ Predict probability for class 1.

        Parameters
        ----------

        X : numpy.array, shape = [n_samples, n_features]
          Samples

        Returns
        -------

        y_prob : array, shape = [n_samples]
          Predicted probability for class 1
        """
        X = [self._get_x(xt) for xt in X]
        pt = [self._get_p(xt)[-1] for xt in X]
        y_prob = np.array(pt).squeeze()
        return y_prob

    def keys(self):
        """Return the keys saved by the model.

        This only works if the model uses the dictionary method for
        storing its keys and values. Otherwise, it returns None.

        Returns
        -------

        keys : None or list of strings
          The keys of the model if they can be retrieved else None.
        """
        if self.ndims is None:
            return self.w[0].keys()
        else:
            return None

    def weights(self):
        """Return the weights saved by the model.

        The weights are not necessarily the same as in the 'w'
        attribute.

        Returns
        -------
        weights : None or list of floats
          The weights of the model. If the model uses a dictionary for
          storing the weights, they are returned in the same order as
          the keys. If the model uses hashing to a list to store the
          weights, returns None.

        """
        if self.ndims is None:
            weights = self._get_w(self.keys())
            return weights
        else:
            return None
