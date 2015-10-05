# -*- coding: utf-8 -*-

from __future__ import division
import warnings

from lasagne.layers import get_all_params
from lasagne.layers import get_output
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adagrad
import numpy as np
import theano
import theano.tensor as T

from _compat import izip
from models import OGDLR
from models import PrintLog


if theano.config.device.startswith('gpu'):
    warnings.warn("You should set the theano.config.device to 'CPU', "
                  "which for this application is much faster than GPU.")


class InitNormal(object):
    def __init__(self, scale=1e-5):
        self.scale = scale

    def __call__(self):
        return self.scale * np.random.randn()


class InitUniform(object):
    def __init__(self, scale=1e-5):
        self.scale = scale

    def __call__(self):
        return 2 * self.scale * (np.random.rand() - 0.5)


def layer_factory(num_cols, num_units=100, num_layers=1, dropout=0.):
    """Produce Lasagne layers for `Neuralnet`.

    Parameters
    ----------
    num_cols : int
      Number of columns used by the network.

    num_units : int
      Number of hidden units. To have different numbers of hidden
      units per layer, roll your own Lasagne layers.

    num_layers : int (default=1)
      Number of layers. If 1, that is only the output layer. For more
      hidden layers, choose a number greater than 1.

    dropout : float (default=0.)
      If 0, no dropout. If greater than 0, add dropout between each
      hidden layer and between hidden and output layer. If you want
      different dropout rates, roll your own Lasagne layers.

    """
    layer = InputLayer((1, 1 + num_cols))
    for i in range(1, num_layers):
        layer = DenseLayer(layer, num_units)
        if dropout:
            layer = DropoutLayer(layer, p=dropout)
    layer = DenseLayer(layer, 2, nonlinearity=softmax)
    return layer


class Neuralnet(OGDLR):
    """Add a hidden layer in addition to the logistic regression.

    Unfortunately, the neural net's gradient checking still suggests
    there must be a bug somewhere, so use this at your own risk.

    """
    def __init__(
            self,
            lambda1=0.,
            lambda2=0.,
            alpha=0.02,
            beta=1.,
            ndims=None,
            alr_schedule='gradient',
            lasagne_layers=None,
            lasagne_update=adagrad,
            interactions=1,
            callbacks=[PrintLog],
            callback_period=10000,
            init_weights=InitUniform(),
            verbose=False,
    ):
        """Parameters
        ----------
        lambda1 (float, default: 1.)
          L1 regularization factor

        lambda2 (float, default: 0.)
          L2 regularization factor

        alpha (float, default: 0.02)
          scales learning rate up

        beta (float, default: 1.)
          scales learning rate down

        ndims (int, default: None)
          Max number of dimensions (use array), if None, no max (use dict)

        interactions : int (default=1):
          Add interaction terms; when 1, no interactins are added,
          when 2, interactions between 2 features are added, etc.

        alr_schedule (string, {'gradient' (default), 'count', 'constant'})
          adaptive learning rate schedule, either decrease with
          gradient or with count or remain constant.

        lasagne_layers: lasagne.layers.Layer (default=None)
          The output layer of a Lasagne network or the output of
          `layer_factory`. If `None`, use `layer_factory` with default
          values.

        lasagne_update: lasagne.updates (default=adagrad)
          An update function from lasagne.updates, e.g. adagrad.

        callbacks (list of callables, default: [PrintLog])
          optional callback function

        callback_period (int, default: 10000)
          period of the callback

        init_weights : callable (default=InitNormal())
          Weight initialization method.

        verbose
          not much here yet

        Notes
        -----
        * alpha must be greater 0.
        * "The optimal value of alpha can vary a fair bit..."
        * "beta=1 is usually good enough; this simply ensures that early
        learning rates are not too high"
        -- McMahan et al.

        """
        if ndims is not None:
            raise NotImplementedError('Neuralnet does not yet support '
                                      'lists for storing weights.')
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.beta = beta
        self.ndims = ndims
        self.alr_schedule = alr_schedule
        self.lasagne_layers = lasagne_layers
        self.lasagne_update = lasagne_update
        self.interactions = interactions
        self.callbacks = callbacks
        self.callback_period = callback_period
        self.init_weights = init_weights
        self.verbose = verbose

    def _initialize(self, X, cols):
        super(Neuralnet, self)._initialize(X, cols)
        self.layers_ = self.lasagne_layers or layer_factory(len(self.cols))
        self._init_theano_funcs()

    def _init_theano_funcs(self):
        Xs = T.fmatrix('Xs')
        ys = T.imatrix('ys')

        params = get_all_params(self.layers_)

        predict_proba = get_output(self.layers_, Xs)
        predict_proba_deterministic = get_output(
            self.layers_, Xs, deterministic=True)
        loss = T.sum(categorical_crossentropy(predict_proba, ys))
        grad_Xs = T.grad(loss, Xs)
        updates = self.lasagne_update(loss, params)

        self.train_func_ = theano.function(
            inputs=[Xs, ys],
            outputs=[predict_proba, loss, grad_Xs],
            updates=updates,
        )
        self.predict_proba_func_ = theano.function(
            inputs=[Xs], outputs=predict_proba_deterministic)

    def _rand_weights(self):
        return self.init_weights()

    def _get_w(self, xt):
        get = self.w.get
        wt = [get(xi, self.init_weights()) for xi in xt]
        return np.array(wt, dtype=np.float32).reshape(1, -1)

    def _get_p(self, xt, weights=None):
        raise NotImplementedError

    def _get_grads(self, yt, wt):
        raise NotImplementedError

    def _vget_delta_num(self, gradt):
        # vectorization of _get_delta_num
        if self.alr_schedule == 'gradient':
            return gradt ** 2
        elif self.alr_schedule == 'count':
            return np.ones_like(gradt)
        elif self.alr_schedule == 'constant':
            return np.zeros_like(gradt)
        else:
            raise TypeError(
                "Do not know learning rate schedule %s" % self.alr_schedule)

    def _update(self, wt, xt, gradt, sample_weight):
        numt = self._get_num(xt)
        # update first layer
        delta_numt = self._vget_delta_num(gradt)
        delta_w = gradt * self.alpha
        delta_w /= (self.beta + np.sqrt(numt).reshape(1, -1))
        for dw, xi, wi, dnum in izip(delta_w[0], xt, wt[0], delta_numt[0]):
            self.w[xi] = wi - dw
            self.num[xi] = self.num.get(xi, 0.) + dnum
        return self

    def _update_valid(self, yt, pt):
        self.valid_history.append((yt, pt[-1][0]))

    def _train(self, xt, yt, wt):
        wt = self._get_w(xt)
        yt = np.array([[yt, 1 - yt]], dtype=np.int32)
        pt, __, gradt = self.train_func_(wt, yt)
        return pt, gradt

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
        y_proba = []
        for t in range(X.shape[0]):
            xt = self._get_x(X[t])
            wt = self._get_w(xt)
            y_proba.append(self.predict_proba_func_(wt))
        y_proba = np.vstack(y_proba)
        return y_proba[:, [1, 0]]

    def weights(self):
        weights = super(Neuralnet, self).weights()
        return np.vstack(weights)
