""" Follow the (proximally) Regularized Leader model.

FTRL proximal according to: McMahan et al. 2013:
Ad click prediction: a view from the trenches.

'OGDLR' model adapted from Sam Hocevar's:
https://www.kaggle.com/c/criteo-display-ad-challenge/forums/t/10322/beat-the-benchmark-with-less-then-200mb-of-memory

"""

# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from sklearn.utils.murmurhash import murmurhash3_32 as mmh

from utils import GetList
from utils import logloss
from utils import sigmoid


SEED = 17411
np.random.seed(seed=SEED)


class OGDLR(object):
    """Online Gradient Descent Logistic Regression

    Model adapted from Sam Hocevar with interface similar to scikit
    learn's.

    Only support for single class prediction right now.
    The model supports per coordinate learning rate schedules.

    """
    def __init__(self,
                 lambda1=0.,
                 lambda2=0.,
                 alpha=0.02,
                 beta=1.,
                 ndims=None,
                 alr_schedule='gradient',
                 callback=None,
                 callback_period=10000,
                 verbose=False):
        """Parameters
        ----------
        * lambda1 (float, default: 1.): L1 regularization factor
        * lambda2 (float, default: 0.): L2 regularization factor
        * alpha (float, default: 0.02): scales learning rate up
        * beta (float, default: 1.): scales learning rate down
        * ndims (int, default: None): Max number of dimensions (use array),
          if None, no max (use dict)
        * alr_schedule (string, {'gradient' (default), 'count', 'constant'}):
          adaptive learning rate schedule, either decrease with gradient
          or with count or remain constant.
        * callback (func, default: None): optional callback function
        * callback_period (int, default: 10000): period of the callback
        * verbose: not much here yet

        Notes on parameter choice:
        * alpha must be greater 0.
        * "The optimal value of alpha can vary a fair bit..."
        * "beta=1 is usually good enough; this simply ensures that early
        learning rates are not too high"
        -- McMahan et al.

        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.beta = beta
        self.ndims = ndims
        self.alr_schedule = alr_schedule
        self.callback = callback
        self.callback_period = callback_period
        self.verbose = verbose

    def _initialize_dicts(self):
        # weights and number of iterations (more or less) for adaptive
        # learning rate
        self.w = {}
        self.num = {}

    def _initialize_lists(self):
        # weights and number of iterations (more or less) for adaptive
        # learning rate
        self.w = GetList([0.] * self.ndims)
        self.num = GetList([0.] * self.ndims)

    def _initialize_cols(self, X, cols):
        # column names
        m = X.shape[1]
        if cols is not None:
            if len(set(cols)) != len(cols):
                raise ValueError("Columns contain duplicate names.")
            self.cols = cols  # name of the columns
        else:
            # generate generic column names
            s1 = "col{0:0%dd}" % len(str(m))
            self.cols = [s1.format(i) for i in range(m)]

    def _initialize(self, X, cols):
        """Initialize some required attributes on first call.
        """
        if self.ndims is None:
            self._initialize_dicts()
        else:
            self._initialize_lists()
        self._initialize_cols(X, cols)
        # Validation for each single iteration. Could be changed
        # to collection.deque if speed and size are an issue here.
        self.valid_history = []
        self._is_initialized = True

    def _get_x(self, xt):
        # 'BIAS' is the bias term.  The other keys are created as a
        # the column name joined with the value itself. This should
        # ensure unique keys.
        if self.ndims is None:
            x = ['BIAS'] + ['__'.join((col, str(val))) for col, val
                            in zip(self.cols, xt)]
        else:
            x = [mmh('BIAS', seed=SEED) % self.ndims]
            x += [mmh(key + '__' + str(val), seed=SEED) % self.ndims
                  for key, val in zip(self.cols, xt)]
        return x

    def _get_w(self, xt):
        wt = [self.w.get(xi, 0.) for xi in xt]
        return wt

    def _get_p(self, xt, wt=None):
        if wt is None:
            wt = self._get_w(xt)
        wTx = sum(wt)
        # bounded sigmoid
        wTx = max(min(wTx, 20.), -20.)
        return sigmoid(wTx)

    def _get_num(self, xt):
        numt = [self.num.get(xi, 0.) for xi in xt]
        return numt

    def _get_delta_num(self, grad):
        if self.alr_schedule == 'gradient':
            return grad * grad
        elif self.alr_schedule == 'count':
            return 1
        elif self.alr_schedule == 'constant':
            return 0
        else:
            raise TypeError("Do not know learning"
                            "rate schedule %s" % self.alr_schedule)

    def _get_grads(self, yt, pt, wt):
        # Get the gradient as a function of true value and predicted
        # probability, as well as the regularization terms. The
        # gradient is the derivative of the cost function with
        # respect to each wi.
        y_err = pt - yt
        costs = self._get_regularization(wt)
        grads = [y_err + cost for cost in costs]
        return grads

    def _get_regularization(self, wt):
        # Get cost from L1 and L2 regularization. Currently, bias is
        # also regularized but should not matter much.
        costs = self.lambda1 * np.sign(wt)  # L1
        costs += 2 * self.lambda2 * np.array(wt)  # L2
        return costs

    def _get_skip_sample(self, class_weight, y):
        if class_weight == 'auto':
            class_weight = np.mean(y) / (1 - np.mean(y))
        if class_weight != 1.:
            rand = np.random.random(len(y))
            skip_sample = rand > class_weight
            skip_sample[np.array(y) == 1] = False  # don't skip 1 labels
            if self.verbose:
                print("Using {:0.3f}% of negative samples".format(
                    100 * class_weight))
        else:
            skip_sample = [False] * len(y)
        return skip_sample

    def _call_back(self, t):
        if (
                (t % self.callback_period == 0) &
                (self.callback is not None) &
                (t != 0)
        ):
            self.callback.plot(self)

    def _update(self, wt, xt, gradt, sample_weight):
        # note: wt is not used here but is still passed so that the
        # interface for FTRL proximal (which requires wt) can stay the
        # same
        numt = self._get_num(xt)
        for xi, numi, gradi in zip(xt, numt, gradt):
            delta_w = gradi * self.alpha / (self.beta + np.sqrt(numi))
            delta_w /= sample_weight
            self.w[xi] = self.w.get(xi, 0.) - delta_w
            delta_num = self._get_delta_num(gradi)
            self.num[xi] = numi + delta_num
        return self

    def _update_valid(self, yt, pt):
        self.valid_history.append((yt, pt))

    def fit(self, X, y, cols=None, class_weight=1.):
        """Fit OGDLR model.

        Can also be used as if 'partial_fit', i.e. calling 'fit' more
        than once continues with the learned weights.

        Parameters
        ----------

        X : numpy.array, shape = [n_samples, n_features]
          Training data

        y : numpy.array, shape = [n_samples]
          Target values

        cols : list of strings, shape = [n_features] (default : None)
          The name of the columns used for training. They are used as
          part of the first part of the key. If None is given, give
          generic names to columns ('col01', 'col02', etc.)

        class_weight : float or 'auto' (optional, default=1.0)
          Subsample the negative class. Assumes that classes are {0,
          1} and that the negative (0) class is the more common
          one. If these assumptions are not met, don't change this
          parameter.

        Returns
        -------

        self

        """
        if not hasattr(self, '_is_initialized'):
            self._initialize(X, cols)
        # skip samples if class weight is given
        skip_sample = self._get_skip_sample(class_weight, y)

        n = X.shape[0]
        for t in range(n):
            # if classes weighted differently
            if skip_sample[t]:
                continue
            else:
                sample_weight = 1. if y[t] == 1 else class_weight

            yt = y[t]
            xt = self._get_x(X[t])
            wt = self._get_w(xt)
            pt = self._get_p(xt, wt)
            gradt = self._get_grads(yt, pt, wt)
            self._update(wt, xt, gradt, sample_weight)
            self._update_valid(yt, pt)
            self._call_back(t)

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
        pt = [self._get_p(xt) for xt in X]
        y_prob = np.array(pt)
        return y_prob

    def predict(self, X):
        """ Predict class label for samples in X.

        Parameters
        ----------

        X : numpy.array, shape = [n_samples, n_features]
          Samples

        Returns
        -------

        y_pred : array, shape = [n_samples]
          Predicted class label per sample.
        """
        pt = self.predict_proba(X)
        y_pred = (pt > 0.5).astype(int)
        return y_pred

    def _cost_function(self, xt, wt, y):
        pt = self._get_p(xt, wt)
        ll = logloss([y], [pt])
        l1 = self.lambda1 * np.abs(wt)
        l2 = self.lambda2 * (np.array(wt) ** 2)
        J = ll + l1 + l2
        return J

    def numerical_grad(self, x, y, epsilon=1e-9):
        """Calculate the gradient and the gradient determined
        numerically; they should be very close.

        Use this function to verify that the gradient is determined
        correctly. The fit method needs to be called once before this
        method may be invoked.

        Parameters
        ----------
        x : list of strings
          The keys; just a single row.

        y : int
          The target to be predicted.

        epsilon : float (default: 1e-6)
          The shift applied to the weights to determine the numerical
          gradient. A small but not too small value such as the
          default should do the job.

        Returns
        -------

        grad : float
          The gradient as determined by this class. For cross-entropy,
          this is simply the prediction minus the true value.

        grad_num : float

          The gradient as determined numerically.

        """
        # analytic
        xt = self._get_x(x)
        wt = self._get_w(xt)
        pt = self._get_p(xt, wt)
        grad = self._get_grads(y, pt, wt)

        # numeric: vary each wi
        grad_num = []
        for i in range(len(wt)):
            wt_pe, wt_me = wt[:], wt[:]
            wt_pe[i] += epsilon
            wt_me[i] -= epsilon
            cost_pe = self._cost_function(xt, wt_pe, y)[i]
            cost_me = self._cost_function(xt, wt_me, y)[i]
            grad_num.append((cost_pe - cost_me) / 2 / epsilon)
        return grad, grad_num

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
            return self.w.keys()
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


class FTRLprox(OGDLR):
    """FTRL proximal model.

    Only support for single class prediction right now.
    The model supports per coordinate learning rate schedules.
    """
    def __init__(self,
                 lambda1=1.,
                 lambda2=0.,
                 alpha=0.02,
                 beta=1.,
                 ndims=None,
                 alr_schedule='gradient',
                 callback=None,
                 callback_period=10000,
                 verbose=False):
        """Parameters
        ----------
        * lambda1 (float, default: 1.): L1 regularization factor
        * lambda2 (float, default: 0.): L2 regularization factor
        * alpha (float, default: 0.02): scales learning rate up
        * beta (float, default: 1.): scales learning rate down
        * ndims (int, default: None): Max number of dimensions (use array),
          if None, no max (use dict)
        * alr_schedule (string, {'gradient' (default), 'count', 'constant'}):
          adaptive learning rate schedule, either decrease with gradient
          or with count or remain constant.
        * callback (func, default: None): optional callback function
        * callback_period (int, default: 10000): period of the callback
        * verbose: not much here yet

        Notes on parameter choice:
        * alpha must be greater 0.
        * "The optimal value of alpha can vary a fair bit..."
        * "beta=1 is usually good enough; this simply ensures that early
        learning rates are not too high"
        -- McMahan et al.

        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.beta = beta
        self.ndims = ndims
        self.alr_schedule = alr_schedule
        self.callback = callback
        self.callback_period = callback_period
        self.verbose = verbose

    def _get_w(self, xt):
        wt = []
        for xi in xt:
            wi = self.w.get(xi, 0.)
            if abs(wi) <= self.lambda1:
                wt.append(0.)
            else:
                num = self.num.get(xi, 0.)
                eta = self.alpha / (self.beta + np.sqrt(num))
                temp = 1 / eta + self.lambda2
                wi = (np.sign(wi) * self.lambda1 - wi) / temp
                wt.append(wi)
        return wt

    def _get_grads(self, yt, pt, wt):
        # Get the gradient as a function of true value and predicted
        # probability. Regularization does not apply here since it is
        # realized differently for FTRLprox, but 'wt' is still passed
        # for consistency.
        y_err = pt - yt
        return y_err

    def _update(self, wt, xt, grad, sample_weight):
        grad_sq = grad * grad
        numt = self._get_num(xt)
        delta_num = self._get_delta_num(grad)
        for xi, wi, numi in zip(xt, wt, numt):
            new_num = numi + delta_num
            # sigma = 1/eta(t) - 1/eta(t-1)
            sigma = (np.sqrt(new_num) - np.sqrt(numi)) / self.alpha
            delta_w = (wi * sigma - grad) / sample_weight
            self.w[xi] = self.w.get(xi, 0.) - delta_w
            self.num[xi] = new_num
        return self
