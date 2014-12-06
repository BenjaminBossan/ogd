# FTRL proximal

Right now, this package includes Online Gradient Descent Logistic
Regression (OGDLR) and Follow the (proximal) Regularized Leader
(FTRLprox) algorithms.

Some features:

* The methods work with extremely sparse data (all treated
  categorically) by using dictionary for storage or hashing
  trick. This allows to train very sparse feature sets without
  exhausting memory.
* The interface is similar to that of scikit learn. 
* Cross-validation is made on the fly.

All this is work in progress, use at your own risk.

For paper, see (here)[http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf].
