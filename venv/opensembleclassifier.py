# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:23:17 2020

@author: hany
"""

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
import numpy as np

class OpEnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Soft Voting/Majority Rule classifier for scikit-learn estimators.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
        A list of classifiers.
        Invoking the `fit` method on the `VotingClassifier` will fit clones
        of those original classifiers that will
        be stored in the class attribute
        `self.clfs_` if `refit=True` (default).
    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probalities, which is recommended for
        an ensemble of well-calibrated classifiers.
    weights : array-like, shape = [n_classifiers], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurances of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the clf being fitted
        - `verbose=2`: Prints info about the parameters of the clf being fitted
        - `verbose>2`: Changes `verbose` param of the underlying clf to
           self.verbose - 2
    refit : bool (default: True)
        Refits classifiers in `clfs` if True; uses references to the `clfs`,
        otherwise (assumes that the classifiers were already fit).
        Note: refit=False is incompatible to mist scikit-learn wrappers!
        For instance, if any form of cross-validation is performed
        this would require the re-fitting classifiers to training folds, which
        would raise a NotFitterError if refit=False.
        (New in mlxtend v0.6.)

    Attributes
    ----------
    classes_ : array-like, shape = [n_predictions]
    clf : array-like, shape = [n_predictions]
        The unmodified input classifiers
    clf_ : array-like, shape = [n_predictions]
        Fitted clones of the input classifiers

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlxtend.sklearn import EnsembleVoteClassifier
    >>> clf1 = LogisticRegression(random_seed=1)
    >>> clf2 = RandomForestClassifier(random_seed=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ... voting='hard', verbose=1)
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>

    For more usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
    """
    def __init__(self, clfs, weights=None, verbose=0, refit=False):

        self.clfs = clfs
        self.weights = weights
        self.verbose = verbose
        self.refit = refit

    def fit(self, X, y, sample_weight=None):
        """Learn weight coefficients from training data for each classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights passed as sample_weights to each regressor
            in the regressors list as well as the meta_regressor.
            Raises error if some regressor does not support
            sample_weight in the fit() method.

        Returns
        -------
        self : object

        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')


        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        if not self.refit:
            self.clfs_ = [clf for clf in self.clfs]

        else:
            self.clfs_ = [clone(clf) for clf in self.clfs]

            if self.verbose > 0:
                print("Fitting %d classifiers..." % (len(self.clfs)))

            for clf in self.clfs_:

                if sample_weight is None:
                    clf.fit(X, y)
                else:
                    clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.

        """
        if not hasattr(self, 'clfs_'):
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        else:  # 'hard' voting
            predictions = self._predict(X)

        return predictions

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        if not hasattr(self, 'clfs_'):
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """ Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'` : array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'` : array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.

        """
        return self._predict_probas(X)


    def get_params(self,deep=False):
        """Return estimator parameter names for GridSearch support."""
        return self.clfs_[0].get_params() 


    def _predict(self, X):
        """Collect results from clf.predict calls."""

        return np.asarray([clf.predict(X) for clf in self.clfs_]).T


    def _predict_probas(self, X):
        """Collect results from clf.predict_proba calls."""
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])
