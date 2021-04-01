# -*- coding: utf-8 -*-
""" RandOm Convolutional KErnel Transform (ROCKET)
"""

__author__ = "Matthew Middlehurst"
__all__ = ["Arsenal"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.rocket import Rocket
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y


class Arsenal(BaseClassifier):
    """
    Classifier wrapped for ensemble of ROCKET transformers using RidgeClassifierCV as
    the base classifiers. Allows for generation of probabilities at the expense of
    scalability.

    Parameters
    ----------
    num_kernels             : int, number of kernels for ROCKET transform
    (default=2,000)
    ensemble_size           : int, size of the ensemble (default=25)
    n_jobs                  : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    classifiers             : array of ROCKET classifiers
    weights                 : weight of each classifier in the ensemble
    weight_sum              : sum of all weights
    n_classes               : extracted from the data

    Notes
    -----
    @article{dempster_etal_2019,
      author  = {Dempster, Angus and Petitjean, Francois and Webb,
      Geoffrey I},
      title   = {ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels},
      year    = {2019},
      journal = {arXiv:1910.13051}
    }

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/shapelet_based/Arsenal.java

    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        num_kernels=2000,
        ensemble_size=25,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classifiers = []

        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}

        super(Arsenal, self).__init__()

    def fit(self, X, y):
        """
        Build an ensemble of pipelines containing the ROCKET transformer and
        RidgeClassifierCV classifier.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        for i in range(self.ensemble_size):
            rocket_pipeline = make_pipeline(
                Rocket(
                    num_kernels=self.num_kernels,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
                RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
            )
            rocket_pipeline.fit(X, y)
            self.classifiers.append(rocket_pipeline)

        self._is_fitted = True
        return self

    def predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X)

        sums = np.zeros((X.shape[0], self.n_classes))

        for n, clf in enumerate(self.classifiers):
            preds = clf.predict(X)
            for i in range(0, X.shape[0]):
                sums[i, self.class_dictionary[preds[i]]] += 1

        return sums / (np.ones(self.n_classes) * self.ensemble_size)
