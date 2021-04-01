# -*- coding: utf-8 -*-
""" Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2
"""

__author__ = "Matthew Middlehurst"
__all__ = ["HIVECOTEV2"]

from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.distance_based import ProximityForest
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.shapelet_based import ShapeletTransformClassifier, Arsenal
from sktime.utils.validation.panel import check_X_y, check_X


class HIVECOTEV2(BaseClassifier):
    """
    Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2
    as described in [?].

    An ensemble of the STC, DrCIF, TDE, PF and ARSENAL classifiers from different
    feature representations using the CAWPE structure.


    Parameters
    ----------
    verbose                 : int, level of output printed to
    the console (for information only) (default = 0)
    n_jobs                  : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    n_classes               : extracted from the data

    Notes
    -----
    TODO

    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        pf_params=None,
        stc_params=None,
        drcif_params=None,
        arsenal_params=None,
        tde_params=None,
        verbose=0,
        n_jobs=1,
        random_state=None,
    ):
        if pf_params is None:
            pf_params = {}
        if stc_params is None:
            stc_params = {"time_contract_in_mins": 60}
        if drcif_params is None:
            drcif_params = {}
        if arsenal_params is None:
            arsenal_params = {}
        if tde_params is None:
            tde_params = {}

        self.pf_params = pf_params
        self.stc_params = stc_params
        self.drcif_params = drcif_params
        self.arsenal_params = arsenal_params
        self.tde_params = tde_params

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.pf = None
        self.stc = None
        self.drcif = None
        self.arsenal = None
        self.tde = None

        self.pf_weight = 0
        self.stc_weight = 0
        self.drcif_weight = 0
        self.arsenal_weight = 0
        self.tde_weight = 0

        self.n_classes = 0
        self.classes_ = []

        super(HIVECOTEV2, self).__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y, enforce_univariate=True)

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class

        self.pf = ProximityForest(
            **self.pf_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.pf.fit(X, y)

        if self.verbose > 0:
            print("PF ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            ProximityForest(**self.pf_params, random_state=self.random_state),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self.n_jobs,
        )
        self.pf_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "PF train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("PF weight = " + str(self.pf_weight))  # noqa

        self.stc = ShapeletTransformClassifier(
            **self.stc_params,
            random_state=self.random_state,
        )
        self.stc.fit(X, y)

        if self.verbose > 0:
            print("STC ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            ShapeletTransformClassifier(
                **self.stc_params,
                random_state=self.random_state,
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self.n_jobs,
        )
        self.stc_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "STC train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("STC weight = " + str(self.stc_weight))  # noqa

        self.drcif = DrCIF(
            **self.drcif_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.drcif.fit(X, y)

        if self.verbose > 0:
            print("DrCIF ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            DrCIF(**self.drcif_params, random_state=self.random_state),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self.n_jobs,
        )
        self.drcif_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "DrCIF train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("DrCIF weight = " + str(self.tsf_weight))  # noqa

        self.arsenal = Arsenal(
            **self.arsenal_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.arsenal.fit(X, y)

        if self.verbose > 0:
            print("Arsenal ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            Arsenal(
                **self.arsenal_params,
                random_state=self.random_state,
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self.n_jobs,
        )
        self.arsenal_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "Arsenal train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("Arsenal weight = " + str(self.rise_weight))  # noqa

        self.tde = TemporalDictionaryEnsemble(
            **self.tde_params, random_state=self.random_state, n_jobs=self.n_jobs
        )
        self.tde.fit(X, y)
        train_probs = self.tde._get_train_probs(X)
        train_preds = self.tde.classes_[np.argmax(train_probs, axis=1)]
        self.tde_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "TDE (estimate included) ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("TDE weight = " + str(self.cboss_weight))  # noqa

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
        X = check_X(X, enforce_univariate=True)

        dists = np.zeros((X.shape[0], self.n_classes))

        dists = np.add(
            dists,
            self.pf.predict_proba(X) * (np.ones(self.n_classes) * self.pf_weight),
        )
        dists = np.add(
            dists,
            self.stc.predict_proba(X) * (np.ones(self.n_classes) * self.stc_weight),
        )
        dists = np.add(
            dists,
            self.drcif.predict_proba(X) * (np.ones(self.n_classes) * self.drcif_weight),
        )
        dists = np.add(
            dists,
            self.arsenal.predict_proba(X) *
            (np.ones(self.n_classes) * self.arsenal_weight),
        )
        dists = np.add(
            dists,
            self.tde.predict_proba(X) * (np.ones(self.n_classes) * self.tde_weight),
        )

        return dists / dists.sum(axis=1, keepdims=True)
