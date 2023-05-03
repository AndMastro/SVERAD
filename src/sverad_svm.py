from src.sverad_kernel import rbf_kernel_matrix_sparse, sverad_f_plus, sverad_f_minus

import abc
import numpy as np
from scipy import sparse
from sklearn.svm import SVR
from sklearn.svm import SVC


def plattscaling(dist_hyperplane, a, b) -> float:
    """ Plattscaling transforms the distance to a hyperplane to probability.

    Parameters
    ----------
    dist_hyperplane: float
        distance to hyperplane
    a: float
        scaling factor a. Determined by likelihood estimations.
    b: float
        offset b. Determined by likelihood estimations.
        Sklearn implementation uses -b instead of +b. Not documented anywhere. Learned by trial and error.

    Returns
    -------
        float

    """
    return 1/(1 + np.exp(a * dist_hyperplane - b))


class ExplainingSVM(abc.ABC):
    """ Baseclass for SVC and SVR. Saves support vectors explicitly.

    """
    def __init__(self, no_player_value=0, gamma: float = 1.0, sigma: float = None):
        self._explicit_support_vectors = None
        self.calculated_shapley_values = dict()
        self.no_player_value = no_player_value
        self.gamma = gamma
        if sigma is not None:
            self.gamma = 1 / (2 * sigma ** 2)

    def vector_feature_weights(self, vector: sparse.csr_matrix) -> np.ndarray:
        """Shapley values for `vector` using SVERAD.

        Parameters
        ----------
        vector:  sparse.csr_matrix
            explained instance.

        Returns
        -------
             np.ndarray: SVs
        """
        support_vectors = self.explicit_support_vectors
        dual_coeff = self.dual_coef_  # This will be set later by child classes.
        if dual_coeff.shape[0] > 1:
            raise NotImplementedError("Only binary Models are supported")
        dual_coeff = dual_coeff.reshape(-1, 1)

        # Determining intersection and symmetric difference.

        # Repeating the explained vector to match number of support vectors
        repeated_vector = vector[np.zeros(support_vectors.shape[0]), :]
        intersecting_f = vector.multiply(support_vectors)
        only_vector = repeated_vector - intersecting_f
        only_support = support_vectors - intersecting_f

        n_shared = intersecting_f.sum(axis=1)
        n_only_v = only_vector.sum(axis=1)
        n_only_sv = only_support.sum(axis=1)

        # Asserting that: intersection + sym. diff == union
        assert np.all(repeated_vector.sum(axis=1) == n_shared + n_only_v)
        assert np.all(support_vectors.sum(axis=1) == n_shared + n_only_sv)

        # Matrix with f+ and f- counts (numbers of intersecting and symmetric difference features)
        comb = np.asarray(np.hstack([n_shared, n_only_v + n_only_sv]))
        # SVERAD values for each Support vector.
        weight_intersection = np.array([self.get_sverad_f_plus(*vec) for vec in comb], dtype=np.float64)
        weight_difference = np.array([self.get_sverad_f_minus(*vec) for vec in comb], dtype=np.float64)
        # Assigning SVERAD values to corresponding vector positions
        feature_contrib_sim = np.array(intersecting_f.toarray(), dtype=np.float64) * weight_intersection.reshape(-1, 1)
        feature_contrib_sim += np.array(only_vector.toarray(), dtype=np.float64) * weight_difference.reshape(-1, 1)
        feature_contrib_sim += np.array(only_support.toarray(), dtype=np.float64) * weight_difference.reshape(-1, 1)

        # Asserting SVs match rbf kernel values.
        sim = rbf_kernel_matrix_sparse(vector, support_vectors, gamma = self.gamma) #TODO check if correct
        assert np.all(np.isclose(sim[0], feature_contrib_sim.sum(axis=1) + self.no_player_value))

        # Multiplying SVERAD values with weight and class label (both together are called dual_coeff)
        fw = feature_contrib_sim * dual_coeff
        # Summation over all support vectors
        fw = fw.sum(axis=0)
        return fw

    def get_sverad_f_plus(self, n_f_plus, n_f_minus):
        """Saves solutions of combinations of `n_f_plus` to decrease computational time.
        """
        if (n_f_plus, n_f_minus) in self.calculated_shapley_values:
            return self.calculated_shapley_values[(n_f_plus, n_f_minus)][0] #0 is f_plus. Return if already calculated
        self.calculated_shapley_values[(n_f_plus, n_f_minus)] = (
            sv_p := sverad_f_plus(n_f_plus, n_f_minus),
            sverad_f_minus(n_f_plus, n_f_minus, gamma=self.gamma))
        return sv_p

    def get_sverad_f_minus(self, n_f_plus, n_f_minus):
        """Saves solutions of combinations of `n_f_minus` to decrease computational time."""
        if (n_f_plus, n_f_minus) in self.calculated_shapley_values:
            return self.calculated_shapley_values[(n_f_plus, n_f_minus)][1] #1 is f_minus. Return if already calculated
        self.calculated_shapley_values[(n_f_plus, n_f_minus)] = (
            sverad_f_plus(n_f_plus, n_f_minus),
            sv_m := sverad_f_minus(n_f_plus, n_f_minus,
            gamma=self.gamma))
        return sv_m

    @property
    def explicit_support_vectors(self):
        return self._explicit_support_vectors

    def feature_weights(self, x: sparse.csr_matrix):
        """SVs for a list of instances, represented by a matrix."""
        return np.vstack([self.vector_feature_weights(x[i, :]) for i in range(x.shape[0])])


class ExplainingSVR(SVR, ExplainingSVM):
    """ SVR copied form sklearn and modified

    """

    def __init__(self, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1,
                 no_player_value=0):
        SVR.__init__(self,
                     kernel=rbf_kernel_matrix_sparse, tol=tol, C=C, epsilon=epsilon, shrinking=shrinking,
                     cache_size=cache_size, verbose=verbose, max_iter=max_iter)
        ExplainingSVM.__init__(self, no_player_value=no_player_value)

    def fit(self, X, y, sample_weight=None):
        x = super().fit(X, y, sample_weight=sample_weight)
        idx = self.support_
        self._explicit_support_vectors = X[idx]
        return self

    @property
    def expected_value(self):
        return self.intercept_ + np.sum(self.dual_coef_) * self.no_player_value


class ExplainingSVC(SVC, ExplainingSVM):
    """ SVC copied form sklearn and modified

    """
    def __init__(self, C=1.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None,
                 verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None,
                 no_player_value=0):
        SVC.__init__(self, C=C, shrinking=shrinking, probability=probability, tol=tol,
                     kernel=rbf_kernel_matrix_sparse, cache_size=cache_size, class_weight=class_weight,
                     verbose=verbose, max_iter=max_iter,
                     decision_function_shape=decision_function_shape, break_ties=break_ties,
                     random_state=random_state)
        ExplainingSVM.__init__(self, no_player_value)

    def fit(self, X, y, sample_weight=None):
        x = super().fit(X, y, sample_weight=sample_weight)
        idx = self.support_
        self._explicit_support_vectors = X[idx]
        return self

    def predict_proba(self, X):
        """Native predict proba has small numerical differences. To keep everything consistent it is redefined.
        """
        dist = (rbf_kernel_matrix_sparse(X, self._explicit_support_vectors) * self.dual_coef_).sum(axis=1)
        dist += self.intercept_
        p = plattscaling(dist, self.probA_, self.probB_)
        pcls0 = 1 - p
        return np.vstack([pcls0, p]).T

    def predict_log_odds(self, X):
        dist = (rbf_kernel_matrix_sparse(X, self._explicit_support_vectors) * self.dual_coef_).sum(axis=1)
        dist += self.intercept_
        log_odds = -dist * self.probA_ + self.probB_
        return np.vstack([-log_odds, log_odds]).T

    @property
    def explicit_support_vectors(self):
        return self._explicit_support_vectors

    def vector_feature_weights(self, vector):
        fw = ExplainingSVM.vector_feature_weights(self, vector)
        return -fw * self.probA_

    @property
    def expected_value(self):
        return -(self.intercept_ + np.sum(self.dual_coef_) * self.no_player_value) * self.probA_ + self.probB_
