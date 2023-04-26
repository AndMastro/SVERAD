from itertools import product
import math
import numpy as np
from scipy import sparse
from scipy.special import binom
from itertools import combinations

##############my functions#####################

#TODO: change gamma with sigma
def rbf_kernel_matrix(matrix_a: np.ndarray, matrix_b: np.ndarray, gamma: float = 1.0, sigma: float = None):
    """Calculates the RBF kernel between two matrices and returns a similarity matrix.

    Parameters
    ----------
    matrix_a: np.ndarray
        matrix a
    matrix_b: np.ndarray
        matrix b
    gamma: float
        gamma parameter of the RBF kernel
    sigma: float
        sigma parameter of the RBF kernel, if not None, gamma is calculated as 1/(2*sigma**2).
        Any value given to gamma when sigma is not None is ignored.
    Returns
    -------
        np.ndarray
    """
    norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
    norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
    distance_squared = np.add.outer(norm_1, norm_2.T) - 2 * matrix_a.dot(matrix_b.transpose())
    
    if sigma is not None:
        gamma = 1 / (2 * sigma ** 2)
    return np.exp(-gamma * distance_squared)

def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0, sigma: float = None):
    euclidean_distance = np.linalg.norm(x - y, ord=2)
    
    if sigma is not None:
        gamma = 1 / (2 * sigma ** 2)
    return np.exp(-gamma * (euclidean_distance**2))



class ExactRBFShapleyComputation:
    """Calculates the Shapley value according to the Shapley formalism for the RBF kernel.
        (Enumerates all possible coalitions, exponential cost!)
    """
    def __init__(self, ref_arr, gamma: float = 1.0, sigma: float = None):
        self.ref_arr = ref_arr
        self.gamma = gamma
        if sigma is not None:
            self.gamma = 1 / (2 * sigma ** 2)

    def rbf_kernel_value(self, x):
        """Calculates the RBF kernel of an instance x with initialized vector `ref_arr`.
        This can be treated equivalent to a prediction of the RBF kernel value to ref_arr given x.
        Hence, we can explain the similarity with SHAP.
        """
        return rbf_kernel_matrix(x, self.ref_arr.reshape((1, -1)), self.gamma)

    def shapley_values(self, x):
        ref = self.ref_arr
        assert x.shape == ref.shape
        shapley_vector = []
        for f_idx, (f_x, f_ref) in enumerate(zip(x, ref)):
            # Initializing the Shapley value as 0
            shapley_value = 0
            # Removing the assessed feature from the vectors
            remaining_f_x = np.delete(x, f_idx)
            remaining_f_ref = np.delete(ref, f_idx)

            # Iterating over all coalition sizes
            for coal_size in np.arange(0, x.shape[0]+1):
                if coal_size == 0: # if empty coalition:
                    if f_x == f_ref == 0:
                        # if the feature is absent in both instances, it does not affect the RBF kernel value.
                        # shapley value += 0 , or simply skip
                        continue
                    shapley_value += (rbf_kernel(np.array([f_x]), np.array([f_ref]), gamma = self.gamma) -0) * inv_muiltinom_coeff(x.shape[0], coal_size)
                    continue
                feature_indices = np.arange(0, remaining_f_x.shape[0], dtype=int)

                #Creating all feature combinations as possible coalitions.
                for sel_features in combinations(feature_indices, r=coal_size):
                    sel_features = np.array(sel_features)
                    coal_x = remaining_f_x[sel_features]  # Coalition without assessed feature
                    coal_ref = remaining_f_ref[sel_features]  # Coalition without assessed feature
                    coal_x_fi = np.hstack([coal_x, [f_x]]) # Coalition with assessed feature
                    coal_ref_refi = np.hstack([coal_ref, [f_ref]])  # Coalition with assessed feature
                    if np.sum(coal_x_fi + coal_ref_refi) == 0:
                        # if the all features (including the assessed feature) are absent in both instances, they do not affect the Tanimoto similarity.
                        # shapley value += 0 , or simply skip
                        continue
                    if np.sum(coal_x + coal_ref) == 0: # if empty coalition (or coalition form only absent features), set subtracting term to 0
                        shapley_value += (rbf_kernel(coal_x_fi, coal_ref_refi)-0) * inv_muiltinom_coeff(x.shape[0], coal_size)
                    else:
                        shapley_value += (rbf_kernel(coal_x_fi, coal_ref_refi) - rbf_kernel(coal_x, coal_ref)) * inv_muiltinom_coeff(x.shape[0], coal_size)
            shapley_vector.append(shapley_value)
        return np.array(shapley_vector)
    

#functions to compute SV as SVERAD values
def sverad_f_plus(n_intersecting_features: int, n_difference_features: int):
    """
    Function to compute SV as SVERAD values for intersecting features. As described in the paper, this
    value only depends on the inverse multinomial coefficient. Given an intersecting
    feature contributes only when added to the empty coalition. 

    Parameters
    ----------
    n_intersecting_features: int
        toatl number of features in the intersection of the two instances
    n_difference_features:
        total number of features in the symmetric difference of the two instances
    
    Returns
    -------
        float
        SV as SVERAD value for the feature
    """
    if n_intersecting_features == 0:
        return 0 #conforming to SV formalism, absent feature do not contribute
    
    return inv_muiltinom_coeff(n_intersecting_features+n_difference_features, 0)


def sverad_f_minus(n_intersecting_features: int, n_difference_features: int, gamma:float = 1.0, sigma: float = None):
    """
    Function to compute SV as SVERAD values for symmetric difference features.
    We presente the simplified and more efficient solution as described in the paper.

    Parameters
    ----------
    n_intersecting_features: int
    n_difference_features: int
    gamma: float
        inverse of the squared sigma value.
    sigma: float
        sigma value. If defined, gamma is ignored.

    Returns
    -------
    float

    """

    if sigma is not None:
        gamma = 1 / (2 * sigma**2)

    if n_difference_features == 0:
        return 0 #conforming to SV formalism, absent feature do not contribute
    
    sverad_value = 0
    total_features = n_intersecting_features + n_difference_features

    # contribution to the empty coalition
    sverad_value += np.exp(-gamma) * inv_muiltinom_coeff(total_features, 0)

    coalition_iterator = product(range(n_intersecting_features + 1), range(n_difference_features))

    # skipping empty coaliton as this is done already
    # this could be further optimized if we confirm the simplification in the paper
    _ = next(coalition_iterator)
    for int_features, sym_diff_features in coalition_iterator:
        delta_rbf_value = rbf_obtimized(sym_diff_features+1) - rbf_obtimized(sym_diff_features)
        n_repr_coal = binom(n_intersecting_features, int_features) * binom(n_difference_features - 1, sym_diff_features)
        sverad_value += delta_rbf_value * n_repr_coal * inv_muiltinom_coeff(total_features, int_features + sym_diff_features)
    
    return sverad_value

def rbf_obtimized(n_difference_features: int, gamma:float = 1.0, sigma: float = None):
    """
    Function to compute RBF using the number of symmetric difference features.

    Parameters
    ----------
    n_intersecting_features: int
    n_difference_features: int
    gamma: float
        inverse of the squared sigma value.
    sigma: float
        sigma value. If defined, gamma is ignored.

    Returns
    -------
    float

    """
    
    if sigma is not None:
        gamma = 1 / (2 * sigma**2)

    return np.exp(-gamma*n_difference_features)
####################################################################################

def tanimoto_similarity_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    """Calculates the Tanimoto similarity between two sparse matrices and returns a similarity matrix.

    Parameters
    ----------
    matrix_a: sparse.csr_matrix
        matrix a
    matrix_b: sparse.csr_matrix
        matrix b
    Returns
    -------
        np.ndarray
    """
    intersection = matrix_a.dot(matrix_b.transpose()).toarray()
    norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
    norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
    union = norm_1 + norm_2.T - intersection
    return intersection / union


def tanimoto_similarity_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
    """Calculates the Tanimoto similarity between two dense matrices and returns a similarity matrix.

    Parameters
    ----------
    matrix_a: np.ndarray
        matrix a
    matrix_b: np.ndarray
        matrix b
    Returns
    -------
        np.ndarray
    """
    intersection = matrix_a.dot(matrix_b.transpose())
    norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
    norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
    union = np.add.outer(norm_1, norm_2.T) - intersection

    return intersection / union


def inv_muiltinom_coeff(number_of_players: int, coalition_size: int) -> float:
    """Factor to weight coalitions ins the Shapley formalism.

    Parameters
    ----------
    number_of_players: int
        total number of available players according to the Shapley formalism
    coalition_size
        number of players selected for a coalition
    Returns
    -------
        float
        weight for contribution of coalition
    """
    n_total_permutations = math.factorial(number_of_players)
    n_permutations_coalition = math.factorial(coalition_size)
    n_permutations_remaining_players = math.factorial(number_of_players - 1 - coalition_size)

    return n_permutations_remaining_players * n_permutations_coalition / n_total_permutations
