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
    print(distance_squared)
    if sigma is not None:
        gamma = 1 / (2 * sigma ** 2)
    return np.exp(-gamma * distance_squared)

def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0, sigma: float = None):
    euclidean_distance = np.linalg.norm(x - y, ord=2)
    print(euclidean_distance**2)
    if sigma is not None:
        gamma = 1 / (2 * sigma ** 2)
    return np.exp(-gamma * (euclidean_distance**2))



class ExactRBFShapleyComputation:
    """Calculates the Shapley value according to the Shapley formalism for the RBF kernel.
        (Enumerates all possible coalitions, exponential cost!)
    """
    def __init__(self, ref_arr):
        self.ref_arr = ref_arr
    def rbf_kernel_value(self, x):
        """Calculates the similarity of an instance x to the initialized vector `ref_arr`.
        This can be treated equivalent to a prediction of the similarity to ref_arr given x.
        Hence, we can explain the similarity with SHAP.
        """
        return rbf_kernel_matrix(x, self.ref_arr.reshape((1, -1)))

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
                    shapley_value += ((f_x*f_ref) -0) * inv_muiltinom_coeff(x.shape[0], coal_size)
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


def sveta_f_plus(n_intersecting_features: int, n_difference_features: int, no_player_value: float = 0):
    """

    Parameters
    ----------
    n_intersecting_features
    n_difference_features
    no_player_value: float
        value of an empty coalition. Should be always zero. Likely to be removed later.

    Returns
    -------

    """
    if n_intersecting_features == 0:
        return 0

    shap_sum = np.float64(0)
    total_features = n_intersecting_features + n_difference_features
    # sampling contribution to empty coalition
    # Tanimoto of emtpy would cause an error (0/0) so it is manually set to the value of no_player_value
    shap_sum += (1 - no_player_value) * inv_muiltinom_coeff(total_features, 0)

    coalition_iterator = product(range(n_intersecting_features), range(n_difference_features + 1))

    # skipping empty coaliton as this is done already
    _ = next(coalition_iterator)
    for coal_present, coal_absent in coalition_iterator:
        coal_size = coal_absent + coal_present
        d_tanimoto = coal_absent / (coal_size * coal_size + coal_size)
        n_repr_coal = binom(n_difference_features, coal_absent) * binom(n_intersecting_features - 1, coal_present)
        shap_sum += d_tanimoto * inv_muiltinom_coeff(total_features, coal_size) * n_repr_coal
    return shap_sum


def sveta_f_minus(n_intersecting_features: int, n_difference_features: int, no_player_value: int = 0):
    """

    Parameters
    ----------
    n_intersecting_features: int
    n_difference_features: int
    no_player_value:float
        value of an empty coalition. Should be always zero. Likely to be removed later.

    Returns
    -------
    float

    """
    if n_difference_features == 0:
        return 0
    shap_sum = 0
    total_features = n_intersecting_features + n_difference_features

    # sampling contribution to empty coalition
    # Tanimoto of emtpy would cause an error (0/0) so it is manually set to the value of no_player_value
    shap_sum += (0 - no_player_value) * inv_muiltinom_coeff(total_features, 0)
    coalition_iterator = product(range(n_intersecting_features + 1), range(n_difference_features))

    n_comb = math.factorial(total_features)
    # skipping empty coaliton as this is done already
    _ = next(coalition_iterator)
    for coal_present, coal_absent in coalition_iterator:
        coal_size = coal_absent + coal_present
        d_tanimoto = -coal_present / (coal_size * coal_size + coal_size)
        n_repr_coal = binom(n_difference_features - 1, coal_absent) * binom(n_intersecting_features, coal_present)
        shap_sum += d_tanimoto * inv_muiltinom_coeff(total_features, coal_present + coal_absent) * n_repr_coal
    return shap_sum
