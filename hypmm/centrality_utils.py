import numba
import numpy as np
import scipy

###############################################################################
# 1. CALCULATE EDGE AND NODE DEGREE CENTRALITY
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def degree_centrality(
    inc_mat_tail, inc_mat_head, edge_weights, node_weights=None
):
    """
    This method calculates the degree centrality for the directed hypergraph.

    In particular, this function will calculate the weighted in-degree and
    out-degree centrality for the nodes in the directed hypergraph, as well
    as the unweighted in-degree and out-degree centrality for the hyperarcs.

    One may use this function to compute the centrality of the standard or
    the dual hypergraph.

    INPUTS
    ----------
        inc_mat_tail (np.array, dtype=np.uint8) : Incidence matrix for the
        tail sets of each hyperarc. Assumed to have shape
        (n_diseases, n_edges).

        inc_mat_head (np.array, dtype=np.uint8) : Incidence matrix for the
        head nodes of each hyperarc. Assumed to have shape
        (n_diseases, n_edges).

        edge_weights (np.array, dtype=np.float64) : Hyperarc edge weights.
        Assumed to be size (n_edges).

        node_weights (np.array, dtype=np.float64) : Node edge weights. Assumed
        to be size (n_diseases).
    """
    # Initialise node degree arrays
    n_diseases, n_edges = inc_mat_tail.shape
    node_degree_tail = np.zeros(n_diseases, dtype=np.float64)
    node_degree_head = np.zeros(n_diseases, dtype=np.float64)

    # Loop over nodes and edges to build node degree arrays
    for i in range(n_diseases):
        for j in range(n_edges):
            node_degree_tail[i] += inc_mat_tail[i, j] * edge_weights[j]
            node_degree_head[i] += inc_mat_head[i, j] * edge_weights[j]

    # If node_weights not specified, left as unitary. If node weights
    # specified, split into tail and head weights, detecting if representation
    # is directed or undirected. Is undirected representation if number of
    # elements of node_weights equal to number of diseases, otherwise is
    # directed representation.
    if node_weights is None:
        node_weights_tail = np.ones(n_diseases, dtype=np.float64)
        node_weights_head = np.ones(n_diseases, dtype=np.float64)
    else:
        node_weights_tail = (
            node_weights[:n_diseases]
            if node_weights.shape[0] == 2 * n_diseases
            else node_weights.copy()
        )
        node_weights_head = (
            node_weights[n_diseases:]
            if node_weights.shape[0] == 2 * n_diseases
            else node_weights.copy()
        )
    edge_degree_tail = np.zeros(n_edges, dtype=np.float64)
    edge_degree_head = np.zeros(n_edges, dtype=np.float64)
    for j in range(n_edges):
        for i in range(n_diseases):
            edge_degree_tail[j] += inc_mat_tail[i, j] * node_weights_tail[i]
            edge_degree_head[j] += inc_mat_head[i, j] * node_weights_head[i]

    # Store output as a 2-tuple of 2-tuples
    node_degs = (node_degree_tail, node_degree_head)
    edge_degs = (edge_degree_tail, edge_degree_head)

    return node_degs, edge_degs


###############################################################################
# 2. ITERATE VECTOR-MATRIX MULTIPLICATION TO MODULARISE CENTRALITY CALCULATION
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def iterate_pagerank_vector(ptm, vector):
    """
     This function performs one Chebyshev iteration to calculate the
    largest eigenvalue and corresponding eigenvector of the probability
    transition matrix of either the standard or dual directed hypergraph
    depending on the choice of representation

    This function calculates vP where P is the probability transition matrix
    and v is some probability vector

    INPUTS:
    -------------------
        ptm (np.array, dtype=np.float64) : Probability transition matrix.

        vector (np.array, dtype=numpy.float64) : The vector to multiply the
        matrix by. Must have the same number of elements as the second axis of
        matrix.
    """
    # Compute the left multiplication of a vector v and the probability
    # transition matrix multiplid by a vector, vP.
    n_objects = vector.shape[0]
    output = np.zeros((n_objects), dtype=np.float64)
    for i in range(n_objects):
        for j in range(n_objects):
            output[i] += vector[j] * ptm[j, i]

    return output


@numba.njit(fastmath=True, nogil=True)
def matrix_mult(matrix, vector):
    """
    For matrices whose long axis is very large, compute matrix multiplication
    of matrix and vector (where the vector is actually a diagonal matrix).

    INPUTS:
    -------------------
        matrix (np.array, dtype=np.float64) : Matrix whose first axis is very
        large.

        vector (np.array, dtype=numpy.float64) : The vector to multiply the
        matrix by. Must have the same number of elements as the second axis of
        matrix.
    """
    # Compute matrix multiplication of a vector and matrix
    nrows, ncols = matrix.shape
    result = np.zeros_like(matrix, dtype=np.float64)
    for i in range(nrows):
        for j in range(ncols):
            result[i, j] += vector[i] * matrix[i, j]

    return result


@numba.njit(
    nogil=True,
    fastmath=True,
)
def iterate_eigencentrality_vector(incidence_matrix, weight, vector):
    """
    This function performs one Chebyshev iteration to calculate the
    largest eigenvalue and corresponding eigenvector of the either the
    standard or dual hypergraph depending on the orientation of the
    incidence matrix.

    This function calculates M * W * M^T * V whilst setting all the
    diagonal elements of M * W * M^T to zero.

    INPUTS:
    ----------
        incidence_matrix (np.array, dtype=numpy.uint8) : The incidence matrix

        weight (np.array, dtype=np.float64) : A vector of weights, which must
        have the same number of elements as the second axis of matrix.

        vector (np.array, dtype=np.float64) : The vector to multiply the matrix
        by. Must have the same number of elements as the second axis of matrix.

    RETURNS:
    ----------
        result (np.array, dtype=np.float64) : The result of
        matrix * weight * transpose(matrix) * vector with diagonal elements of
        matrix * weight * transpose(matrix) set to zero.
    """

    # we are calculating [M W M^T - diag(M W M^T)] v
    term_1 = np.zeros_like(vector)

    # 1) W M^T
    weighted_incidence = np.zeros_like(incidence_matrix, dtype=np.float64)
    for i in range(weighted_incidence.shape[0]):
        for j in range(weighted_incidence.shape[1]):
            weighted_incidence[i, j] += incidence_matrix[i, j] * weight[j]

    # 2) W M^T v
    intermediate = np.zeros(weighted_incidence.shape[1], dtype=np.float64)
    for k in range(weighted_incidence.shape[1]):
        for j in range(len(vector)):
            intermediate[k] += weighted_incidence[j, k] * vector[j]

    # 3) M W M^T v
    for i in range(len(vector)):
        for k in range(weighted_incidence.shape[1]):
            term_1[i] += incidence_matrix[i, k] * intermediate[k]

    # 4) diag(M W M^T v) can be done in one step using W M^T from before
    subt = np.zeros_like(vector)
    for i in range(len(vector)):
        for k in range(weighted_incidence.shape[1]):
            subt[i] += (
                incidence_matrix[i, k] * weighted_incidence[i, k] * vector[i]
            )

    # 5) subtract one from the other.
    result = np.zeros_like(vector)
    for i in range(len(vector)):
        result[i] = term_1[i] - subt[i]

    return result


###############################################################################
# 3. FOR REASOMABLY SMALL MATRICES, DEDUCE WHETHER PROBABILITY TRANSITION
# MATRIX IS IRREDUCIBLE
###############################################################################


def is_irreducible(P):
    """
    For some stochastic matrix P, check if P is irreducible, i.e. do all states
    communicate (can every state reach every other state) after a finite number
    of transitions t, P^T. Formally, this is where state i and j communicate
    (or are reachable) if for some time steps s and t

    P^s_{ij} > 0 and P^t_{ji} > 0

    One can tell if P is irreducible quickly by estimation the matrix
    exponential of P, i.e. sum_i [(1/i!) * P^i], where if every element is
    nonzero then this means every state communicates with one another at some
    time step, therefore showing that P is irreducible.

    Note that this function is only suitable for square matrices which aren't
    egregiously large. For example, the following could took 7.5 seconds to
    execute.

    A = np.random.random(size=(5000,5000))
    A_norm = np.linalg.norm(A.T, axis=0, ord=1)
    for i in range(5000):
        A[i] /= A_norm[i]

    %%timeit
    _is_irreducible(A)

    For larger matrices, it is simpler to understand the transition rule
    followed when constructing the probability transition matrix. For example,
    for the dual probability transition matrices using progressive and
    regressive transition, we already know that degree 2 hyperarcs and degree N
    hyperarcs can only transition to themselves and therefore mean these two
    transition matrices are reducible.

    INPUTS:
    ----------------------
        P (np.array, np.float64) : Probability transition matrix, such that P
        is a square matrix whose rows sum to 1.
    """
    mat_exp = scipy.linalg.expm(P)
    is_irr = np.all(mat_exp > 0)

    return is_irr


###############################################################################
# 4. GIVEN A REDUCIBLE PROBABILITY TRANSITION MATRIX, CONVERT IT TO AN
# IRREDUCIBLE ONE
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def generate_irreducible_ptm(P, alpha_ord=5):
    """
    Given an irreducible probability transition matrix, convert stochastic
    matrix into irreducible, strictly positive, stochastic matrix by applying
    the transformation

    P_irr = alpha*P + (1-alpha)/N * 1*1^T

    where N is the dimension of the square matrix P and 1 is a column vector
    of 1's, i.e. 1*1^T is a matrix of ones the same shape as the square matrix
    P. alpha is called the damping factor in Brin and Page's original work
    (the group who first defined the reducible -> irreducible formulation). The
    damping factor is used to describe a webpage surfer's probability in
    randomly jumping to any webpage. For webpages previously not reachable in
    the reducible transition matrix, their transition probability is now
    (1-alpha)/N.

    In our case, when an individual is transitioning across disease
    progressions, we don't want them to transition to a random progression, but
    in order for the maths to work out, we need to make sure out transition
    matrix is irreducible, so our demping factor is very close to 1 and is
    defined as (1 - 1/N).

    INPUTS:
    ------------------
        P (np.array, dtype=np.float64) : Reducible, non-negative,
        row-stochastic matrix.
    """
    # Assume P is reducible.
    N = P.shape[0]
    alpha = 1 - 1e-6
    damp_fact = (1.0 - alpha) / N
    ptm = np.zeros_like(P)

    # Fill elements with newly transformed, irreducible stochastic matrix
    for i in range(N):
        for j in range(N):
            ptm[i, j] += alpha * P[i, j] + damp_fact

    return ptm
