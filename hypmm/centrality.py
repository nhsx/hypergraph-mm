import time as t

import numba
import numpy as np

from hypmm import centrality_utils

###############################################################################
# 1. ADJACENCY MATRIX
###############################################################################


@numba.njit(
    nogil=True,
    fastmath=True,
)
def comp_adjacency(
    incidence_matrix,
    edge_weight,
    node_weight,
    rep="standard",
    weight_resultant=True,
):
    """
    This function computes the adjacency matrix for an undirected hypergraph
    for both the standard and dual representations, allowing the resultant
    adjacency matrix to be weighted by both edges and nodes if user chooses.

    This function calculates M * W * M^T whilst setting all the
    diagonal elements of M * W * M^T to zero.

    Parameters
    ----------
        incidence_matrix (np.array, dtype=np.uint8) : The incidence matrix.

        edge_weight (np.array, dtype=np.float64) : A vector of edge weights,
        which must have the same number of elements as the second axis of
        matrix.

        node_weight (np.array, dtype=np.float64) : A vector of node weights,
        which must have the same number of elements as the first axis of
        matrix.

        weight_resultant (bool) : This flag tells the function whether to
        include both edge and node weights in the adjacency matrix calculation.
        If this is set to True then the matrix is computed as of
        sqrt(W_a) M^T W_b M sqrt(W_a), where a and b represent either
        edge/node weight or node/edge weight matrices.
        If set to False, adjacency matrix calcultes as M^T W_b M

        rep (string) : Type of adjacency matrix to compute, if rep="standard",
        then adjacency matrix is size of first axis of matrix, i.e. of the
        nodes of the hypergraph. If rep="dual", then adjacency matix is size of
        second axis, i.e. of the number of edges in the hypergraph.
    """

    # 1) Set up hyergraph type
    if rep == "standard":
        res_weight = node_weight
        weight = edge_weight
        inc_mat = incidence_matrix
        nrows, ncols = inc_mat.shape
    else:
        res_weight = edge_weight
        weight = node_weight
        inc_mat = incidence_matrix.T
        nrows, ncols = inc_mat.shape

    # 1) M W
    weighted_incidence = np.zeros((nrows, ncols), dtype=np.float32)
    for i in range(nrows):
        for j in range(ncols):
            weighted_incidence[i, j] += inc_mat[i, j] * weight[j]

    # 2) M W M^T
    intermediate = np.zeros((nrows, nrows), dtype=np.float32)
    for i in range(nrows):
        for j in range(nrows):
            for k in range(ncols):
                intermediate[i, j] += weighted_incidence[i, k] * inc_mat.T[k, j]

    # 3) diag(M W M^T v) can be done in one step using W M^T from before
    subt = np.zeros(nrows, dtype=np.float32)
    for i in range(nrows):
        for k in range(ncols):
            subt[i] += inc_mat[i, k] * weight[k]

    # 4) Subtract degree or node/degree
    for i in range(nrows):
        intermediate[i, i] -= subt[i]

    # 5) Check if weight_resultant has been flagged
    if weight_resultant:
        result = np.zeros_like(intermediate)
        sqr_rt_weights = np.sqrt(res_weight)
        for i in range(nrows):
            for j in range(nrows):
                result[i, j] += (
                    sqr_rt_weights[i] * intermediate[i, j] * sqr_rt_weights[j]
                )
    else:
        result = intermediate

    return result


###############################################################################
# 2. EIGENVECTOR CENTRALITY FOR UNDIRECTED REPRESENTATIONS
###############################################################################


def eigenvector_centrality(
    incidence_matrix,
    edge_weight,
    node_weight,
    rep="standard",
    tolerance=1e-6,
    max_iterations=100,
    random_seed=12345,
    weight_resultant=False,
    return_error=False,
):
    """
    This function uses a Chebyshev algorithm to find the eigenvector
    and largest eigenvalue (corresponding to the eigenvector
    centrality) from the adjacency matrix calculated using the incidence
    matrix, a set of weights (for either edges/weights) and another set
    of weights (for either nodes/edges) for the potential resultant weighting.

    Depending on the representation chosen, the length of the returned
    eigenvector will be the size of the first dimension of the tail/head
    incidence matrix, i.e. for standard it will be the size of nodes and
    dual it will be the size of the hyperarcs.

    One may use this function to compute the centrality of the standard or
    the dual hypergraph of the undirected representation of the directed
    hypergraph.

    INPUTS
    --------------------
        incidence_matrix (np.array, dtype=np.uint8) : The incidence matrix.

        edge_weight (np.array, dtype=np.float64) : A vector of edge weights,
        which must have the same number of elements as the second axis of
        matrix.

        node_weight (np.array, dtype=np.float64) : A vector of node weights,
        which must have the same number of elements as the first axis of
        matrix.

        rep (string, optional) : The representation of the hypergraph for which
        to calculate the eigenvector centrality. Options are "standard" or
        "dual". (default: "standard")

        tolerance (float, optional) :The difference between iterations in the
        eigenvalue at which to assume the algorithm has converged
        (default: 1e-6).

        max_iterations (int, optional) : The maximum number of iterations (of
        the power method) to perform before terminating the algorithm and
        assuming it has not converged (default: 100)

        weight_resultant (bool) : Flag to tell function to include the node
        weights during computation of the transition matrix for the dual
        directed hypergraph.

        random_seed (int, optional) : The random seed to use in generating the
        initial vector (default: 12345)

        return_error (bool) : Flag to return eigenvector, but also eigenvalue,
        its error and the iteration of convergence.
    """
    # 1) setup up hyergraph type and initialise eigenvector
    rng = np.random.default_rng(random_seed)
    if rep == "standard":
        res_weight = node_weight
        weight = edge_weight
        inc_mat_original = incidence_matrix
        nrows, ncols = inc_mat_original.shape
    else:
        res_weight = edge_weight
        weight = node_weight
        inc_mat_original = incidence_matrix.T
        nrows, ncols = inc_mat_original.shape
    old_eigenvector_estimate = rng.random(nrows, dtype="float32")

    # 2) Normalise using L_2 norm and apply weighted resultant
    norm_factor = np.linalg.norm(old_eigenvector_estimate)
    old_eigenvector_estimate /= norm_factor
    eigenvalue_estimates, eigenvalue_error_estimates = [], []
    if weight_resultant:
        res_weight_sqroot = np.sqrt(res_weight)
        inc_mat = centrality_utils.matrix_mult(inc_mat_original, res_weight_sqroot)
    else:
        inc_mat = inc_mat_original

    # 3) do the Chebyshev
    # In principle, the body of this loop could be compiled with Numba.
    # However, _iterate_vector() is quadratic in long_axis_size, whereas
    # all the other operations here are linear in it, so we are spend very
    # little time in the rest of this loop body.
    for iteration in range(max_iterations):
        print("\rRunning iteration {}...".format(iteration), end="")

        # Iterate over vector
        new_eigenvector_estimate = centrality_utils.iterate_eigencentrality_vector(
            inc_mat, weight, old_eigenvector_estimate
        )

        # To estimate eigenvalue and error, take ratio of new to old
        # eigenvector ignoring zeros, whose mean and standard deviation
        # represent new estimate and error
        mask = (new_eigenvector_estimate != 0) & (old_eigenvector_estimate != 0)
        iter_eigenvalue_estimates = (
            new_eigenvector_estimate[mask] / old_eigenvector_estimate[mask]
        )
        eigenvalue_estimate = iter_eigenvalue_estimates.mean()
        eigenvalue_error_estimate = iter_eigenvalue_estimates.std()

        # Append new estimates for eigenvalue and error
        eigenvalue_estimates.append(eigenvalue_estimate)
        eigenvalue_error_estimates.append(eigenvalue_error_estimate)

        # Check to see if tolerance has been met
        if eigenvalue_error_estimate / eigenvalue_estimate < tolerance:
            eig_est = float(f"{eigenvalue_estimate:.3g}")
            eig_err = float(f"{eigenvalue_error_estimate:.3f}")
            print(
                (
                    f"\nConverged at largest eigenvalue {eig_est:g} "
                    f"± {eig_err:f} after {iteration} iterations"
                )
            )
            break

        # Normalise to maintain probability space and prevent overflows
        norm_factor = np.linalg.norm(new_eigenvector_estimate)
        old_eigenvector_estimate = np.array(new_eigenvector_estimate / norm_factor)

    # Break out of algorithm if maximum iterations were reached
    else:
        print(f"\nFailed to converge after {iteration} iterations.")
        eig_est = eigenvalue_estimate
        eig_err = eigenvalue_error_estimate
        print(
            (
                f"Last estimate was {float(f'{eig_est:.3g}'):g} "
                f"± {float(f'{eig_err:.3f}'):f}"
            )
        )

    # Once out of iterative eigenvalue estimator, define eigenvalue,
    # normalise eigenvector and return
    eigenvalue = eigenvalue_estimate
    norm_factor = np.linalg.norm(new_eigenvector_estimate)
    eigenvector = np.array(new_eigenvector_estimate / norm_factor)

    # Depending on if outputing error
    if return_error:
        output = (
            eigenvector,
            (eigenvalue, eigenvalue_error_estimate),
            iteration,
        )
    else:
        output = eigenvector

    return output


def eigenvector_centrality_w_adjacency(
    incidence_matrix,
    edge_weights,
    node_weights,
    rep="standard",
    tolerance=1e-6,
    max_iterations=100,
    random_seed=12345,
    weight_resultant=False,
    return_error=False,
):
    """
    This function uses a Chebyshev algorithm to find the eigenvector
    and largest eigenvalue (corresponding to the eigenvector
    centrality) from the adjacency matrix calculated using the incidence
    matrix, a set of weights (for either edges/weights) and another set
    of weights (for either nodes/edges) for the potential resultant weighting.

    Depending on the representation chosen, the length of the returned
    eigenvector will be the size of the first dimension of the tail/head
    incidence matrix, i.e. for standard it will be the size of nodes and
    dual it will be the size of the hyperarcs.

    One may use this function to compute the centrality of the standard or
    the dual hypergraph of the undirected representation of the directed
    hypergraph.

    INPUTS
    --------------------
        incidence_matrix (np.array, dtype=np.uint8) : The incidence matrix.

        edge_weight (np.array, dtype=np.float64) : A vector of edge weights,
        which must have the same number of elements as the second axis of
        matrix.

        node_weight (np.array, dtype=np.float64) : A vector of node weights,
        which must have the same number of elements as the first axis of
        matrix.

        rep (string, optional) : The representation of the hypergraph for which
        to calculate the eigenvector centrality. Options are "standard" or
        "dual". (default: "standard")

        tolerance (float, optional) :The difference between iterations in the
        eigenvalue at which to assume the algorithm has converged
        (default: 1e-6).

        max_iterations (int, optional) : The maximum number of iterations (of
        the power method) to perform before terminating the algorithm and
        assuming it has not converged (default: 100)

        weight_resultant (bool) : Flag to tell function to include the node
        weights during computation of the transition matrix for the dual
        directed hypergraph.

        random_seed (int, optional) : The random seed to use in generating the
        initial vector (default: 12345)

        return_error (bool) : Flag to return eigenvector, but also eigenvalue,
        its error and the iteration of convergence.
    """
    # 1) setup, compute adjacency matrix and initialise eigenvector
    rng = np.random.default_rng(random_seed)
    n_diseases, n_edges = incidence_matrix.shape
    adjacency = comp_adjacency(
        incidence_matrix,
        edge_weights,
        node_weights,
        rep=rep,
        weight_resultant=weight_resultant,
    )
    adjacency[adjacency < 0] = 0
    adjacency = centrality_utils.generate_irreducible_ptm(adjacency)
    n_objects = adjacency.shape[0]
    old_eigenvector_estimate = rng.random(n_objects, dtype="float32")

    # 2) Normalise using L_2 norm to ensure that initial eigenvector forms
    # a probability space under the objects (nodes/hyperarcs)
    norm_factor = np.linalg.norm(old_eigenvector_estimate)
    old_eigenvector_estimate /= norm_factor
    eigenvalue_estimates, eigenvalue_error_estimates = [], []

    # 3) do the Chebyshev
    # In principle, the body of this loop could be compiled with Numba.
    # However, _iterate_vector() is quadratic in long_axis_size, whereas
    # all the other operations here are linear in it, so we are spend very
    # little time in the rest of this loop body.
    for iteration in range(max_iterations):
        print("\rRunning iteration {}...".format(iteration), end="")

        # Iterate over vector
        new_eigenvector_estimate = centrality_utils.iterate_pagerank_vector(
            adjacency, old_eigenvector_estimate
        )

        # To estimate eigenvalue and error, take ratio of new to old
        # eigenvector ignoring zeros, whose mean and standard deviation
        # represent new estimate and error
        mask = (new_eigenvector_estimate != 0) & (old_eigenvector_estimate != 0)
        iter_eigenvalue_estimates = (
            new_eigenvector_estimate[mask] / old_eigenvector_estimate[mask]
        )
        eigenvalue_estimate = iter_eigenvalue_estimates.mean()
        eigenvalue_error_estimate = iter_eigenvalue_estimates.std()

        # Append new estimates for eigenvalue and error
        eigenvalue_estimates.append(eigenvalue_estimate)
        eigenvalue_error_estimates.append(eigenvalue_error_estimate)

        # Check to see if tolerance has been met
        if eigenvalue_error_estimate / eigenvalue_estimate < tolerance:
            print(
                (
                    "\nConverged at largest eigenvalue "
                    f"{eigenvalue_estimate:.2f} ± "
                    f"{eigenvalue_error_estimate:.4f} after {iteration} "
                    "iterations"
                )
            )
            break

        # Normalise to maintain probability space and prevent overflows
        norm_factor = np.linalg.norm(new_eigenvector_estimate)
        old_eigenvector_estimate = np.array(new_eigenvector_estimate / norm_factor)

    # Break out of algorithm if maximum iterations were reached
    else:
        print(f"\nFailed to converge after {iteration} iterations.")
        print(
            "Last estimate was {:.2f} ± {:.4f}".format(
                eigenvalue_estimate, eigenvalue_error_estimate
            )
        )

    # Once out of iterative eigenvalue estimator, define eigenvalue,
    # normalise eigenvector and return
    eigenvalue = eigenvalue_estimate
    norm_factor = np.linalg.norm(new_eigenvector_estimate)
    eigenvector = np.array(new_eigenvector_estimate / norm_factor)

    # Depending on if outputing error
    if return_error:
        output = (
            eigenvector,
            (eigenvalue, eigenvalue_error_estimate),
            iteration,
        )
    else:
        output = eigenvector

    return output


###############################################################################
# 3. SUCCESSOR AND PREDECESSOR (NODE) PROBABILITY TRANSITION MATRIX
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def comp_ptm_std(
    inc_mat_tail,
    inc_mat_head,
    weights,
    n_deg,
    e_deg=None,
    rep="successor",
    eps=0,
):
    """
    This function calculates the probability transition matrix of the
    the standard directed hypergraph which models the markov chain
    of a random walker transitioning between nodes on a directed hypergraph
    using the transition rule:

    If representation is for sucessor detection:

    A random walker may transition from node u to v if they share a hyperarc e
    where u is in its tail set and v in its head set, i.e. we follow the
    direction of the hyperarc.

    In matrix notation, this function calculates P = D-v^{-1} M_ We M+^T

    If representation is for predecessor detection:

    A random walker may transition from node u to v if they share a hyperarc e
    where u is the head set and v is in its tail set, i.e. we move in the
    opposite direction of the hyperarc.

    In matrix notation, this function calculates
    P = D+v^{-1} M+ We D-e^{-1} M-^T

    INPUTS:
    -------------------
        inc_mat_tail (np.array, dtype=np.uint8) : Incidence matrix of the tail
        nodes of each hyperarc, which has shape (n_diseases, n_edges).

        inc_mat_head (np.array, dtype=np.uint8) : Incidence matrix of the head
        nodes of each hyperarc, which has shape (n_diseases, n_edges)

        weights (np.array, dtype=np.float64) : A vector of weights, which must
        have the same number of elements as the second axis of matrix.

        n_deg (np.array, dtype=np.float64) : Node head degree which must be of
        length (n_nodes).

        e_deg (np.array, dtype=np.float64) : Edge tail degree which must be of
        length (n_edges)

        rep (str) : Either "sucessor" or "predecessor" to define which
        transition rule is used to compute transition probability matrix

        eps (float) : Small value to prevent zero division error. Only applied
        to node degrees as it is assumed edge degrees are non-zero due to
        non-zero edge weights and edges which always have nonzero tail and head
        nodes.

        incl_mort (bool) : Flag to include mortality or not. If including
        mortality the tail-node degrees of these nodes will be 0.
    """
    # Compute the probability transition matrix
    n_diseases, n_edges = inc_mat_tail.shape

    # If computing successor diseases, fill edge degree with ones and
    # use correct order of tail/head inc mats. If predecessor, swap
    # ordering of inc mats, assuming specification of node and edge
    # degrees are correct.
    if rep == "successor":
        inc_mat_back = inc_mat_tail.copy()
        inc_mat_front = inc_mat_head.copy()
        e_deg = np.ones_like(weights)
    elif rep == "predecessor":
        inc_mat_back = inc_mat_head.copy()
        inc_mat_front = inc_mat_tail.copy()

    # 1) Compute node_degree * inc_mat * edge_weights
    term1 = np.zeros((n_diseases, n_edges), dtype=np.float64)
    for i in range(n_diseases):
        coef = 1.0 / (n_deg[i] + eps)
        for j in numba.prange(n_edges):
            term1[i, j] += coef * inc_mat_back[i, j] * weights[j]

    # 2) Compute [node_degree * inc_mat(+/-) * edge_weights] * inc_mat(-/+)
    # (* edge degree)
    prob_trans_mat = np.zeros((n_diseases, n_diseases), dtype=np.float64)
    for i in range(n_diseases):
        for j in range(n_diseases):
            for k in numba.prange(n_edges):
                coef = 1.0 / e_deg[k]
                prob_trans_mat[i, j] += term1[i, k] * inc_mat_front[j, k] * coef

        # For any nodes where they were never a successor/predecessor, i.e.
        # their row is entirely 0, set the self-returning probability to 1 -
        # causing a reducible PTM
        row_sum = prob_trans_mat[i].sum()
        if row_sum == 0:
            prob_trans_mat[i, i] = 1

    return prob_trans_mat


###############################################################################
# 4. PAGERANK CENTRALITY FOR DIRECTED REPRESENTATIONS
###############################################################################


def pagerank_centrality(
    inpt,
    rep="standard",
    typ="successor",
    tolerance=1e-6,
    max_iterations=100,
    random_seed=12345,
    is_irreducible=True,
    weight_resultant=False,
    verbose=True,
    eps=0,
    return_error=False,
):
    """
    This function uses a Chebyshev algorithm to find the eigenvector
    and largest eigenvalue (corresponding to the eigenvector
    centrality) from the transition probability matrix calculated
    using the incidence matrix, a set of weights and node and edge degrees.

    Depending on the representation chosen, the length of the returned
    eigenvector will be the size of the first dimension of the tail/head
    incidence matrix, i.e. for standard it will be the size of nodes and
    dual it will be the size of the hyperarcs.

    One may use this function to compute the centrality of the standard or
    the dual directed hypergraph.

    INPUTS
    --------------------
        inpt (tuple, required) : A 4-tuple of 2-tuples containing all necessary
        arrays for computing PageRank vector. In future, this function will
        be part of a class which will provided as attributes of the class.

        rep (string, optional) : The representation of the hypergraph for which
        to calculate the eigenvector centrality. Options are "standard" or
        "dual". (default: "standard")

        typ (str) : The type of transition rule used for node transitions. If
        typ="successor" then node transition is in direction of observed
        disease progression. If typ="predecessor", then node transition is in
        inverse direction of observed disease progression.

        tolerance (float, optional) :The difference between iterations in the
        eigenvalue at which to assume the algorithm has converged
        (default: 1e-6).

        max_iterations (int, optional) : The maximum number of iterations (of
        the power method) to perform before terminating the algorithm and
        assuming it has not converged (default: 100)

        is_irreducible (bool) : Flag to tell function whether transition matrix
        will be irreducible. If True, then steady state probability vector will
        be unique. If False, the steady state probability vector will not be
        unique and so convert transition matrix to an irreducible transition
        matrix such that the steady state vector will be.

        weight_resultant (bool) : Flag to tell function to include the node
        weights during computation of the transition matrix for the dual
        directed hypergraph.

        random_seed (int, optional) : The random seed to use in generating the
        initial vector (default: 12345).

        eps (float) : Small value to prevent zero division error. Only applied
        to node degrees as it is assumed edge degrees are non-zero due to
        non-zero edge weights and edges which always have nonzero tail and head
        nodes.

        return_error (bool) : Flag to return eigenvector, but also eigenvalue,
        its error and the iteration of convergence.
    """
    # 0) setup - if NOT weight_resultant, set node_weights to None
    rng = np.random.default_rng(random_seed)
    inc_mats, weights, node_degs, edge_degs = inpt
    inc_mat_tail, inc_mat_head = inc_mats
    node_degree_tail, node_degree_head = node_degs
    edge_degree_tail = edge_degs
    edge_weights, node_weights = weights
    n_diseases, n_edges = inc_mat_tail.shape

    # Time taken to compute PTM
    if verbose:
        st = t.time()
        print(f"Computing probability transition matrix for {rep} representation")

    # 1) Select representation, i.e. standard or dual (simple/prog/reg)
    # Choose probability transition matrix via representation
    if typ == "successor":
        n_deg = node_degree_tail
    elif typ == "predecessor":
        n_deg = node_degree_head
    ptm = comp_ptm_std(
        inc_mat_tail,
        inc_mat_head,
        edge_weights,
        n_deg,
        edge_degree_tail,
        rep=typ,
        eps=eps,
    )
    old_eigenvector_estimate = rng.random(n_diseases, dtype="float64")

    # 2) Check if stochastic probability transition matrix is irreducible
    if not is_irreducible:
        ptm = centrality_utils.generate_irreducible_ptm(ptm)

    # Print out time taken for PTM calculation
    if verbose:
        en = t.time()
        print(f"Completed in {round(en-st,2)} seconds.")

    # 3) Normalise using L_1 norm to ensure that initial eigenvector forms
    # a probability space under the objects (nodes/hyperarcs)
    norm_factor = np.linalg.norm(old_eigenvector_estimate, ord=1)
    old_eigenvector_estimate /= norm_factor
    eigenvalue_estimates, eigenvalue_error_estimates = [], []

    # 4) do the Chebyshev
    # In principle, the body of this loop could be compiled with Numba.
    # However, _iterate_pagerank_vector() is quadratic in long_axis_size,
    # whereas all the other operations here are linear in it, so we are spend
    # very little time in the rest of this loop body.
    for iteration in range(max_iterations):
        if verbose:
            print("\rRunning iteration {}...".format(iteration), end="")

        # Iterate over vector
        new_eigenvector_estimate = centrality_utils.iterate_pagerank_vector(
            ptm, old_eigenvector_estimate
        )

        # To estimate eigenvalue and error, take ratio of new to old
        # eigenvector ignoring zeros, whose mean and standard deviation
        # represent new estimate and error
        mask = (new_eigenvector_estimate != 0) & (old_eigenvector_estimate != 0)
        iter_eigenvalue_estimates = (
            new_eigenvector_estimate[mask] / old_eigenvector_estimate[mask]
        )
        eigenvalue_estimate = iter_eigenvalue_estimates.mean()
        eigenvalue_error_estimate = iter_eigenvalue_estimates.std()

        # Append new estimates for eigenvalue and error
        eigenvalue_estimates.append(eigenvalue_estimate)
        eigenvalue_error_estimates.append(eigenvalue_error_estimate)

        # Check to see if tolerance has been met
        if eigenvalue_error_estimate / eigenvalue_estimate < tolerance:
            if verbose:
                print(
                    (
                        "\nConverged at largest eigenvalue "
                        f"{eigenvalue_estimate:.2f} "
                        f"± {eigenvalue_error_estimate:.4f} after "
                        f"{iteration} iterations"
                    )
                )
            break

        # Normalise to maintain probability space and prevent overflows
        norm_factor = np.linalg.norm(new_eigenvector_estimate, ord=1)
        old_eigenvector_estimate = np.array(new_eigenvector_estimate / norm_factor)

    # Break out of algorithm if maximum iterations were reached
    else:
        if verbose:
            print(f"\nFailed to converge after {iteration} iterations.")
            print(
                "Last estimate was {:.2f} ± {:.4f}".format(
                    eigenvalue_estimate, eigenvalue_error_estimate
                )
            )

    # 5) Once out of iterative eigenvalue estimator, define eigenvalue,
    # normalise eigenvector and return
    eigenvalue = eigenvalue_estimate
    norm_factor = np.linalg.norm(new_eigenvector_estimate, ord=1)
    eigenvector = np.array(new_eigenvector_estimate / norm_factor)

    # Depending on if outputing error
    if return_error:
        output = (
            eigenvector,
            (eigenvalue, eigenvalue_error_estimate),
            iteration,
        )
    else:
        output = eigenvector

    return output
