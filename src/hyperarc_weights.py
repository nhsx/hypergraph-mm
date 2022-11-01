import numba
import numpy as np

# Hyperarc weight functions
# Hyperedge Overlap


@numba.njit(fastmath=True, nogil=True)
def _comp_hyperedge_overlap(data, inds, hyperarc_prev, hyperedge_prev):
    """
    Compute the overlap the hyperarc has with the prevalence of the hyperedge,
    i.e. it is the number of individuals with the hyperarc divided by the
    number of individuals with the hyperedge.

    Inputs:
    ----------------
        data (np.array, dtype=np.int8) : Directed hypergraph flag matrix where
        rows are observed hyperarcs and columns are diseases with -1's are tail
        nodes and 1's as head nodes. Needed for compatability with Jim's
        multimorbidity hypergraph module.

        inds (np.array, dtype=np.int8) : Numpy array of indexes to represent
        hyperarc.

        hyperarc_prev (np.array, dtype=np.float64) : 2-D Prevalence array for
        hyperarcs where row entries are indexed by binary representation of
        tail nodes and columns are indexed by the number of disease
        representing the head node.

        hyperedge_prev (np.array, dtype=np.float64) : 1-D Prevalence array for
        the parent hyperedges including population sizes for single-set
        diseases.
    """
    # Determine number of diseases since inds is in Numba compatibility form
    inds = inds.astype(np.int64)
    n_diseases = inds.shape[0]  # - np.argwhere(inds == -1).shape[0]

    # Work out binary integer mappings
    bin_tail = 0
    for i in range(n_diseases - 1):
        bin_tail += 2 ** inds[i]
    head_node = inds[n_diseases - 1]
    bin_hyperedge = bin_tail + 2**head_node

    # Numerator is prevalence of child hyperarc.
    numerator = hyperarc_prev[bin_tail, head_node]

    # Denominator is prevalence of parent hyperedge
    denominator = hyperedge_prev[bin_hyperedge]

    return numerator / denominator, denominator


# Hyperarc Weight


@numba.njit(fastmath=True, nogil=True)
def _comp_hyperarc_weight(
    data, inds, hyperedge_func, hyperarc_prev, hyperedge_prev, *args
):
    """
    This weights a hyperarc by weighting the prevalence of the parent hyperedge
    it is a children of to by how prevalent the hyperarc is relative to all
    other children of the hyperedge.

    Hyperedge prevalence is computed using the pre-specified hyperedge_func
    alongside optional arguments. Currently only support use of
    undirected_functions._modified_dice_coefficient and
    undirected_functions._comp_overlap_coeff.

    Note that using undirected_functions._comp_overlap_coeff is actually
    equivalent to just using _comp_overlap_coeff() since for parent hyperedge
    H, hyperarc h and minimum single set X_i, we have
    _comp_hyperarc_weight = (|H|/|X_i|) x (|h|/|H|)
    = |h|/|X_i| = _comp_overlap_coeff.

    Perhaps a better approach is to compute it as (|H|/|X_i|) x (|h|/|X_i|) so
    it measures the overlap coefficient of the parent hyperedge relative to the
    overlap of the child hyperarc, the issue here obviously is that we now have
    the square of a much larger set in the denominator.

    INPUTS:
    ----------------
        data (np.array, dtype=np.int8) : Directed hypergraph flag matrix where
        rows are observed hyperarcs and columns are diseases with -1's are tail
        nodes and 1's as head nodes. Needed for compatability with Jim's
        multimorbidity hypergraph module.

        inds (np.array, dtype=np.int8) : Numpy array of indexes to represent
        hyperarc.

        hyperarc_prev (np.array, dtype=np.float64) : 2-D Prevalence array for
        hyperarcs where row entries are indexed by binary representation of
        tail nodes and columns are indexed by the number of disease
        representing the head node.

        hyperedge_prev (np.array, dtype=np.float64) : 1-D Prevalence array for
        the parent hyperedges including population sizes for single-set
        diseases.
    """
    # Determine weight of parent hyperedge
    hyperedge_weight, _ = hyperedge_func(data, inds, *args)

    # Prevalence of hyperarc relative to hyperedge
    hyperarc_weight, denom = _comp_hyperedge_overlap(
        data, inds, hyperarc_prev, hyperedge_prev
    )

    return hyperedge_weight * hyperarc_weight, denom
