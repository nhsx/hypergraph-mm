from individual_reps import _create_set_union, _generate_powerset

import numba
import numpy as np

# Hyperedge weight functions
# Overlap Coefficient


@numba.njit(fastmath=True, nogil=True)
def _comp_overlap_coeff(data, inds, disease_cols, prev_arr, denom_arr):
    """
    Compute the overlap coefficient for a hyperedge given by inds from the
    prevalence array.

    Prevalence array is dependent on choice of weight_system:

    If "power set", prevalence is calculated during runtime using inds and
    splits individuals using only those diseases in inds, ignoring the
    possibility of individuals having other diseases other than those in inds.
    If "exclusive", prevalence specified prior to runtime as we use most
    granular split of individuals, as they only contribute to the final
    multimorbidity set they belong to. If "progression set", prevalence also
    specified prior to runtime and individuals are allowed to contribute
    prevalence to hyperedges which conform to their observed, ordered disease
    progression.

    Inputs:
    ----------------
        data (np.array, dtype=np.uint8) : Binary flag matrix containing
        observations (rows) and their conditions (columns. This is not used in
        this function but only required for compatability with Jim's
        multimorbidity hypergraph package.

        inds (np.array, dtype=np.int8) : Numpy array of indexes to represent
        hyperarc.

        disease_cols (np.array, dtype="<U24") : Numpy array of disease columns

        prev_arr (dict, dtype=np.int64) : Numpy array of contributions from
        individuals to each hyperedge.

        denom_arr (np.array, dtype=np.int64) : This is used to determine the
        denominator for the weight.
    """
    # Determine number of diseases
    n_diseases = inds.shape[0]
    inds = inds.astype(np.int64)

    # Work out binary integer representing intersection of individuals with all
    # diseases in inds
    bin_int = 0
    for i in range(n_diseases):
        bin_int += 2 ** inds[i]

    # Numerator and denominator for overlap coefficient
    numerator = prev_arr[bin_int]
    denominator = denom_arr[2 ** inds[0]]
    for i in range(1, n_diseases):
        if denom_arr[inds[i]] < denominator:
            denominator = denom_arr[2 ** inds[i]]

    return numerator / denominator, denominator


# Modified Sorensen-Dice Coefficient (Power Set)


@numba.njit(
    nogil=True,
    fastmath=True,
)
def _modified_dice_coefficient_pwset(
    data, inds, disease_cols, prev_arr, denom_arr
):
    """
    Modified Sorensen-Dice coefficient which computes the multi-set
    intersection relative to a weighted sum of all permuted multi-subset
    intersections, where we weigh those intersections with more diseases closer
    to inds less than those intersections with fewer diseases than inds.

    This implementation allows different weight systems to be chosen which will
    compute this coefficient differently.

    If power set, then the Sorensen-Dice coefficient is calculated by computing
    the numerator as anyone with the diseases in inds, regardless of whether
    they have other diseases. The denominator is a weighted sum of this
    intersection and all multi-subset-intersections of inds, ignoring that fact
    that people may have other diseases outwith from inds.

    If exclusive, then the Sorensen-Dice coefficient is calculated by
    computing the numerator as only those individuals with exactly those
    diseases in inds. The denominator is a weighted sum of this intersections
    and all multi-subset-intersections of inds, but these
    multi-subset-intersections only allow individuals with *only* these
    subsetted diseases from inds.

    if progression set, then the Sorensen-Dice coefficient is calculated by
    computing the numerator as those individuals whose progression set (list of
    observable disease progressions) contains the diseases in inds. The
    denominator is a weighted sum of this intersection and all
    multi-subset-intersections of inds, but these multi-subset-intersections
    only allow individuals with *only* these subsetted diseases from inds.
    Also, the multi-subset-intersections are computed so that each individual
    only contributes to one multi-set intersection, rather than their
    progression set which is how the numerator and first term of the
    denominator are calculated.

    For example, given a individual with 4 diseases: A, B, C and D. Say inds
    specified diseases A, B and D.

    In the numerator, if "power set" the individual would contribute to the
    numerator as the multi-subset {A, B, D} is part of the power set of
    {A, B, C, D}. If "exclusive", the individual would *not* contribute to the
    numerator as they only contribute to the edge where inds specifies A, B, C
    and D. If "progression set", the individual would *only* contribute to the
    numerator if we observed their disease progression such that disease C came
    *after* A, B and D or we observed diseases C and D on the same date in the
    individual.

    For the denominator, for cases "power set", "exclusive" the denominator
    uses the prevalences computed in the same way as the numerator is
    calculated to compute the weighted sum of all multi-subset-intersections.
    If weight_system="progression set", the denominator's first term (the
    numerator) is the same as described in the paragraph above, but the second
    term computes multi-subset-intersections by dividing the individuals into
    their exclusive multimorbidity sets.

    INPUTS:
    --------------------
        data (np.array, dtype=np.uint8) : Binary flag matrix with rows as
        individuals (observations) and columns as diseases.

        inds (np.array, dtype=np.int8) : Tuple of column indexes to use to
        compute integer representation with.

        disease_cols (np.array, dtype="<U24") : Array of disease names.

        prev_arr (np.array, dtype=np.int64) : This is required when
        weight_system is "exclusive" or "progression set" for ease of
        computation. It is the prevalence array, assigning contribution from
        individuals to hyperedges based on the weighting system (exclusive or
        progression set).

        denom_arr (np.array, dtype=np.int64) : This is another prevalence array
        specifically required for weight_system="progression set" to save at
        runtime. The difference between prev_arr and denom_arr is that the
        former provides population counts of multimorbidity sets for the
        numerator and the latter does the same but for the weighted sum in the
        denominator. It is also required for weight_system="exclusive" but is
        actually the same as prev_arr and again removes the need for running
        the IF statement so saves on runtime.
    """
    # Sort indexes and deduce number of diseases and fetch intersection and
    # number of groupings
    inds = np.sort(inds)
    inter_int = (2 ** inds.astype(np.int64)).sum()
    # n_diseases = inds.shape[0]
    n_all_diseases = disease_cols.shape[0]
    # narange_diseases = np.arange(n_all_diseases).astype(np.int8)
    if inds.max() > n_all_diseases - 1:
        print("Invalid choice of inds.")
        return

    # Fetch intersection
    intersection = float(prev_arr[inter_int])

    # Deduce combinations of inds to determine which prevalences to index
    node_combs = _generate_powerset(inds)
    n_nodecombs = len(node_combs)

    # Loop over number of combinations, compute integer mapping and add
    # weighted prevalence to denominator
    denominator = 0.0
    for i in range(n_nodecombs):
        node_comb = node_combs[i].astype(np.int64)
        n_nodes = len(node_comb)
        w = 1.0 / (2.0**n_nodes)  # 1.0/(1.0+n_nodes)
        node_int = 0
        for j in range(n_nodes):
            node_int += 2 ** node_comb[j]
        denom_prev = float(denom_arr[node_int])
        denominator += float(w * denom_prev)

    return (
        intersection / (intersection + denominator),
        intersection + denominator,
    )


# Modified Sorensen-Dice Coefficient (Complete Set)


@numba.njit(
    nogil=True,
    fastmath=True,
)
def _modified_dice_coefficient_comp(
    data, inds, disease_cols, prev_arr, denom_arr
):
    """
    This is a complete version of the modified sorensen-dice coefficient, where
    instead of penalising intersections based on the weighted sum of
    individuals part of each subset of diseases specified in inds, we take this
    power set into account as well as individuals with all diseases in inds and
    others.

    INPUTS:
    --------------------
        data (np.array, dtype=np.uint8) : Binary flag matrix with rows as
        individuals (observations) and columns as diseases.

        inds (np.array, dtype=np.int8) : Tuple of column indexes to use to
        compute integer representation with.

        prev_arr (np.array, dtype=np.int64) : This is required when
        weight_system is "exclusive" or "progression set" for ease of
        computation. It is the prevalence array, assigning contribution from
        individuals to hyperedges based on the weighting system (exclusive or
        progression set).

        denom_arr (np.array, dtype=np.int64) : This is another prevalence
        array specifically required for weight_system="progression set" to save
        at runtime. The difference between prev_arr and denom_arr is that the
        former provides population counts of multimorbidity sets for the
        numerator and the latter does the same but for the weighted sum in the
        denominator. It is also required for weight_system="exclusive" but is
        actually the same as prev_arr and again removes the need for running
        the IF statement so saves on runtime.

        disease_cols (np.array, dtype="<U24") : Array of disease names.
    """
    # Sort indexes and deduce number of diseases and fetch intersection and
    # number of groupings
    inds = np.sort(inds)
    inter_int = (2 ** inds.astype(np.int64)).sum()
    n_diseases = inds.shape[0]
    n_all_diseases = disease_cols.shape[0]
    if inds.max() > n_all_diseases - 1:
        print("Invalid choice of inds.")
        return

    # Fetch intersection
    intersection = float(prev_arr[inter_int])

    # Deduce power set of inds to determine which prevalences to index
    node_ps_combs = _generate_powerset(inds, full=False)

    # Deduce all sets of diseases which include all those mentioned in inds
    non_inds = np.sort(
        np.array(
            list(set(np.arange(n_all_diseases, dtype=np.int8)) - set(inds)),
            dtype=np.int8,
        )
    )
    node_cps = _generate_powerset(non_inds, full=True)
    node_cps_comb = [
        np.asarray(list(_create_set_union(inds, e)), dtype=np.int8)
        for e in node_cps
    ]

    # Combine all node sets
    node_combs = node_ps_combs + node_cps_comb
    n_nodecombs = len(node_combs)

    # Loop over number of combinations, compute integer mapping and add
    # weighted prevalence to denominator
    denominator = 0.0
    # w_lst = create_empty_list(value=0.)
    # n_lst = create_empty_list(value=0)
    for i in range(n_nodecombs):
        node_comb = node_combs[i].astype(np.int64)
        n_nodes = len(node_comb)
        w = abs(n_diseases - n_nodes)
        node_int = 0
        for j in range(n_nodes):
            node_int += 2 ** node_comb[j]
        # n_lst.append(node_int)
        # w_lst.append(w)
        denom_prev = float(denom_arr[node_int])
        denominator += float(w * denom_prev)

    return (
        intersection / (intersection + denominator),
        intersection + denominator,
    )
