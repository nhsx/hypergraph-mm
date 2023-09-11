import matplotlib.pyplot as plt
import numba
import numpy as np

from hypmm import utils

###############################################################################
# 1. CALCULATE OVERLAP COEFFICIENT
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def comp_overlap_coeff(
    data, inds, disease_cols, prev_arr, denom_arr, contribution_type
):
    """
    Compute the overlap coefficient for a hyperedge given by inds from the
    prevalence array.

    Denominator array is the minimum single-set population of diseases in inds.

    Prevalence array is dependent on choice of weight_system:

    If "power set", prevalence is calculated by counting individuals which
    have those diseases in inds, ignoring the observed ordering of conditions
    and the possibility of individuals having other diseases other than those
    in inds.

    If "exclusive", prevalence is calculated using the multimorbidity sets of
    individuals at the end of the period of analysis, so individuals only
    contribute to the multimorbidity set they were observed to have at the end
    of the period of analysis.

    If "progression set", prevalence is calculated such that individuals
    contribute prevalence to the hyperedge in inds if their observed disease
    progression contained some sequential ordering of diseases in inds.

    Inputs:
    ----------------
        data (np.array, dtype=np.uint8) : Binary flag matrix containing
        observations (rows) and their conditions (columns).  Not actually used
        but for possible compatability with Jim's hypergraph module.

        inds (np.array, dtype=np.int8) : Numpy array of indexes to represent
        hyperarc.

        disease_cols (np.array, dtype="<U24") : Numpy array of disease columns

        prev_arr (dict, dtype=np.int64) : Numpy array of contributions from
        individuals to each hyperedge.

        denom_arr (np.array, dtype=np.int64) : This is used to determine the
        denominator for the weight.

        contribution_type (string) : If "power" then intersection is found
        during runtime. Otherwise, use prev_arr.
    """
    # Determine number of diseases
    n_diseases = inds.shape[0]
    inds = inds.astype(np.int64)

    # If using Power set  contribution, numerator comes from
    # utils.compute_integer_repr()
    if contribution_type == "power":
        numerator = utils.compute_integer_repr(data, inds, disease_cols)[-1]

    # Otherwise, use prev_arr to fetch numerator
    else:
        # Work out binary integer representing intersection of individuals
        # with all diseases in inds
        bin_int = 0
        for i in range(n_diseases):
            bin_int += 2 ** inds[i]

        # Numerator and denominator for overlap coefficient
        numerator = prev_arr[bin_int]

    denominator = denom_arr[inds[0]]
    for i in range(1, n_diseases):
        new_denom = denom_arr[inds[i]]
        if new_denom < denominator:
            denominator = new_denom

    return numerator / denominator, denominator


###############################################################################
# 2. CALCULATE MODIFIED SORENSEN-DICE COEFFICIENT (POWER SET)
###############################################################################


@numba.njit(
    nogil=True,
    fastmath=True,
)
def modified_dice_coefficient_pwset(
    data, inds, disease_cols, prev_arr, denom_arr, contribution_type
):
    """
    Computes the multi-set intersection relative to a weighted sum of all
    permuted multi-subset intersections given by the power set of the inds,
    where we weight these multi-subset intersections according to the number of
    diseases in common each subset has with those diseases in inds, i.e. the
    weight will be smaller for more diseases in common and larger for less
    diseases in common.

    Denominator array is the count of individuals according to their final
    multimorbidity set at the end of the period of analysis.

    Prevalence array is dependent on choice of weight_system:

    If "power set", prev_arr is calculated by counting individuals which have
    those diseases in inds, ignoring the observed ordering of conditions and
    the possibility of individuals having other diseases other than those in
    inds.

    If "exclusive", prev_arr is calculated using the multimorbidity sets of
    individuals at the end of the period of analysis, so individuals only
    contribute to the multimorbidity set they were observed to have at the end
    of the period of analysis.

    If "progression set", prev_arr is calculated such that individuals
    contribute prevalence to the hyperedge in inds if their observed disease
    progression contained some sequential ordering of diseases in inds,
    i.e. no other disease was observed between any of the diseases observed in
    inds.

    INPUTS:
    --------------------
        data (np.array, dtype=np.uint8) : Binary flag matrix with rows as
        individuals (observations) and columns as diseases.  Not actually used
        but for possible compatability with Jim's hypergraph module.

        inds (np.array, dtype=np.int8) : Tuple of column indexes to use to
        compute integer representation with.

        disease_cols (np.array, dtype="<U24") : Array of disease names.

        prev_arr (np.array, dtype=np.int64) : The prevalence array for the
        numerator to fetch the number of contributions the individuals have
        made to the disease set in inds. The array stores the contribution
        counts individuals made to all hyperedges in the hypergraph, based on
        the contribution system.

        denom_arr (np.array, dtype=np.int64) : This is another prevalence array
        but solely for the denominator to penalise the contribution fetched in
        prev_arr. It stores contributions from individuals based on their final
        multimorbidity set, i.e. each individual only contributes to one
        hyperedge, the one representing their final multimorbidity set at the
        end of the period of analysis.

        contribution_type (string) : If "power" then intersection is found
        during runtime. Otherwise, use prev_arr.
    """
    # Sort indexes and deduce number of diseases and fetch intersection and
    # number of groupings
    inds = np.sort(inds)
    inter_int = (2 ** inds.astype(np.int64)).sum()
    n_all_diseases = disease_cols.shape[0]
    if inds.max() > n_all_diseases - 1:
        print("Invalid choice of inds.")
        return

    # Fetch intersection - if Power set contribution, find intersection at run
    # time: If using Power set  contribution, numerator comes from
    # utils.compute_integer_repr()
    if contribution_type == "power":
        intersection = utils.compute_integer_repr(data, inds, disease_cols)[-1]

    # Otherwise, fetch from prev_arr
    else:
        intersection = float(prev_arr[inter_int])

    # Deduce combinations of inds to determine which prevalences to index
    node_combs = utils.generate_powerset(inds)
    n_nodecombs = len(node_combs)

    # Loop over number of combinations, compute integer mapping and add
    # weighted prevalence to denominator
    denom = 0.0
    for i in range(n_nodecombs):
        node_comb = node_combs[i].astype(np.int64)
        n_nodes = len(node_comb)
        w = 1  # abs(n_diseases-n_nodes)+1
        node_int = 0
        for j in range(n_nodes):
            node_int += 2 ** node_comb[j]
        denom_prev = float(denom_arr[node_int])
        denom += float(w * denom_prev)

    # Force no zerodivisionerror
    zero_denom = int(denom > 0)
    denominator = [1.0, denom][zero_denom]

    return (
        intersection / (intersection + denominator),
        intersection + denominator,
    )


###############################################################################
# 3. CALCULATE MODIFIED SORENSEN-DICE COEFFICIENT (COMPLETE SET)
###############################################################################


@numba.njit(
    nogil=True,
    fastmath=True,
)
def modified_dice_coefficient_comp(
    data, inds, disease_cols, prev_arr, denom_arr, contribution_type
):
    """
    Computes the multi-set intersection relative to a weighted sum of all
    multi-set intersections given by the power set of inds and all those
    disease sets where inds is a subset of. Our weighting scheme will weight
    each multi-set intersection in the denominator based on the number of
    diseases disjoint from those specified by inds. This weight scheme is
    symmetric so that the weight for a multi-set intersection for a disease set
    missing 2 diseases in inds is the same as the multi-set intersection for a
    disease set including inds and 2 more diseases.

    Denominator array is the count of individuals according to their final
    multimorbidity set at the end of the period of analysis.

    Prevalence array is dependent on choice of weight_system:

    If "power set", prev_arr is calculated by counting individuals which have
    those diseases in inds, ignoring the observed ordering of conditions and
    the possibility of individuals having other diseases other than those in
    inds.

    If "exclusive", prev_arr is calculated using the multimorbidity sets of
    individuals at the end of the period of analysis, so individuals only
    contribute to the multimorbidity set they were observed to have at the end
    of the period of analysis.

    If "progression set", prev_arr is calculated such that individuals
    contribute prevalence to the hyperedge in inds if their observed disease
    progression contained some sequential ordering of diseases in inds, i.e. no
    other disease was observed between any of the diseases observed in inds.

    INPUTS:
    --------------------
        data (np.array, dtype=np.uint8) : Flag matrix with rows as individuals
        (observations) and columns as diseases.  Not actually used but for
        possible compatability with Jim's hypergraph module.

        inds (np.array, dtype=np.int8) : Tuple of column indexes to use to
        compute integer representation with.

        disease_cols (np.array, dtype="<U24") : Array of disease names.

        prev_arr (np.array, dtype=np.int64) : The prevalence array for the
        numerator to fetch the number of contributions the individuals have
        made to the disease set in inds. The array stores the contribution
        counts individuals made to all hyperedges in the hypergraph, based on
        the contribution system.

        denom_arr (np.array, dtype=np.int64) : This is another prevalence array
        but solely for the denominator to penalise the contribution fetched in
        prev_arr. It stores contributions from individuals based on their final
        multimorbidity set, i.e. each individual only contributes to one
        hyperedge, the one representing their final multimorbidity set at the
        end of the period of analysis.

        contribution_type (string) : If "power" then intersection is found
        during runtime. Otherwise, use prev_arr.
    """
    # Sort indexes and deduce number of diseases and fetch intersection and
    # number of groupings
    inds = np.sort(inds)
    inter_int = (2 ** inds.astype(np.int64)).sum()
    n_all_diseases = disease_cols.shape[0]
    if inds.max() > n_all_diseases - 1:
        print("Invalid choice of inds.")
        return

    # Fetch from prev_arr if now using Power Set contribution.
    if contribution_type != "power":
        intersection = float(prev_arr[inter_int])

    # if Power set contribution, find intersection at run time, numerator
    # comes from utils.compute_integer_repr()
    else:
        intersection = utils.compute_integer_repr(data, inds, disease_cols)[-1]

    # Deduce power set of inds to determine which prevalences to index
    node_ps_combs = utils.generate_powerset(inds, full=False)

    # Deduce all sets of diseases which include all those mentioned in inds
    non_inds = np.sort(
        np.array(
            list(set(np.arange(n_all_diseases, dtype=np.int8)) - set(inds)),
            dtype=np.int8,
        )
    )
    node_cps = utils.generate_powerset(non_inds, full=True)
    node_cps_comb = [
        np.asarray(list(utils.create_set_union(inds, e)), dtype=np.int8)
        for e in node_cps
    ]

    # Combine all node sets
    node_combs = node_ps_combs + node_cps_comb

    n_nodecombs = len(node_combs)

    # Loop over number of combinations, compute integer mapping and add
    # weighted prevalence to denominator
    denominator = 0.0
    for i in range(n_nodecombs):
        node_comb = node_combs[i].astype(np.int64)
        n_nodes = len(node_comb)
        w = 1  # abs(n_diseases-n_nodes)+1
        node_int = 0
        for j in range(n_nodes):
            node_int += 2 ** node_comb[j]
        denom_prev = float(denom_arr[node_int])
        denominator += float(w * denom_prev)

    return (
        intersection / (intersection + denominator),
        intersection + denominator,
    )


###############################################################################
# 3.5. UPDATED MODIFIED SORENSEN-DICE COEFFICIENT COMBINING POWER SET &
# COMPLETE SET INTO ONE
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def modified_sorensen_dice_coefficient(
    hyperedge_worklist, hyperedge_N, hyperedge_idx, prev_arr, denom_arr, typ=1
):
    """
    For fast Numba computation, wrap computation of hyperedge weights in
    Numba-compatible functions

    INPUTS:
    ---------------------
        hyperedge_worklist (np.array, dtype=np.int8) : Hyperedge worklist.

        hyperedge_N (np.array, dtype=np.int8) : Array of edge degree of
        hyperedge

        hyperedge_idx (np.array, dtype=np.int64) : Array of unique integer
        representation of hyperedges.

        prev_arr (np.array, dtype=np.float64) : Hyperedge prevalence array for
        numerator

        denom_arr (np.array, dtype=np.float64) : Prevalence used to penalise
        the hyperedge numerator prevalence.

        counter (int) : Counter used if using Power Sorensen-Dice Coefficient.

        typ (int) : Type of Sorensen-Dice coefficient. If 1, then Complete, if
        0 then Power.
    """
    # Initialise numerator and denominator contributions to weights
    N_weights, N_diseases = hyperedge_worklist.shape
    hyperedge_num = np.zeros(N_weights, dtype=np.float64)
    hyperedge_denom = np.zeros(N_weights, dtype=np.float64)
    typ_flag = [(0, 1), (1, 1), (1, 0)][typ]

    # Loop over hyperedges in worklist
    for src_idx, src_elem in enumerate(hyperedge_worklist):
        # Extract disease indexes of hyperedge, degree and add prevalence
        # to numerator and increment denominator with same value
        src_hyper_idx = hyperedge_idx[src_idx]
        src_num_prev = prev_arr[src_hyper_idx]
        src_denom_prev = denom_arr[src_hyper_idx]

        # Incremement source hyperedge prevalence to numerator and denominator
        # arrays
        hyperedge_num[src_idx] += src_num_prev
        hyperedge_denom[src_idx] += src_num_prev

        # Check out of all hyperedges, which contain the source hyperedge using
        # binary AND operation. This will always contain the source hyperedge
        # in src_in_tgt as the first element, so skip this one using [1:]
        src_in_tgt = np.where(src_hyper_idx & hyperedge_idx == src_hyper_idx)[0][1:]
        for tgt_idx in src_in_tgt:
            # Work out target hyper edge unique integer and prevalence from
            # denom_arr
            tgt_hyper_idx = hyperedge_idx[tgt_idx]
            tgt_denom_prev = denom_arr[tgt_hyper_idx]

            # Work out weighting to multiple denominator prevalence
            tgt_denom_w = 1  # abs(src_N_hyper_edge - tgt_N_hyper_edge)+1

            hyperedge_denom[src_idx] += (
                tgt_denom_w * tgt_denom_prev * typ_flag[0]
            )  # Upper power set
            hyperedge_denom[tgt_idx] += (
                tgt_denom_w * src_denom_prev * typ_flag[1]
            )  # Lower power set
    return hyperedge_num / hyperedge_denom


###############################################################################
# 4. HYPERARC OVERLAP WITH PARENT
###############################################################################

# DEPRECATED


###############################################################################
# 5. CALCULATE HYPERARC WEIGHT
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def compute_hyperarc_weights(
    hyperarc_weights,
    hyperedge_worklist,
    hyperarc_prev,
    hyperedge_prev,
    hyperedge_weights,
    hyperarc_counter,
):
    """
    New hyperarc weight function
    """
    N_diseases = hyperedge_worklist.shape[1]
    N_hyperarcs = hyperarc_weights.shape[0]
    hyperarc_worklist = np.empty((N_hyperarcs, N_diseases), dtype=np.int8)
    for i, elem in enumerate(hyperedge_worklist):
        # Extract disease indexes of hyperedge and disease titles
        hyper_edge = elem[elem != -1]
        degree = hyper_edge.shape[0]

        # Compute hyperarc weights of hyperedge's children
        child_worklist = -1 * np.ones((degree, N_diseases), dtype=np.int8)
        if degree != 1:
            hyper_edge_sq = 2**hyper_edge
            child_prevs = np.empty(degree, dtype=np.float64)
            hyper_edge_idx = 0
            for j in range(degree):
                head = hyper_edge[j]
                tail = np.sum(hyper_edge_sq[:j]) + np.sum(hyper_edge_sq[j + 1 :])

                child_worklist[j, degree - 1] = head
                child_worklist[j, : degree - 1] = np.array(
                    list(
                        set(hyper_edge).difference(set(np.array([head], dtype=np.int8)))
                    ),
                    dtype=np.int8,
                )

                ch_prev = hyperarc_prev[tail, head]
                child_prevs[j] = ch_prev

                hyper_edge_idx += hyper_edge_sq[j]

            # Compute hyperarc weights
            child_weights = hyperedge_weights[i] * (
                child_prevs / hyperedge_prev[hyper_edge_idx]
            )
            child_worklist = child_worklist[child_weights > 0]
            child_weights = child_weights[child_weights > 0]
        else:
            sin_idx = hyper_edge[0]
            child_worklist[0, 0] = sin_idx
            child_prev = np.array([hyperarc_prev[0, sin_idx]], dtype=np.float64)
            child_weights = hyperedge_weights[i] * (
                child_prev / hyperedge_prev[2**sin_idx]
            )

        # Append hyperarc weights, child worklist and increment counters
        hyperarc_weights[
            hyperarc_counter : hyperarc_counter + child_weights.shape[0]
        ] = child_weights
        hyperarc_worklist[
            hyperarc_counter : hyperarc_counter + child_weights.shape[0]
        ] = child_worklist
        hyperarc_counter += child_weights.shape[0]

    return hyperarc_weights, hyperarc_worklist


###############################################################################
# 6. WEIGHT UTILITY FUNCTION FOR MORTALITY
###############################################################################


def setup_weight_comp(
    dis_cols,
    dis_names,
    data_binmat,
    node_prev,
    hyperarc_prev,
    hyperarc_worklist,
    weight_function,
    dice_type,
):
    """
    Setup variables for computing hyperarc weights

    INPUTS:
    -------------------------
        dis_cols (list) : List of disease column names.

        dis_names (list) : List of full disease names.

        data_binmat (np.array, dtype=np.uint8) : Binary array storing
        individuals and their condition flags.

        node_prev (np.array, dtype=np.int64) : Prevalence of each node. Must be
        of length at least 2*n_diseases. More entires imply use of mortality.

        hyperarc_prev (np.array, dtype=np.int64) : 2d-array of hyperarc
        prevalence. First dimension of length maximum hyperedges for an
        n_disease-hypergraph. Second dimension at least n_diseases long. Any
        longer implies mortality.

        hyperarc_worklist (np.array, dtype=np.int8) : Numba compatible worklist
        for hyperarcs.

        weight_function (numba function) : Only used to set up hyperarc weights
        and disease progression strings.

        dice_type (int) : Type of Modified Sorensen-Dice Coefficient (1 is
        Complete, 0 is Power).
    """
    # Number of diseases and crude single-set disease prevalence
    n_diseases = data_binmat.shape[1]
    dis_cols = dis_cols.copy()
    dis_names = dis_names.copy()
    dis_pops = data_binmat.sum(axis=0)

    # Build string list/arrays of disease names/nodes
    dis_nodes = [dis + "_-" for dis in dis_cols] + [dis + "_+" for dis in dis_cols]
    disease_dict = {name: dis_cols[i] for i, name in enumerate(dis_names)}

    # Default palette for node weights
    # palette = 2*list(iter(sns.color_palette(
    # palette="bright",
    # n_colors=n_diseases
    # )))
    palette = 2 * list(plt.get_cmap("nipy_spectral")(np.linspace(0.1, 0.9, n_diseases)))

    # If Complete Set Sorensen-Dice coefficient, we can compute self-edges
    # within formulation, so add self-edges to hyperarc worklist
    N_hyperarcs = hyperarc_worklist.shape[0] + n_diseases
    hyperarc_weights = np.zeros(N_hyperarcs, dtype=np.float64)
    hyperarc_progs = np.zeros(N_hyperarcs, dtype="<U512")
    if weight_function == modified_dice_coefficient_comp or (
        weight_function == modified_sorensen_dice_coefficient and dice_type == 1
    ):
        self_edge_worklist = np.array(
            [[i] + (n_diseases - 1) * [-1] for i in range(n_diseases)],
            dtype=np.int8,
        )
        hyperarc_worklist = np.concatenate(
            [self_edge_worklist, hyperarc_worklist], axis=0
        )
        hyperarc_counter = 0

    # Otherwise, edge weights for self-edges are coputed as the number of
    # individuals with only Di and nothing else divided by the total number of
    # individuals that ever had disease Di, regardless of any other set of
    # diseases they may have.
    else:
        hyperarc_weights[:n_diseases] = np.array(
            [
                sing_pop / dis_pops[i]
                for i, sing_pop in enumerate(hyperarc_prev[0, :n_diseases])
            ]
        )
        hyperarc_progs[:n_diseases] = np.array(
            [f"{dis} -> {dis}" for dis in dis_cols[:n_diseases]], dtype=object
        )
        hyperarc_counter = 0

    # Node weights for disease nodes taking proportion of node prevalences for
    # head- and tail- counterpart for each disease
    node_weights = [
        prev / node_prev[i % n_diseases :: n_diseases].sum()
        for i, prev in enumerate(node_prev)
    ]

    # Convert string lists to arrays and collect output
    disease_colarr = np.array(dis_cols, dtype="<U24")
    disease_nodes = np.array(dis_nodes, dtype="<U24")
    string_outputs = (disease_colarr, disease_nodes, disease_dict)
    hyperarc_output = (
        hyperarc_progs,
        hyperarc_weights,
        hyperarc_worklist,
        hyperarc_counter,
    )

    return string_outputs, hyperarc_output, node_weights, palette


###############################################################################
# 7. WRAPPER FUNCTIONS FOR COMPUTING HYPEREDGE AND HYPERARC WEIGHTS
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def compute_hyperedge_weights(
    binmat,
    hyperedge_weights,
    worklist,
    disease_cols,
    prev_num,
    prev_denom,
    weight_function,
    contribution_type,
    counter,
):
    """
    For fast Numba computation, wrap computation of hyperedge weights in
    Numba-compatible functions

    INPUTS:
    ---------------------
        binmat (np.array, dtype=np.uint8) : Binary disease flag array for
        individuals.

        hyperedge_weights (3-tuple) : 3-tuple of lists storing initialise hyperedge
        weights, hyperedge binary-integer encodings and hyperedge disease titles.

        worklist (np.array, dtype=np.int8) : Hyperedge worklist.

        disease_cols (np.array, dtype=<U24) : String array of disese columns.

        prev_num (np.array, dtype=np.float64) : Hyperedge prevalence array for
        numerator

        prev_denom (np.array, dtype=np.float64) : Prevalence used to penalise
        the hyperedge numerator prevalence.

        weight_function (numba function) : Weight function Numba compatible

        contribution_type (str) : Type of contribution system.

        counter (int) : Index counter to track whether single set diseases were
        already computed or not
    """
    # Loop over hyperedges in worklist
    N_hyperedges = worklist.shape[0]
    for i in range(counter, N_hyperedges):
        # Extract disease indexes of hyperedge and disease titles
        elem = worklist[i]
        hyper_edge = elem[elem != -1]

        # Apply weight function
        weight, denom = weight_function(
            binmat,
            hyper_edge,
            disease_cols,
            prev_num,
            prev_denom,
            contribution_type,
        )

        # Append weight, index and disease titles to appropriate lists
        hyperedge_weights[counter] = weight
        counter += 1

    return hyperedge_weights
