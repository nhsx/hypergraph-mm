import numpy as np
import numba
import matplotlib.pyplot as plt

from hypmmm import utils

##########################################################################################################
# 1. CALCULATE OVERLAP COEFFICIENT
##########################################################################################################


@numba.njit(fastmath=True, nogil=True)
def comp_overlap_coeff(
    data, inds, disease_cols, prev_arr, denom_arr, contribution_type
):
    """
    Compute the overlap coefficient for a hyperedge given by inds from the prevalence array.

    Denominator array is the minimum single-set population of diseases in inds.

    Prevalence array is dependent on choice of weight_system:

    If "power set", prevalence is calculated by counting individuals which have those diseases
    in inds, ignoring the observed ordering of conditions and the possibility of individuals
    having other diseases other than those in inds.

    If "exclusive", prevalence is calculated using the multimorbidity sets of individuals at the
    end of the period of analysis, so individuals only contribute to the multimorbidity set they
    were observed to have at the end of the period of analysis.

    If "progression set", prevalence is calculated such that individuals contribute prevalence
    to the hyperedge in inds if their observed disease progression contained some sequential
    ordering of diseases in inds.

    Args:
        data (np.array, dtype=np.uint8) : Binary flag matrix containing observations (rows) and
        their conditions (columns).  Not actually used but for possible compatability with Jim's
        hypergraph module.

        inds (np.array, dtype=np.int8) : Numpy array of indexes to represent hyperarc.

        disease_cols (np.array, dtype="<U24") : Numpy array of disease columns.

        prev_arr (dict, dtype=np.int64) : Numpy array of contributions from individuals to each hyperedge.

        denom_arr (np.array, dtype=np.int64) : This is used to determine the denominator for the weight.

        contribution_type (string) : If "power" then intersection is found during runtime. Otherwise, use
        prev_arr.
    """
    # Determine number of diseases
    n_diseases = inds.shape[0]
    inds = inds.astype(np.int64)

    # If using Power set contribution, numerator comes from utils.compute_integer_repr()
    if contribution_type == "power":
        numerator = utils.compute_integer_repr(data, inds, disease_cols)[-1]

    # Otherwise, use prev_arr to fetch numerator
    else:
        # Work out binary integer representing intersection of individuals with all diseases
        # in inds
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


##########################################################################################################
# 2. MODIFIED SORENSEN-DICE COEFFICIENT (POWER SET & COMPLETE SET)
##########################################################################################################


@numba.njit(fastmath=True, nogil=True)
def modified_sorensen_dice_coefficient(
    hyperedge_worklist, hyperedge_N, hyperedge_idx, prev_arr, denom_arr, typ=1
):
    """
    Power and Complete Sorensen-Dice Coefficient calculation.

    Args:
        hyperedge_worklist (np.array, dtype=np.int8) : Hyperedge worklist.

        hyperedge_N (np.array, dtype=np.int8) : Array of edge degree of hyperedge.

        hyperedge_idx (np.array, dtype=np.int64) : Array of unique integer representation of hyperedges.

        prev_arr (np.array, dtype=np.float64) : Hyperedge prevalence array for numerator.

        denom_arr (np.array, dtype=np.float64) : Prevalence used to penalise the hyperedge numerator
        prevalence.

        typ (int) : Type of Sorensen-Dice coefficient. If 1, then Complete, if 0 then Power.
    """

    # Initialise numerator and denominator contributions to weights
    N_weights, N_diseases = hyperedge_worklist.shape
    hyperedge_num = np.zeros(N_weights, dtype=np.float64)
    hyperedge_denom = np.zeros(N_weights, dtype=np.float64)

    # Loop over hyperedges in worklist
    for src_idx, src_elem in enumerate(hyperedge_worklist):
        # Extract disease indexes of hyperedge, degree and add prevalence
        # to numerator and increment denominator with same value
        # src_N_hyper_edge = hyperedge_N[src_idx]

        # src_hyper_idx is the index of hyperedge_prev
        src_hyper_idx = hyperedge_idx[src_idx]

        # prevalence of a hyperedge (from the hyperedge prev array)
        src_num_prev = prev_arr[src_hyper_idx]
        src_denom_prev = denom_arr[src_hyper_idx]

        # Incremement source hyperedge prevalence to numerator and denominator arrays
        # C(e_i) part of numerator and denominator
        hyperedge_num[src_idx] += src_num_prev
        hyperedge_denom[src_idx] += src_num_prev

        # Check out of all hyperedges, which contain the source hyperedge using
        # binary AND operation. This will always contain the source hyperedge
        # in src_in_tgt as the first element, so skip this one using [1:]

        # src_in_tgt finds the possible supersets of the hyperedge
        src_in_tgt = np.where(src_hyper_idx & hyperedge_idx == src_hyper_idx)[0][1:]

        # for each set in the super set
        for tgt_idx in src_in_tgt:
            # Work out target hyper edge unique integer and prevalence from denom_arr
            tgt_hyper_idx = hyperedge_idx[tgt_idx]
            tgt_denom_prev = denom_arr[tgt_hyper_idx]

            # Work out weighting to multiple denominator prevalence
            # tgt_N_hyper_edge = hyperedge_N[tgt_idx]
            tgt_denom_w = 1  # abs(src_N_hyper_edge - tgt_N_hyper_edge)+1

            # This adds weighted contribution of the target hyperedge to the denominator of the
            # hyperedge weight for the source hyperedge, i.e. this adds part of the upper power set
            # of the weighted linear combination of prevalences on the denominator.

            hyperedge_denom[src_idx] += (
                tgt_denom_w * tgt_denom_prev * typ
            )  # Upper power set

            # This adds weighted contribution of the source hyperedge to the denominator of the
            # hyperedge weight for the target hyperedge, i.e. this adds part of the lower power set
            # of the weighted linear combination of prevalences on the denominator.

            hyperedge_denom[tgt_idx] += tgt_denom_w * src_denom_prev  # Lower power set

    return hyperedge_num / hyperedge_denom


##########################################################################################################
# 3. HYPERARC OVERLAP WITH PARENT
##########################################################################################################


@numba.njit(fastmath=True, nogil=True)
def comp_hyperedge_overlap(inds, hyperarc_prev, hyperedge_prev, is_single_mort):
    """
    Compute the prevalence of the hyperarc relative to the prevalence of its parent hyperedge, i.e.
    it is the number of individuals with the observed disease progression as part of their
    progression set divided by the total number of individuals who have all diseases in inds,
    regardless of observer ordering.

    Args:
        inds (np.array, dtype=np.int8) : Numpy array of indexes to represent hyperarc.

        hyperarc_prev (np.array, dtype=np.float64) : 2-D Prevalence array for hyperarcs where row
        entries are indexed by binary representation of tail nodes and columns are indexed by the
        disease index representing the head node.

        hyperedge_prev (np.array, dtype=np.float64) : 1-D Prevalence array for the parent hyperedges
        including population sizes for single-set diseases.

        is_single_mort (int) : Flag to select the correct element in hyperarc_prev.
    """
    # Determine number of diseases since inds is in Numba compatibility form
    inds = np.abs(inds).astype(np.int64)
    n_diseases = inds.shape[0]

    # Work out binary integer mappings
    bin_tail = 0
    for i in range(n_diseases - 1):
        bin_tail += 2 ** inds[i]
    head_node = inds[n_diseases - 1]
    bin_hyperedge = bin_tail + 2**head_node

    # Numerator is prevalence of child hyperarc.
    hyperarc_tail = [bin_tail, -1][is_single_mort]
    numerator = hyperarc_prev[hyperarc_tail, head_node]

    # Denominator is prevalence of parent hyperedge
    denom = hyperedge_prev[bin_hyperedge]
    zero_denom = int(denom > 0)
    denominator = [1.0, denom][zero_denom]

    return numerator / denominator, denom


##########################################################################################################
# 4. CALCULATE HYPERARC WEIGHT
##########################################################################################################


@numba.njit(fastmath=True, nogil=True)
def comp_hyperarc_weight(
    inds,
    hyperarc_prev,
    hyperedge_prev,
    hyperedge_weights,
    hyperedge_indexes,
    is_single_mort,
):
    """
    This weights a hyperarc by weighting the prevalence of its parent hyperedge by how prevalent
    the hyperarc is relative to all other children of the same parent hyperedge.

    Args:
        inds (np.array, dtype=np.int8) : Numpy array of indexes to represent hyperarc.

        hyperarc_prev (np.array, dtype=np.float64) : 2-D Prevalence array for hyperarcs where row
        entries are indexed by binary representation of tail nodes and columns are indexed by the
        disease index representing the head node.

        hyperedge_prev (np.array, dtype=np.float64) : 1-D Prevalence array for the parent hyperedges
        including population sizes for single-set diseases.

        hyperedge_weights (np.array, dtype=np.float64) : 1-D weight array for all hyperedges.

        hyperedge_indexes (np.array, dtype=np.int64) : Order of hyperedges in hyperedge_weights by binary
        encoding for fast calling of parent edge weight.

        is_single_mort (int) : Flag to select the correct element in hyperarc_prev.
    """
    # Fetch weight of parent hyperedge
    bin_ind = (2 ** (np.abs(inds).astype(np.int64))).sum()
    parent_weight = hyperedge_weights[hyperedge_indexes == bin_ind][0]

    # Prevalence of hyperarc relative to hyperedge
    hyperarc_weight, denom = comp_hyperedge_overlap(
        inds, hyperarc_prev, hyperedge_prev, is_single_mort
    )

    return parent_weight * hyperarc_weight, denom


##########################################################################################################
# 5. WEIGHT UTILITY FUNCTION FOR MORTALITY
##########################################################################################################


def setup_weight_comp(
    dis_cols,
    dis_names,
    data_binmat,
    node_prev,
    hyperarc_prev,
    hyperarc_worklist,
    weight_function,
    dice_type,
    incl_mort,
    mort_type,
    n_died,
):
    """
    Setup variables for computing hyperarc weights

    Args:
        dis_cols (list) : List of disease column names.

        dis_names (list) : List of full disease names.

        data_binmat (np.array, dtype=np.uint8) : Binary array storing individuals and their condition flags.

        node_prev (np.array, dtype=np.int64) : Prevalence of each node. Must be of length at least 2*n_diseases.
        More entires imply use of mortality.

        hyperarc_prev (np.array, dtype=np.int64) : 2d-array of hyperarc prevalence. First dimension of length maximum
        hyperedges for an n_disease-hypergraph. Second dimension at least n_diseases long. Any longer implies mortality.

        hyperarc_worklist (np.array, dtype=np.int8) : Numba compatible worklist for hyperarcs.

        weight_function (numba function) : Only used to set up hyperarc weights and disease progression strings.

        dice_type (int) : Type of Modified Sorensen-Dice Coefficient (1 is Complete, 0 is Power).

        incl_mort (bool) : Flag to use mortality or not.

        mort_type (int) : Mortality set-up. None or 1.

        n_died (int) : Number of individuals who died in dataset.
    """
    # Number of diseases and crude single-set disease prevalence
    n_diseases = data_binmat.shape[1]
    n_obs = data_binmat.shape[0]
    dis_cols = dis_cols.copy()
    dis_names = dis_names.copy()
    dis_pops = data_binmat.sum(axis=0)
    # n_morts = [0, [1, 2][mort_type]][incl_mort]
    mort_type = [None, mort_type][incl_mort]
    # N_cols = n_diseases + incl_mort

    # Build string list/arrays of disease names/nodes
    dis_nodes = [dis + "_-" for dis in dis_cols] + [dis + "_+" for dis in dis_cols]
    disease_dict = {name: dis_cols[i] for i, name in enumerate(dis_names)}

    # Default palette for node weights
    palette = 2 * list(plt.get_cmap("nipy_spectral")(np.linspace(0.1, 0.9, n_diseases)))

    if mort_type == 1:
        N_hyperarcs = hyperarc_worklist.shape[0]
    else:
        N_hyperarcs = hyperarc_worklist.shape[0] + (incl_mort + 1) * n_diseases

    hyperarc_weights = np.zeros(N_hyperarcs, dtype=np.float64)
    hyperarc_progs = np.zeros(N_hyperarcs, dtype="<U512")

    # If including mortality
    if mort_type is not None:
        # mort_idxs = [
        #     [-1, n_diseases],
        #     [n_diseases, n_diseases + 1],
        # ][mort_type]
        mort_cols = [
            ["MORT"],
            ["ALIVE", "MORT"],
        ][mort_type]

        # Compute edge weights for self-edge mortality hyperarcs. These are only used when NOT using
        # Complete Dice Coefficient.
        # selfedge_weights = np.array(
        #     [
        #         sing_pop / dis_pops[i]
        #         for i, sing_pop in enumerate(hyperarc_prev[0, :n_diseases])
        #     ],
        #     np.float64,
        # )
        # mortedge_weights = np.array(
        #     [
        #         sing_pop / dis_pops[i]
        #         for i, sing_pop in enumerate(hyperarc_prev[-1, :n_diseases])
        #     ],
        #     np.float64,
        # )

    # Append mortality columns to disease columns
    dis_cols = np.concatenate(
        [np.array(dis_cols, dtype="<U24"), np.array(mort_cols, dtype="<U24")],
        axis=0,
    )
    dis_nodes = np.concatenate(
        [np.array(dis_nodes, dtype="<U24"), np.array(mort_cols, dtype="<U24")],
        axis=0,
    )

    # If Complete Set Sorensen-Dice coefficient, we can compute self-edges within formulation, so add
    # self-edges to hyperarc worklist
    if weight_function == modified_sorensen_dice_coefficient and dice_type == 1:
        hyperarc_counter = 0

        # If excluding mortality
        if mort_type is None:
            selfedge_worklist = np.array(
                [[i] + (n_diseases - 1) * [-1] for i in range(n_diseases)],
                dtype=np.int8,
            )
            hyperarc_worklist = np.concatenate(
                [selfedge_worklist, hyperarc_worklist], axis=0
            )

            # Node weights for disease nodes taking proportion of node prevalences for head- and tail- counterpart
            # for each disease
            node_weights = [
                prev / node_prev[i % n_diseases :: n_diseases].sum()
                for i, prev in enumerate(node_prev)
            ]

        elif mort_type == 1:
            # Node weights for disease nodes taking proportion of node prevalences for head- and tail- counterpart
            # for each disease
            node_weights = [
                prev / node_prev[i % n_diseases : 2 * n_diseases : n_diseases].sum()
                for i, prev in enumerate(node_prev[: 2 * n_diseases])
            ]

            # Alive and Mortality node weights computed as proportion of total individuals who either lived or died.
            palette += [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
            node_weights.append(1 - n_died / n_obs)  # node weight for alive nodes
            node_weights.append(n_died / n_obs)  # node weights for mort nodes

    else:
        # Otherwise, edge weights for self-edges are computed as the number of individuals with only Di and nothing else
        # divided by the total number of individuals that ever had disease Di, regardless of any other set of
        # diseases they may have.
        if mort_type is None:
            hyperarc_weights[:n_diseases] = np.array(
                [
                    sing_pop / dis_pops[i]
                    for i, sing_pop in enumerate(hyperarc_prev[0, :n_diseases])
                ]
            )
            hyperarc_progs[:n_diseases] = np.array(
                [f"{dis} -> {dis}" for dis in dis_cols[:n_diseases]], dtype="<U512"
            )
            hyperarc_counter = n_diseases
            hyperarc_counter = 0

            # Node weights for disease nodes taking proportion of node prevalences for head- and tail- counterpart for
            # each disease
            node_weights = [
                prev / node_prev[i % n_diseases :: n_diseases].sum()
                for i, prev in enumerate(node_prev)
            ]

        elif mort_type == 1:
            hyperarc_counter = 2 * n_diseases
            hyperarc_counter = 0

            # Node weights for disease nodes taking proportion of node prevalences for head- and tail- counterpart for
            # each disease
            node_weights = [
                prev / node_prev[i % n_diseases : 2 * n_diseases : n_diseases].sum()
                for i, prev in enumerate(node_prev[: 2 * n_diseases])
            ]

            # Alive and Mortality node weights computed as proportion of total individuals who either lived or died.
            palette += [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
            node_weights.append((1 - n_died) / n_obs)
            node_weights.append(n_died / n_obs)

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


##########################################################################################################
# 6. WRAPPER FUNCTIONS FOR COMPUTING HYPEREDGE AND HYPERARC WEIGHTS
##########################################################################################################


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
    For fast Numba computation, wrap computation of hyperedge weights in Numba-compatible functions

    Args:
        binmat (np.array, dtype=np.uint8) : Binary disease flag array for individuals.

        inpt (3-tuple) : 3-tuple of lists storing initialise hyperedge weights, hyperedge binary-integer
        encodings and hyperedge disease titles.

        worklist (np.array, dtype=np.int8) : Hyperedge worklist.

        disease_cols (np.array, dtype=<U24) : String array of disese columns.

        prev_num (np.array, dtype=np.float64) : Hyperedge prevalence array for numerator

        prev_denom (np.array, dtype=np.float64) : Prevalence used to penalise the hyperedge numerator
        prevalence.

        weight_function (numba function) : Weight function Numba compatible

        contribution_type (str) : Type of contribution system.

        counter (int) : Index counter to track whether single set diseases were already computed or not
    """
    # Loop over hyperedges in worklist
    N_hyperedges = worklist.shape[0]
    for i in range(counter, N_hyperedges):
        # Extract disease indexes of hyperedge and disease titles
        elem = worklist[i]
        hyper_edge = elem[elem != -1]

        # Apply weight function
        weight, denom = weight_function(
            binmat, hyper_edge, disease_cols, prev_num, prev_denom, contribution_type
        )

        # Append weight, index and disease titles to appropriate lists
        hyperedge_weights[counter] = weight
        counter += 1

    return hyperedge_weights


def compute_hyperarc_weights(
    hyperarc_weights,
    hyperarc_progs,
    worklist,
    disease_cols,
    hyperarc_prev,
    hyperedge_prev,
    hyperedge_weights,
    hyperedge_indexes,
    mort_type,
    counter,
):
    """
    For fast Numba computation, wrap computation of hyperarc weights in Numba-compatible functions

    Args:
        hyperarc_weights (np.array, dtype=np.float64) : Hyperarc weights.

        hyperarc_progs (np.array, dtype=string) : Hyperarc disease progression titles

        worklist (np.array, dtype=np.int8) : Hyperarc worklist.

        disease_cols (np.array, dtype=<U24) : String array of disese columns.

        hyperarc_prev (np.array, dtype=np.float64) : Hyperarc prevalence array for hyperarcs.

        hyperedge_prev (np.array, dtype=np.float64) : Hyperedge prevalence array for deducing child hyperarc overlap

        hyperedge_weights (np.array, dtype=np.float64) : Weights of parent hyperedges.

        hyperedge_indexes (np.array, dtype=np.int64) : Binary-integer encodings of parent hyperedges, ordered
        the same as hyperedge_weights.

        mort_type (int) : Mortality type. Will be an integer from 0 to 5.

        counter (int) : Counter for adding weights to array depending on if Complete Set Dice Coefficient or
        anything else.
    """
    # Given a mortality type, specify mortality column and number of diseases
    if mort_type == 0:
        mort_col = "MORT"
        # n_diseases = disease_cols.shape[0] - 1
    elif mort_type == 1:
        mort_col = "MORT"
        # n_diseases = disease_cols.shape[0] - 2

    # Loop over hyperarc worklist
    for hyperarc in worklist:
        # Extract indices of hyperarc and diseases part of the progression.
        hyperarc = hyperarc[hyperarc != -1]
        degree = hyperarc.shape[0]

        # Locate if single-set disease mortality
        single_mort = int(hyperarc[0] < 0)
        is_single_mort = 0
        if degree != 1:
            hyperarc_cols = disease_cols[hyperarc]
            tail_set = np.sort(hyperarc_cols[:-1])
            progression = ", ".join(tail_set) + " -> " + hyperarc_cols[-1]
        elif single_mort and mort_type // 2 == 0:
            hyperarc = hyperarc // 2 + 1
            disease_name = disease_cols[-1 * hyperarc[0]]
            progression = f"{disease_name} -> {mort_col}"
            is_single_mort = 1
        else:
            hyperarc_cols = disease_cols[hyperarc]  # node/disease
            # self-loops
            progression = f"{hyperarc_cols[-1]} -> {hyperarc_cols[-1]}"

        # Compute weight
        weight, denom = comp_hyperarc_weight(
            hyperarc,
            hyperarc_prev,
            hyperedge_prev,
            hyperedge_weights,
            hyperedge_indexes,
            is_single_mort,
        )

        # Add weight and disease progression title
        hyperarc_weights[counter] = weight
        hyperarc_progs[counter] = progression
        counter += 1

    return hyperarc_weights, hyperarc_progs
