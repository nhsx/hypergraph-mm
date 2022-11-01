from individual_reps import _compute_integer_repr

import numba
import numpy as np
import seaborn as sns

# Utility functions for computing edge weights
# Set up variables for computing hyperarc weights


def _setup_weight_comp(
    dis_cols,
    dis_names,
    data_binmat,
    node_prev,
    hyperarc_prev,
    incl_mort,
    mort_type,
    n_died,
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

        incl_mort (bool) : Flag to use mortality or not.

        mort_type (int) : Type of mortality setup.

        n_died (int) : Number of individuals who died in dataset.
    """
    # Number of diseases and crude single-set disease prevalence
    n_diseases = data_binmat.shape[1]
    n_obs = data_binmat.shape[0]
    dis_cols = dis_cols.copy()
    dis_names = dis_names.copy()
    dis_pops = data_binmat.sum(axis=0)

    # Build string list/arrays of disease names/nodes
    dis_nodes = [dis + "_-" for dis in dis_cols] + [
        dis + "_+" for dis in dis_cols
    ]
    disease_dict = {name: dis_cols[i] for i, name in enumerate(dis_names)}

    # Default palette for node weights
    palette = 2 * list(
        iter(sns.color_palette(palette="bright", n_colors=n_diseases))
    )

    # mort_arr only includes -1 if we don't include mortality.
    if incl_mort:

        # If we include mortality, these edge weights are for self-edges,
        # Di -> Di_ALIVE AND Di -> Di_MORT Computed as proportion of total #
        # individuals ever observed to have Di where they only had Di
        # throughout the period of analysis and either died (Di -> Di_MORT) or
        # was still alive (Di -> Di_ALIVE)
        edge_weights = [
            sing_pop / dis_pops[i]
            for i, sing_pop in enumerate(hyperarc_prev[0, :n_diseases])
        ] + [
            sing_pop / dis_pops[i]
            for i, sing_pop in enumerate(hyperarc_prev[-1, :n_diseases])
        ]

        # Simple case with one dead node and one alive node
        if mort_type == 0:
            # Colour palette for node weights
            palette += [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]

            # Disease columns and self-edges, i.e.
            # Di -> ALIVE AND Di -> MORTALITY
            mort_cols = ["ALIVE", "MORTALITY"]
            disease = [f"{dis} -> ALIVE" for dis in dis_cols[:n_diseases]] + [
                f"{dis} -> MORTALITY"
                for i, dis in enumerate(dis_cols[:n_diseases])
            ]

            # Alive and Mortality node weights computed as proportion of total
            # individuals who either lived or died.
            node_weights = [
                prev
                / node_prev[i % n_diseases : 2 * n_diseases : n_diseases].sum()
                for i, prev in enumerate(node_prev[: 2 * n_diseases])
            ]
            node_weights.append(node_prev[-2] / node_prev[-2:].sum())
            node_weights.append(node_prev[-1] / node_prev[-2:].sum())

        # Comprehensive case with a dead node and an alive node per disease
        elif mort_type == 1:
            # Colour palette for node weights
            palette += list(
                iter(sns.color_palette(palette="pastel", n_colors=n_diseases))
            ) + list(
                iter(sns.color_palette(palette="dark", n_colors=n_diseases))
            )

            # Disease columns and self-edges,
            # i.e. Di -> Di_ALIVE AND Di -> Di_MORT
            mort_cols = [dis + "_ALIVE" for dis in dis_cols] + [
                dis + "_MORT" for dis in dis_cols
            ]
            disease = [
                f"{dis} -> {mort_cols[i]}"
                for i, dis in enumerate(dis_cols[:n_diseases])
            ] + [
                f"{dis} -> {mort_cols[n_diseases+i]}"
                for i, dis in enumerate(dis_cols[:n_diseases])
            ]
            # Alive and Mortality node weights are proportion of total
            # alive/mortality prevalence per disease
            node_weights = [
                prev / node_prev[i % n_diseases :: n_diseases].sum()
                for i, prev in enumerate(node_prev)
            ]

        # Mortality node per disease and self-edges to represent ALIVE by end
        # of PoA
        elif mort_type == 2:
            # Colour palette for node weights
            palette += list(
                iter(sns.color_palette(palette="dark", n_colors=n_diseases))
            )

            # Disease columns and self-edges, i.e. Di -> Di AND Di -> Di_MORT
            mort_cols = [dis + "_MORT" for dis in dis_cols]
            disease = [f"{dis} -> {dis}" for dis in dis_cols[:n_diseases]] + [
                f"{dis} -> {mort_cols[i]}"
                for i, dis in enumerate(dis_cols[:n_diseases])
            ]

            # For disease nodes and mortality nodes, each disease node is split
            # into tail-, head- or dead-components and therefore, the node
            # weight for Di_tail = Di_tail / (Di_tail + Di_head + Di_mort)
            node_weights = [
                prev / node_prev[i % n_diseases :: n_diseases].sum()
                for i, prev in enumerate(node_prev)
            ]

        # Single death node and self-edges to represent ALIVE by end of PoA
        elif mort_type == 3:
            # Colour palette for node weights
            palette += [(1.0, 0.0, 0.0)]

            # Disease columns and self-edges, i.e. Di -> Di AND Di -> MORTALITY
            mort_cols = ["MORTALITY"]
            disease = [f"{dis} -> {dis}" for dis in dis_cols[:n_diseases]] + [
                f"{dis} -> MORTALITY"
                for i, dis in enumerate(dis_cols[:n_diseases])
            ]

            # Need to be implemented...
            node_weights = [
                prev / node_prev[i % n_diseases :: n_diseases].sum()
                for i, prev in enumerate(node_prev)
            ]

        # Single alive node and a death node per diseases
        elif mort_type == 4:
            # Colour palette for node weights
            palette += [(0.0, 0.0, 1.0)] + list(
                iter(sns.color_palette(palette="dark", n_colors=n_diseases))
            )

            # Disease columns and self-edges,
            # i.e. Di -> ALIVE AND Di -> Di_MORT
            mort_cols = ["ALIVE"] + [dis + "_MORT" for dis in dis_cols]
            disease = [f"{dis} -> ALIVE" for dis in dis_cols[:n_diseases]] + [
                f"{dis} -> {mort_cols[i+1]}"
                for i, dis in enumerate(dis_cols[:n_diseases])
            ]

            # For disease nodes and mortality nodes, each disease node is split
            # into tail-, head- or dead-components and therefore, the node
            # weight for Di_tail = Di_tail / (Di_tail + Di_head + Di_mort)
            noalive_prev = np.concatenate(
                [node_prev[: 2 * n_diseases], node_prev[2 * n_diseases + 1 :]],
                axis=0,
            )
            node_weights = [
                prev / noalive_prev[i % n_diseases :: n_diseases].sum()
                for i, prev in enumerate(noalive_prev[: 2 * n_diseases])
            ]

            # Alive node weight is proportion of individuals alive by the end
            # of PoA out of all individuals
            node_weights.append(n_died / n_obs)

            # To preserve ordering, add the final elements for the mortality
            # death nodes
            node_weights += [
                prev / noalive_prev[i % n_diseases :: n_diseases].sum()
                for i, prev in enumerate(noalive_prev[2 * n_diseases :])
            ]

        # Death node per disease and 0 alive nodes
        elif mort_type == 5:
            # Colour palette for node weights
            palette += list(
                iter(sns.color_palette(palette="dark", n_colors=n_diseases))
            )

            # Disease columns and self-edges, i.e. Di -> Di AND Di -> Di_MORT
            mort_cols = [dis + "_MORT" for dis in dis_cols]
            disease = [f"{dis} -> {dis}" for dis in dis_cols[:n_diseases]] + [
                f"{dis} -> {mort_cols[i]}"
                for i, dis in enumerate(dis_cols[:n_diseases])
            ]

            # Each disease node is split into tail-, head- or dead-components
            # and therefore, the node weight for
            # Di_tail = Di_tail / (Di_tail + Di_head + Di_mort)
            node_weights = [
                prev / node_prev[i % n_diseases :: n_diseases].sum()
                for i, prev in enumerate(node_prev)
            ]

        # 1 death node, 0 alive node
        if mort_type == 6:
            # Colour palette for node weights
            palette += [(1.0, 0.0, 0.0)]

            # Disease columns and self-edges,
            # i.e. Di -> ALIVE AND Di -> MORTALITY
            mort_cols = ["MORTALITY"]
            disease = [f"{dis} -> {dis}" for dis in dis_cols[:n_diseases]] + [
                f"{dis} -> MORTALITY"
                for i, dis in enumerate(dis_cols[:n_diseases])
            ]

            # Alive and Mortality node weights computed as proportion of total
            # individuals who either lived or died.
            node_weights = [
                prev
                / node_prev[i % n_diseases : 2 * n_diseases : n_diseases].sum()
                for i, prev in enumerate(node_prev[: 2 * n_diseases])
            ]
            node_weights.append(n_died / n_obs)

        # Append mortality columns to disease columns
        dis_cols += mort_cols
        dis_nodes += mort_cols

    else:
        # Edge weights for self-loops are computed as the number of individuals
        # with only Di and nothing else divided by the total number of
        # individuals that ever had disease Di, regardless of any other set of
        # diseases they may have.
        edge_weights = [
            sing_pop / dis_pops[i]
            for i, sing_pop in enumerate(hyperarc_prev[0, :n_diseases])
        ]

        # Regular node weights for tail-component and head-component of disease
        # nodes. Weight for Di_tail computed as proportion individuals who had
        # Di in their final multimorbidity set (at the end of PoA) but where Di
        # was in the tail set. Same procedure for Di_head except weighting is
        # multiplied by a factor of how many disease individual had in the tail
        # to balance weighting.
        node_weights = [
            prev / node_prev[i % n_diseases :: n_diseases].sum()
            for i, prev in enumerate(node_prev)
        ]

        # Build the hyperarc self-edge names
        disease = [f"{dis} -> {dis}" for dis in dis_cols[:n_diseases]]

    # Convert string lists to arrays and collect output
    disease_colarr = np.array(dis_cols, dtype="<U24")
    disease_nodes = np.array(dis_nodes, dtype="<U24")
    string_outputs = (disease_colarr, disease_nodes, disease_dict)

    return string_outputs, disease, edge_weights, node_weights, palette


# Compute balanced hyperedge denominator for hyperarc weights
# *when including mortality*


@numba.njit(nogil=True, fastmath=True)
def _mort_hyperedge_denom(
    data, conds, indexes, mort_arr, mort_type, balance_mort=False
):
    """
    To help in computing weights for those hyperarcs with mortality nodes, we
    must create a new binary flag matrix, depending on the type of mortality
    setup, to help build the hyperedge denominator.

    INPUTS:
    -----------------
        data (np.array, dtype=np.uint8) : Original binary data matrix of
        individuals and condition flags.

        conds (np.arrange, dtype=np.int8) : List of ordered conditions for each
        individual.

        indexes (np.array, dtype=np.int8) : List of locations of duplicates for
        each individual.

        mort_arr (np.array, dtype=np.uint8) : 1-D array of mortality flags, is
        1 when individual has died and 0 when the individual is still alive by
        the end of cohort PoA.

        mort_type (int) : Type of mortality setup. Dictates the number of
        mortality nodes and how we encode death and life into the hypergraph as
        nodes.

        balance_mort (bool) : Because hyperedge_denom variable now defines
        final multimorbidity set to include mortality, this means that the
        prevalance array for those disease sets which are then followed by the
        mortality node (depending on mort_type) are suppressed because those
        individuals' final disease set now contains said mortality node. This
        means the prevalence array used to weight parent hyperedges of children
        hyperarcs are suppressed. This variable means that individuals with a
        mortality node contribute 1/2 to their disease set excluding the
        mortality node and 1/2 to the disease set including their mortality
        node. This is to stop overexaggerating edges which don't include
        mortality.
    """
    # Number of obs, disease, mort nodes and whether we include alive nodes
    # into mortality binmat
    n_obs = data.shape[0]
    n_diseases = data.shape[1]
    n_morts = [
        2,
        2 * n_diseases,
        n_diseases,
        1,
        n_diseases + 1,
        n_diseases,
        1,
    ][mort_type]
    alive_node = [1, 1, 0, 0, 1, 0, 0][mort_type]

    # Initialise mortality binmat
    mortality_binmat = np.zeros((n_obs, n_diseases + n_morts), dtype=np.uint8)
    mort_dupl1_binmat = np.zeros_like(mortality_binmat, dtype=np.uint8)
    mort_dupl2_binmat = np.zeros_like(mortality_binmat, dtype=np.uint8)

    # Create disease binary matrix to compute hyperedges prevalence of
    # individuals and their final multimorbidity set
    disease_binmat1 = np.zeros_like(mortality_binmat)
    disease_binmat2 = np.zeros_like(mortality_binmat)

    # loop over individuals
    for ii in range(n_obs):

        # Check whether weinclude alive node and whether individual lived or
        # died and update mortality binmat by setting condition flags for
        # disease nodes
        ind_mort = mort_arr[ii]
        incl_alive_node = int(alive_node + ind_mort > 0)
        mortality_binmat[ii, :n_diseases] = data[ii]

        # If including alive node, i.e. if individual had died or if individual
        # lived but we account for that through an alive node, add individual's
        # final multimorbidity set to disease binary matrix
        if incl_alive_node:
            disease_binmat1[ii, :n_diseases] = data[ii]
            disease_binmat2[ii, :n_diseases] = data[ii]

        # Extract condition set and duplicate set
        ind_conds = conds[ii]
        ind_conds = ind_conds[ind_conds != -1]
        n_ind_conds = ind_conds.shape[0]
        ind_idx = indexes[ii]
        ind_idx_max = ind_idx.max()

        # Last condition
        final_cond = ind_conds[-1]

        # setup mortality node lists to update binmat depending
        # on if individual lived or died and which mortality setup we're using
        surv_mort_list = [
            [n_diseases, n_diseases + 1],
            [n_diseases + final_cond, 2 * n_diseases + final_cond],
            [0, n_diseases + final_cond],
            [0, n_diseases],
            [n_diseases, n_diseases + final_cond + 1],
            [0, n_diseases + final_cond],
            [0, n_diseases],
        ][mort_type]
        mortality_binmat[ii, surv_mort_list[ind_mort]] += incl_alive_node

        # Add mortality/alive node to disease_binmat2 here
        if incl_alive_node:
            disease_binmat2[ii, surv_mort_list[ind_mort]] += 1

        # For the case where individual has duplicate at the end of
        # progression, this needs catered for when mort_type 1,2,4,5
        if (
            (ind_idx_max > -1)
            and (ind_idx_max == n_ind_conds - 2)
            and (mort_type in [1, 2, 4, 5])
        ):
            # second last condition for individual with duplicate at end of
            # progression
            dupl_cond = ind_conds[-2]

            # setup mortality node lists to update binmat depending
            # on if individual lived or died and which mortality setup we're
            # using
            dupl_surv_mort_list = [
                [n_diseases, n_diseases + 1],
                [n_diseases + dupl_cond, 2 * n_diseases + dupl_cond],
                [0, n_diseases + dupl_cond],
                [0, n_diseases],
                [n_diseases, n_diseases + dupl_cond + 1],
                [0, n_diseases + dupl_cond],
                [0, n_diseases],
            ][mort_type]
            mort_dupl1_binmat[ii, :n_diseases] = data[ii]
            mort_dupl1_binmat[ii, surv_mort_list[ind_mort]] += incl_alive_node
            mort_dupl2_binmat[ii, :n_diseases] = data[ii]
            mort_dupl2_binmat[
                ii, dupl_surv_mort_list[ind_mort]
            ] += incl_alive_node

    # Compute hyperedge prevalence
    idx_arr = np.arange(n_diseases + n_morts).astype(np.int8)
    hyperedge_denom = _compute_integer_repr(
        mortality_binmat, idx_arr, None
    ).astype(np.float64)

    # If we are balancing hyperedge denominator
    if balance_mort:
        # Only store individuals which actually had a duplicate at the end of
        # their progression
        disease_binmat1 = disease_binmat1[disease_binmat1.sum(axis=1) > 0]
        disease_binmat2 = disease_binmat2[disease_binmat2.sum(axis=1) > 0]

        # Compute hyperedge prevalence for the binary matrices computed for
        # individuals which have a mortality node for their final
        # multimorbidity including and excluding the mortality node
        hyperedge_denom_dis1 = _compute_integer_repr(
            disease_binmat1, idx_arr, None
        ).astype(np.float64)
        hyperedge_denom_dis2 = _compute_integer_repr(
            disease_binmat2, idx_arr, None
        ).astype(np.float64)

        # Combine all three
        hyperedge_denom += (
            0.5 * hyperedge_denom_dis1 - 0.5 * hyperedge_denom_dis2
        )

    # Compute hyperedge prevalence if we have duplicates
    if mort_type in [1, 2, 4, 5]:
        # Only store individuals which actually had a duplicate at the end of
        # their progression
        mort_dupl1_binmat = mort_dupl1_binmat[
            mort_dupl1_binmat.sum(axis=1) > 0
        ]
        mort_dupl2_binmat = mort_dupl2_binmat[
            mort_dupl2_binmat.sum(axis=1) > 0
        ]

        # Compute hyperedge prevalence for the binary matrices computed for
        # individuals with duplicates at the end of their progression
        hyperedge_denom_dupl1 = _compute_integer_repr(
            mort_dupl1_binmat, idx_arr, None
        ).astype(np.float64)
        hyperedge_denom_dupl2 = _compute_integer_repr(
            mort_dupl2_binmat, idx_arr, None
        ).astype(np.float64)

        # Combine all three
        hyperedge_denom += (
            0.5 * hyperedge_denom_dupl2 - 0.5 * hyperedge_denom_dupl1
        )

    return hyperedge_denom.astype(np.int64)
