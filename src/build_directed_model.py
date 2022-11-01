import math
import numba
import numpy as np

# Build directed model
# Build progression set for individual based on ordered condition
#  set, duplicates, undirectedness and mortality


@numba.njit(fastmath=True, nogil=True)
def _compute_progset(
    ind_cond, ind_idx, undirected=False, mort_flag=-1, mort_type=0
):
    """
    Construct disease progression set for an individual with an ordered array
    of diseases.

    ind_idx specified where in the ordered progression any 1-duplicates exist.
    In the case where duplicates are observed, the progression set will be
    constructed for an individual assuming a clean progression, and then any
    duplicates are constructed afterward by permuting those conditions which
    were observed at the same time.

    Note that this function permits the inclusion of mortality using the
    mort_flag variable. If -1, then exclude mortality alltogether, if 0 then
    individual lived by the end of cohort PoA. If 1 then individual died. Extra
    hyperarcs are created to represent progression to death/survival, taking
    special account of where a duplicate is observed for the last conditions
    for an individual.

    Note further we have implemented multiple ways of account for mortality
    using the mort_type variable. If mort_type=0 then we have a single node for
    death by end of cohort PoA and a single node for alive by end of cohort
    PoA. If mort_type=1 then we create a death node for each disease and an
    alive node for each disease. If mort_type=2 will have a dead node for each
    disease and progressions will contain self-loops for thise individuals who
    lived and mort_type=3 will have one death node and contain self. If
    mort_type=4 then will have a dead node for each disease and 1 alive node.
    If mort_type=5 then will have a dead node for each disease but no alive
    node or self-loops. If mort_type=6 then will only have 1 mortality node.

    INPUTS:
    ------------------
        ind_cond (np.array, dtype=np.int8) : Numpy array of integers
        representing order of observed conditions.

        ind_idx (np.array, dtype=np.int8) : Numpy array of integers
        representing index of ordered conditions where a 1-duplicate has
        occurred. If array contains -1, individual is assumed to have a clean
        disease progression

        undirected (bool) : Flag to specify whether progression set is
        producing undirected progressions, i.e. where duplicates don't care
        about hyperarc ordering of tail and head.

        mort_flag (int) : Integer flag for whether individual died.

        mort_type (int) : Type or mortality setup.
    """

    # Make copies of cond and idx arrays and work out maximum degree hyperarc
    # (excluding mortality) the individual contributes to
    ind_cond = ind_cond.copy()
    ind_idx = ind_idx.copy()
    hyp_degree = ind_cond.shape[0] - np.sum(ind_cond == -1)

    # Work out maximum duplicate, and check if duplicate is at the end of
    # individual progression
    max_dupl_idx = ind_idx.max()
    end_dupl_flag = int(max_dupl_idx == hyp_degree - 2)

    # Create binary flag to include mortality alltogether (where mort_flag = 0
    # or 1) or if not using mortality, this value is -1, as mort_flag = -1. If
    # mort_type=5, then don't include alive node.
    incl_mort = [1, 1, 0][mort_flag]
    alive_node = [1, 1, 1, 1, 1, 0, 0][mort_type]
    incl_alive_node = int(alive_node + mort_flag > 0)

    # Number of duplicates to deal with during creation of individual's
    # progressions Number of diseases, incremented by two if we're using
    # mortality (one node for death, another for alive by end of cohort).
    n_dupl = ind_idx.shape[0] - np.sum(ind_idx == -1)
    n_disease_only = ind_cond.shape[0]
    n_diseases = n_disease_only + incl_mort

    # create dummy array to use to build clean progression set
    dummy_vec = -1 * np.ones(shape=(n_diseases), dtype=np.int8)

    # If individual has 1 diseases
    if ind_idx[0] == -2:
        print("Individual only has 1 disease.")
        return

    # Create progression set as if individual had a clean progression
    prog_set_list = [
        ind_cond[:j].astype(np.int8) for j in range(2, hyp_degree + 1)
    ]
    prog_set_arr = np.zeros(
        (len(prog_set_list) + incl_mort * incl_alive_node, n_diseases),
        dtype=np.int8,
    )
    for i, arr in enumerate(prog_set_list):
        prog_set_arr[i] = np.array(
            list(arr) + list(dummy_vec[: (n_diseases - len(arr))]),
            dtype=np.int8,
        )

    # If including mortality, tag on last progression which will represent
    # whether the individual lived or died by the end of the cohort. Second
    # condition is for whether to include alive node.
    if (incl_mort == 1) and (incl_alive_node == 1):
        n_morts = n_disease_only
        last_prog = list(prog_set_list[-1])
        last_disease = last_prog[-1]
        mortality_list = [
            [n_morts, n_morts + 1],
            [n_morts + last_disease, 2 * n_morts + last_disease],
            [last_disease, n_morts + last_disease],
            [last_disease, n_morts],
            [n_morts, n_morts + last_disease + 1],
            [-1, n_morts + last_disease],
            [-1, n_morts],
        ][mort_type]
        mort_node = mortality_list[mort_flag]
        mort_dummy = list(dummy_vec[: n_diseases - 1 - hyp_degree])
        prog_set_arr[-1] = np.array(
            last_prog + [mort_node] + mort_dummy, dtype=np.int8
        )
        ind_idx[n_dupl] = [-1, hyp_degree - 1][end_dupl_flag]

    # Check if ind_index is -1. If not, individual has a duplicate
    if ind_idx[0] != -1:

        # If constructing undirected progressions then build this into model
        # through the mult variable, mult is used to determine number of extra
        # hyperarcs/hyperedges
        is_und = int(undirected)
        mult = [2, 1][is_und]

        # Check number of duplicates to find out number of new hyperarcs
        # which_mort is 1 when incl_mort=end_dupl_flag, i.e. if including
        # mortality and their last duplicate index was at the end of their
        # progression. It is the same as:
        # which_mort = (1-m)*e + (1-e)*m (e = end_dupl_flag, m = incl_mort).
        which_mort = [incl_mort, 1 - incl_mort][end_dupl_flag]
        if ind_idx[0] == 0:
            const = 1 + (end_dupl_flag * incl_mort) * (is_und - 1)
            n_new_hyperarcs = (
                mult * n_dupl - const - incl_mort * (1 - incl_alive_node)
            )  # Additional term is whether we have alive node or not
        else:
            n_new_hyperarcs = mult * n_dupl - 1 * is_und * (1 - which_mort)

        ind_indexes = (
            ind_idx[: n_dupl + end_dupl_flag * incl_mort * incl_alive_node]
            if n_new_hyperarcs > 0
            else ind_idx[:0]
        )
        extra_progset = np.zeros((n_new_hyperarcs, n_diseases), dtype=np.int8)

        # loop over indexes where 1-duplicates occurred
        j = 0
        for idx in ind_indexes:

            # If including mortality and we have reached the duplicate index
            # which requires the swapping of the last 2 conditions before
            # tagging on the mortality node. Can only happen when
            # incl_mort == 1 so mortality_node is known.
            if idx == hyp_degree - 1:
                last_disease = last_prog[-2]
                mortality_list = [
                    [n_morts, n_morts + 1],
                    [n_morts + last_disease, 2 * n_morts + last_disease],
                    [last_disease, n_morts + last_disease],
                    [last_disease, n_morts],
                    [n_morts, n_morts + last_disease + 1],
                    [-1, n_morts + last_disease],
                    [-1, n_morts],
                ][mort_type]
                mort_node = mortality_list[mort_flag]
                extra_progset[mult * j - 1] = np.array(
                    last_prog[:-2]
                    + last_prog[-2:][::-1]
                    + [mort_node]
                    + mort_dummy,
                    dtype=np.int8,
                )
                j += 1
                continue

            # Store first condition, the swapped pair of conditions to be
            # permuted and the second element of this pair
            deg0 = ind_cond[0]
            deg1 = ind_cond[idx : idx + 2][::-1].astype(np.int8)
            deg2 = ind_cond[idx + 1]

            # If 1-duplicate at beginning, append the swapped pair of
            # conditions if computing progression set for directed hypergraph,
            # otherwise skip
            if idx == 0 and not undirected:
                extra_progset[j] = np.array(
                    list(deg1) + list(dummy_vec[: (n_diseases - 2)]),
                    dtype=np.int8,
                )
                j += 1

            # If 1-duplicate swaps the second and third conditions, then we
            # can't use deg_prev as it indexes 2 before second condition, i.e.
            # -1'th condition, so manually construct, i.e. for permuted
            # progressiong element A, (B -> C) we require the additional
            # hyperarcs deg_alt1 = A -> C and deg_alt2 = A, C -> B, etc.
            elif idx == 1:
                deg_alt1 = np.array([deg0, deg2], dtype=np.int8)
                deg_alt2 = np.array([deg0] + list(deg1), dtype=np.int8)
                extra_progset[mult * j - 1] = np.array(
                    list(deg_alt1)
                    + list(dummy_vec[: (n_diseases - deg_alt1.shape[0])]),
                    dtype=np.int8,
                )
                if not undirected:
                    extra_progset[mult * j] = np.array(
                        list(deg_alt2)
                        + list(dummy_vec[: (n_diseases - deg_alt2.shape[0])]),
                        dtype=np.int8,
                    )
                j += 1

            # Otherwise, we can use deg_prev to ensure we add on permutations
            # after known progression prior to 1-duplicate
            else:
                deg_prev = prog_set_list[idx - 2]
                deg_alt1 = np.array(list(deg_prev) + list(deg1), dtype=np.int8)
                deg_alt2 = np.array(list(deg_prev) + [deg2], dtype=np.int8)
                extra_progset[mult * j - 1] = np.array(
                    list(deg_alt2)
                    + list(dummy_vec[: (n_diseases - deg_alt2.shape[0])]),
                    dtype=np.int8,
                )
                if not undirected:
                    extra_progset[mult * j] = np.array(
                        list(deg_alt1)
                        + list(dummy_vec[: (n_diseases - deg_alt1.shape[0])]),
                        dtype=np.int8,
                    )
                j += 1

        # Add the original progression set to the extended progression set
        full_prog_set = np.zeros(
            (prog_set_arr.shape[0] + extra_progset.shape[0], n_diseases),
            dtype=np.int8,
        )
        full_prog_set[: prog_set_arr.shape[0]] = prog_set_arr
        full_prog_set[prog_set_arr.shape[0] :] = extra_progset
    else:
        full_prog_set = prog_set_arr

    return full_prog_set


# Functions to estimate maximum number of hyperarcs, hyperedges, etc.


@numba.njit(fastmath=True, nogil=True)
def _n_choose_k(n, k):
    """
    Numba-compiled combinatorial calculator - number of ways to choose k items
    from n items without repetition and without order.

    INPUTS:
    -------------------
        n (int) : Total number of items.

        k (int) : Number of items to choose.
    """
    # n factorial as numerator divided by k factorial multiplied by n-k
    # factorial as denominator
    # Note that n! ~ \gamma(n+1) where \gamma is the gamma function.
    numerator = math.gamma(n + 1)
    denom = math.gamma(k + 1) * math.gamma(n - k + 1)

    return numerator / denom


@numba.njit(fastmath=True, nogil=True)
def _n_max_hyperarcs(n_diseases, b_hyp=True, mort_type=None):
    """
    Compute the maximum possible number of hyperarcs

    INPUTS:
    ----------------
        n_diseases (int) : Number of diseases (nodes) in the directed
        hypergraph

        b_hyp (bool) : Flag to only count the number of B-hyperarcs. If False,
        will count B-, F- and BF-hyperarcs. Note, if set to True, this is
        incidentally how many F-hyperarcs are possible due to the symmetry of
        constructing the tail and head node sets.

        mort_type (int) : Mortality type. If None, mortality is excluded. If
        integer from 0 and 6, then mortality nodes are included and instead of
        using maximum # hyperarcs from n_diseases+n_mort-node directed
        hypergraph, we know all mortality nodes are part of head sets, so we
        can intelligently count the maximum knowing this.
    """
    # Initialise sum as number of nodes as to account for self-loops
    hyperarc_sum = n_diseases

    # Loop over hyperedge degrees
    for k in range(2, n_diseases + 1):

        # Estimate n choose k using math.gamma supported by numba
        comb = _n_choose_k(n_diseases, k)
        # comb = math.comb(n_diseases, k) NOT IMPLEMENTED IN NUMBA

        # Count possible hyperarcs of hyperedge degree, depending on if we only
        # count B-hyperarcs or not
        if not b_hyp:
            hyperarc_sum += (2**k - 2) * comb
        else:
            hyperarc_sum += k * comb

    # If including mortality, mort_type will be an integer from 0 to 6
    if mort_type is not None:

        # Additive constant, this is where an alive/dead node is a generic one
        # and not tied to any disease
        add_const = [2, 0, 0, 1, 1, 0, 1][mort_type]

        # Loop over number of diseases
        for i in range(1, n_diseases + 1):
            # Multipicative constant, to take care of mortality nodes which
            # relate to the last condition observed in a hyperarc.
            mult_const = [0, 2 * i, i, 0, i, i, 0][mort_type]

            # Combine additive and multiplicative constants with the fact for
            # any (i+1)-degree mortality hyperarc, there are
            # math.comb(n_diseases, i) ways to form the tail set
            hyperarc_sum += (mult_const + add_const) * _n_choose_k(
                n_diseases, i
            )

    return int(hyperarc_sum)


@numba.njit(fastmath=True, nogil=True)
def _n_max_hyperedges(n_diseases, mort_type=None):
    """
    Given an n-node hypergraph, how many edges of degree 2 or more are there?

    INPUTS:
    -----------------
        n_diseases (int) : Number of total nodes in hypergraph.

        mort_type (int) : Mortality type. If None, mortality is excluded. If
        integer from 0 and 6, then mortality nodes are included and instead of
        using maximum # hyperarcs from n_diseases+n_mort-node directed
        hypergraph, we know all mortality nodes are part of head sets, so we
        can intelligently count the maximum knowing this.
    """
    # If including mortality, mort_type will be an integer from 0 to 6
    if mort_type is not None:
        # Initialise number of hyperedges
        no_hyperedges = 0

        # Additive constant, this is where an alive/dead node is a generic one
        # and not tied to any disease
        add_const = [2, 0, 0, 1, 1, 0, 1][mort_type]

        # Loop over number of diseases
        for i in range(1, n_diseases + 1):
            # Multipicative constant, to take care of mortality nodes which
            # relate to the last condition observed in a hyperarc.
            mult_const = [0, 2 * i, i, 0, i, i, 0][mort_type]

            # Combine additive and multiplicative constants with the fact for
            # any (i+1)-degree mortality hyperarc, there are
            # math.comb(n_diseases, i) ways to form the tail set
            no_hyperedges += (mult_const + add_const) * _n_choose_k(
                n_diseases, i
            )
    else:
        # Total number of hyperedges of n disease node undirected hypergraph
        # without mortality
        no_hyperedges = 2**n_diseases

    return int(no_hyperedges)


# Build directed model via the prevalence arrays for hyperedges,
# hyperarcs and nodes for directed model using progression-based individual
# contribution.


@numba.njit(fastmath=True, nogil=True)
def _compute_directed_model(
    data, ordered_cond, ordered_idx, mortality=None, mortality_type=0
):
    """
    Compute prevalence for all hyperarcs and hyperedge and build the incidence
    matrix negative entries represent tail nodes of the hyperarc and positive
    entries represent the head node of the hyperarc.

    This requires not only the known binary flag matrix of individuals and
    their multimorbidity set, but also an ordering of their observed disease
    progression via ordered_cond as well as information on whether certain
    conditions for the same individual were first observed on the same episode
    during interaction with the healthcare system.

    Prevalence is stored in two numpy arrays, the 1-dimensional numpy array
    hyperedge_prev stores parent hyperedge prevalence as well as single-set
    disease populations. The 2-dimensional numpy array hyperarc_prev orders the
    columns as head node entries and the rows as different tail node
    combinations. Both arrays are of type np.float64.

    We store hyperarc and hyperedge prevalences to help with computation of
    hyperarc weights downstream, i.e. for Overlap Coefficient and Modified
    Sorensen-Dice Coefficient.

    INPUTS:
    --------------------
        data (np.array, dtype=np.uint8) : Binary array representing
        observations as rows and their conditions as columns.

        ordered_cond (np.array, dtype=np.int8) : Numpy array of integers
        representing order of observed conditions.

        ordered_idx (np.array, dtype=np.int8) : Numpy array of integers
        representing index of ordered conditions where a 1-duplicate has
        occurred. If array contains -1, individual is assumed to have a clean
        disease progression.

        mortality (np.array, dtype=np.uint8) : If None, then mortality is not
        used. If numpy binary array then mortality is introduced into
        hypergraph set up.

        mortality_type (int) : If including mortality, then this is the type of
        mortality set up. There are 5 set ups. if 0, we have 1 dead node and 1
        alive node. If 1, we have n dead nodes and n alive nodes where n is the
        number of diseases. If 2, we have n dead nodes and all individuals
        alive by end of cohorot PoA loop back to their latest condition
        (self-loop). if 3, we have 1 death node and self-loop for survival. If
        4, we have n death nodes and self-loop for survival. If 5, we have n
        death nodes and nothing happens if you survive. If 6, then will have 1
        death node.
    """
    # INITIALISATION OF PREVALENCE ARRAYS, MORTALITY, INCIDENCE MATRICES, ETC.

    # Number of diseases and observations
    n_diseases = data.shape[1]
    n_obs = data.shape[0]

    # Setup for using mortality
    incl_mort = 0 if mortality[0] == -1 else 1
    n_morts = [
        2,
        2 * n_diseases,
        n_diseases,
        1,
        n_diseases + 1,
        n_diseases,
        1,
    ][mortality_type]
    alive_node = [1, 1, 1, 1, 1, 0, 0][mortality_type]

    # deduce maximum number of hyperarcs and number of hyperedges
    mort_type = [None, mortality_type][incl_mort]
    max_hyperarcs = _n_max_hyperarcs(
        n_diseases, b_hyp=True, mort_type=mort_type
    )
    max_disease_hyperedges = _n_max_hyperedges(n_diseases)
    max_total_hyperedges = (
        2 ** (n_diseases + incl_mort * n_morts)
        + incl_mort * (2 ** np.arange(n_diseases)).sum()
    )

    # Initialise hyperarc work list, hyperarc and node prevalence arrays and
    # directed incidence matrix. Dummy vector used to fill row of incidence
    # matrix as initialised as empty
    hyperarc_worklist = np.empty(
        shape=(max_hyperarcs, n_diseases + incl_mort), dtype=np.int8
    )
    hyperarc_prev = np.zeros(
        (max_disease_hyperedges, n_diseases + n_morts * incl_mort),
        dtype=np.float64,
    )
    node_prev = np.zeros(
        (2 * n_diseases + n_morts * incl_mort), dtype=np.float64
    )
    inc_mat = np.empty(
        shape=(max_hyperarcs, n_diseases + n_morts * incl_mort), dtype=np.int8
    )
    dummy_vec = np.zeros(n_diseases + n_morts * incl_mort, dtype=np.int8)

    # Hyperedge prevalence array is now split into two arrays, one which will
    # store the prevalence value, and the other which records the binary
    # encoding corresponding to the hyperedge of that prevalence as a look-up
    # array. i.e. prevalence for hyperedge {0, 4, 5} =>
    # hyperedge_prev[hyperedge_tracker == (2**0 + 2**4 + 2**5)]
    hyperedge_prev = np.zeros((max_total_hyperedges), dtype=np.int64)

    # Loop over each individuals binary flag vector representing undirected
    # multimorbidity set
    n_row = 0
    for ii in range(n_obs):

        # INITIALISATION OF PREVALENCE ARRAYS, MORTALITY, INCIDENCE MATRICES,
        # ETC.

        # Select binary realisations of individual, order of morbidities
        # representing disease progression, potential indices of duplicates

        # ind_binmat = data[ii]
        ind_cond = ordered_cond[ii]
        ind_idx = ordered_idx[ii]

        # Obtain information on whether individual died. Is -1 if excluding
        # mortality. Also ensure we are creating alive node(s) which are
        # dictated by incl_alive_node as long as incl_mort=1 and at least one
        # of alive_node and mort_flag are 1.
        ind_mort = mortality[ii]
        incl_alive_node = incl_mort * int(alive_node + ind_mort > 0)

        # # Add individual to prevalence counter for single condition sets and
        # tail/head node count
        n_ind_cond = ind_cond[ind_cond != -1].shape[0]
        node_weight = (
            n_ind_cond - 1
        )  # It was originally 1.0 for unbalanced node weighting
        for c in ind_cond[:n_ind_cond]:
            hyperedge_prev[2**c] += 1

        # Check if individual only has 1 disease, if not, then continue to
        # deduce their hyperarcs. If they are, move to next individual
        if ind_idx[0] != -2:

            # COMPUTE DISEASE/MORTALITY NODE PREVALENCE

            # If individual has a clean progression or if the duplicate ISN'T
            # at end of progression then contribution don't need to alter
            # contribution.
            if ind_idx[0] == -1 or ind_idx.max() != n_ind_cond - 1:
                fin_cond = ind_cond[n_ind_cond - 1]
                for i in range(n_ind_cond - 1):
                    node_prev[ind_cond[i]] += 1.0
                node_prev[n_diseases + fin_cond] += node_weight

                # Add prevalence to mortality node. First select the potential
                # alive-index and dead-index based on the mort_type and then
                # increment node prevalence for that node index (based on
                # ind_mort). If not including mortality, add 0 prevalence to
                # 0th index of node_prev so that we do nothing.
                surv_mort_list = [
                    [2 * n_diseases, 2 * n_diseases + 1],
                    [2 * n_diseases + fin_cond, 3 * n_diseases + fin_cond],
                    [fin_cond, 2 * n_diseases + fin_cond],
                    [fin_cond, 2 * n_diseases],
                    [2 * n_diseases, 2 * n_diseases + fin_cond + 1],
                    [n_diseases, 2 * n_diseases + fin_cond],
                    [n_diseases, 2 * n_diseases],
                ][mortality_type]
                node_prev[
                    incl_alive_node * surv_mort_list[ind_mort]
                ] += incl_alive_node * (node_weight + 1)

                # Hyperedge prevalence
                hyp_idx = incl_alive_node * (
                    2 ** (surv_mort_list[ind_mort] - n_diseases) - 1
                )
                hyperedge_prev[hyp_idx] += incl_alive_node

            # If individual has duplicate at end of progression then
            # contribution to last two diseases is halved to take into account
            # hyperarcs where these diseases have swapped their tail and head
            # role
            else:
                dupl_nodes = ind_cond[n_ind_cond - 2 : n_ind_cond]
                for i in range(n_ind_cond - 2):
                    node_prev[ind_cond[i]] += 1.0
                node_prev[dupl_nodes] += 0.5
                node_prev[n_diseases + dupl_nodes] += node_weight / 2

                # Add prevalence to mortality node. First select the potential
                # alive-index and dead-index based on the mort_type and then
                # increment node prevalence for that node index (based on
                # ind_mort). If not including mortality, add 0 prevalence to
                # 0th index of node_prev so that we do nothing.
                surv_mort_list = [
                    [2 * n_diseases, 2 * n_diseases + 1],
                    [2 * n_diseases, 3 * n_diseases],
                    [0, 2 * n_diseases],
                    [0, 2 * n_diseases],
                    [2 * n_diseases, 2 * n_diseases + 1],
                    [n_diseases, 2 * n_diseases],
                    [n_diseases, 2 * n_diseases],
                ][mortality_type]
                incl_dupl = [[0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0]][
                    mortality_type
                ][ind_mort]
                node_idx = (
                    incl_dupl * dupl_nodes[: incl_dupl + 1]
                    + surv_mort_list[ind_mort]
                )
                node_prev[incl_alive_node * node_idx] += (
                    incl_alive_node * (node_weight + 1) / 2
                )

                # Hyperedge prevalence
                hyp_idx = incl_alive_node * (2 ** (node_idx - n_diseases) - 1)
                hyperedge_prev[hyp_idx] += incl_alive_node  # - incl_dupl/2

            # COMPUTE PROGRESSION SET

            # Compute progression set based on ordering
            prog_set = _compute_progset(
                ind_cond,
                ind_idx,
                undirected=False,
                mort_flag=ind_mort,
                mort_type=mortality_type,
            )

            # Keep track of elements looping through progression set
            n_prog_obs = len(prog_set)
            prog_bin_int = np.zeros(n_prog_obs, dtype=np.int64)

            # Work out number of conditions in each progression element of
            # prog_set
            n_conds_prog = prog_set.shape[1] - (prog_set == -1).sum(axis=1)

            # LOOP OVER OBSERVED PROGRESSIONS AND INCREMENT CONTRIBUTIONS TO
            # HYPERARCS AND PARENT HYPEREDGES

            # Loop over individual's progression set
            for jj in range(n_prog_obs):

                # COMPUTE BINARY INTEGER MAPPING OF HYPERARC/HYPEREDGES

                # Extract progression set element
                elem = prog_set[jj]
                n_conds = n_conds_prog[jj]

                # Work out binary integer mappings for hyperarc
                # (bin_tail, bin_head) and parent hyperedge
                # (bin_tail + bin_head)
                bin_tail = 0
                for i in range(n_conds - 1):
                    bin_tail += 2 ** elem[i]
                head_node = elem[n_conds - 1]
                bin_headtail = bin_tail + 2**head_node

                # Work out if contribution needs to be halved to take into
                # account any duplicates
                hyperedge_cont = 1
                hyperarc_cont = 1.0
                if bin_headtail in prog_bin_int:
                    hyperedge_cont = 0
                if (n_conds == n_conds_prog).sum() == 2:
                    hyperarc_cont = 0.5

                # Add element of progression set to prog_bin_int
                prog_bin_int[jj] = bin_headtail

                # IF UNOBSERVED PROGRESSION HYPERARC, ADD TO INCIDENCE MATRIX
                # AND HYPERARC WORKLIST OTHERWISE, CONTINUE TO INCREMENT
                # CONTRIBUTIONS

                # Check if hyperarc has been seen before, if not then it should
                # still be 0 and needs to be added to incidence matrix
                if hyperarc_prev[bin_tail, head_node] == 0:

                    # Add hyperarc to worklist
                    hyperarc_worklist[n_row] = elem

                    # Update incidence matrix
                    inc_mat[n_row] = dummy_vec
                    for i in range(n_conds - 1):
                        inc_mat[n_row, elem[i]] = -1
                    inc_mat[n_row, elem[n_conds - 1]] = 1

                    n_row += 1

                # Initialise prevalence for this hyperarc and also the parent
                # hyperedge using contribution from individual
                hyperedge_prev[bin_headtail] += hyperedge_cont
                hyperarc_prev[bin_tail, head_node] += hyperarc_cont

        # If individual only has 1 disease, then half contribution to head and
        # tail disease node and contribute to single-set disease prevalence
        else:

            # CONTRIBUTE PREVALENCE FOR INDIVIDUALS WITH ONLY 1 CONDITIONS

            # If the individual only had one condition and then died, then
            # prevalence is contributed at the end row of the array, while if
            # they lived then it is contributed at start row of array
            single_cond = ind_cond[0]
            hyperarc_prev[-1 * incl_mort * ind_mort, single_cond] += 1.0
            node_prev[single_cond] += 1.0

            # List of lists specifying which the index of mortality/survival to
            # select based on mort_type
            surv_mort_list = [
                [2 * n_diseases, 2 * n_diseases + 1],
                [2 * n_diseases + single_cond, 3 * n_diseases + single_cond],
                [single_cond, 2 * n_diseases + single_cond],
                [single_cond, 2 * n_diseases],
                [2 * n_diseases, 2 * n_diseases + single_cond + 1],
                [n_diseases + single_cond, 2 * n_diseases + single_cond],
                [n_diseases, 2 * n_diseases],
            ][mortality_type]

            # For the prevalence of head node for individual with one
            # condition, if not using mortality, then add contribution to
            # head-component of single condition node. If including mortality
            # then select the index corresponding to the mort_type and whether
            # the individual lived or died by end of cohort PoA.
            cond_headnode = [
                n_diseases + single_cond,
                surv_mort_list[ind_mort],
            ][incl_mort]
            node_prev[cond_headnode] += [1.0, float(incl_alive_node)][
                incl_mort
            ]

            # Add prevalence to mortality/survival node if including mortality.
            # If not including mortality, add 0 prevalence to 0th index of
            # hyperedge_prev so that we do nothing.
            hyp_idx = incl_alive_node * (2 ** (cond_headnode - n_diseases) - 1)
            hyperedge_prev[hyp_idx] += incl_alive_node

    return (
        inc_mat[:n_row],
        hyperarc_worklist[:n_row],
        hyperarc_prev,
        hyperedge_prev,
        node_prev,
    )
