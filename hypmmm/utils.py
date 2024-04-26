import numpy as np
import numba
import math
import itertools
from string import ascii_uppercase as auc

##########################################################################################################
# 1. CREATE BINMAT, CONDS_WORKLIST AND IDX_WORKLIST
##########################################################################################################


def create_initial_worklists(num_dis, edge_list):
    """
    Creates binmat, conds_worklist and idx_worklist from edge_list.
    Only works with ficitious patients as it ignores duplicates.
    When only one disease occurs the first column should be -2.

    Args:
        num_dis (int) : Number of different possible diseases.
        edge_list (list of tuples) : Each tuple is an individual's trajectory of diseases where the first element
            in the tuple is the first disease, the second is the second disease etc.

    Returns:
        numpy array : binmat, gives binary array for whether a disease occurred.
        numpy array : conds_worklist, gives the order that the diseases occur in.
        numpy array : idx_worklist, shows whether duplicates occurred (+int), or whether only one disease
        occurred [1, -1, -1]
    """
    diseases = [*auc][:num_dis]

    # binmat
    binmat = np.zeros((len(edge_list), len(diseases)), dtype=np.int8)
    for row_index, row in enumerate(edge_list):
        for disease in row:
            binmat[row_index, diseases.index(disease)] = 1

    # conds_worklist
    conds_worklist = np.full((len(edge_list), len(diseases)), -1, dtype=np.int8)
    for row_index, row in enumerate(edge_list):
        if len(set(row)) == 1:
            for disease in set(row):
                # this for loop deals with patients progressions with only one disease
                conds_worklist[row_index, 0] = diseases.index(disease)
        else:
            for col_index, disease in enumerate(row):
                conds_worklist[row_index, col_index] = diseases.index(disease)

    # idx_worklist
    # when only one disease occurs the first column should be -2
    idx_worklist = np.full((len(edge_list), num_dis), -1, dtype=np.int8)

    for row_index, row in enumerate(edge_list):
        if len(set(row)) == 1:
            idx_worklist[row_index, 0] = -2
    return binmat, conds_worklist, idx_worklist


##########################################################################################################
# 2. CALCULATE MAXIMUM HYPEREDGES AND HYPERARCS
##########################################################################################################


@numba.njit(fastmath=True, nogil=True)
def N_choose_k(n, k):
    """
    Numba-compiled combinatorial calculator - number of ways to choose k items from n
    items without repetition and without order.

    Args:
        n (int) : Total number of items.
        k (int) : Number of items to choose.

    """
    # n factorial as numerator divided by k factorial multiplied by n-k factorial as denominator
    # Note that n! ~ \gamma(n+1) where \gamma is the gamma function.
    numerator = math.gamma(n + 1)
    denom = math.gamma(k + 1) * math.gamma(n - k + 1)

    return numerator / denom


@numba.njit(fastmath=True, nogil=True)
def N_deg_hyperarcs(n, d, b_hyperarcs=True):
    """
    Given an n-node directed hypergraph, how many d-degree hyperarcs are there?

    Args:
        n (int) : Number of total nodes in directed hypergraph.
        d (int) : Degree of hyperarcs to count
        b_hyperarcs (bool) : Flag to tell function whether to only count B-hyperarcs
        or all hyperarc variants (B-, BF- and F-hyperarcs).
    """
    # Estimate n choose k using math.gamma supported by numba
    no_hyperedges = int(N_choose_k(n, d))
    if b_hyperarcs:
        no_hyperarcs = d
    else:
        no_hyperarcs = 0
        for i in range(1, d):
            no_i_hyp = int(N_choose_k(d, i))
            no_hyperarcs += no_i_hyp

    return no_hyperedges * no_hyperarcs


@numba.njit(fastmath=True, nogil=True)
def N_max_hyperarcs(n_diseases, b_hyp=True, mort_type=None):
    """
    Compute the maximum possible number of hyperarcs.

    Args:
        n_diseases (int) : Number of diseases (nodes) in the directed hypergraph

        b_hyp (bool) : Flag to only count the number of B-hyperarcs. If False, will count
                B-, F- and BF-hyperarcs. Note, if set to True, this is incidentally how many
                F-hyperarcs are possible due to the symmetry of constructing the tail and head
                node sets.

        mort_type (int) : Mortality type. If None, mortality is excluded. If integer from 0
                and 2, then mortality nodes are included and instead of using maximum # hyperarcs
                from n_diseases+n_mort node directed hypergraph, we know all mortality nodes are
                part of head sets, so we can intelligently count the maximum knowing this.
    """
    # Initialise sum as number of nodes to account for self-loops
    hyperarc_sum = n_diseases

    # Loop over hyperedge degrees
    for k in range(2, n_diseases + 1):
        # Estimate n choose k using math.gamma supported by numba
        comb = N_choose_k(n_diseases, k)

        # Count possible hyperarcs of hyperedge degree, depending on
        # if we only count B-hyperarcs or not
        if not b_hyp:
            hyperarc_sum += (2**k - 2) * comb
        else:
            hyperarc_sum += k * comb

    # If including mortality, we account for extra hyperarcs as a result of extra node only ever fixed
    # at the head of any hypearc.
    if mort_type is not None:
        for i in range(1, n_diseases + 1):
            # Number of possible mortality nodes.
            const = [1, 2, i, i + 1, i, i + 1][mort_type]
            hyperarc_sum += const * N_choose_k(n_diseases, i)

    return int(hyperarc_sum)


##########################################################################################################
# 3. BINARY ENCODING OF INDIVIDUALS
##########################################################################################################


@numba.njit(fastmath=True, nogil=True)
def compute_bin_to_int(data):
    """
    For set of binary flag data, return the unique integer representation of each row
    (where each row is assumed to be a binarystring).

    Args:
        data (np.array, dtype=np.uint8) : Binary matrix whose rows are to be converted
        to a unique integer representation.
    """
    # Initialise array to store integer representations
    N_rows, N_cols = data.shape
    int_repr = np.empty(N_rows, dtype=np.int64)
    N_dis_arr = np.empty(N_rows, dtype=np.int64)

    # Convert each row from binary representation to unique integer representation
    for i in range(N_rows):
        elem = data[i]
        hyperedge = elem[elem != -1].astype(np.int64)
        int_repr[i] = (2**hyperedge).sum()
        N_dis_arr[i] = hyperedge.shape[0]

    return int_repr, N_dis_arr


@numba.njit(fastmath=True, nogil=True)
def compute_integer_repr(data, inds, disease_cols):
    """
    For set of binary flag data and a subset of columns specified by inds, return the
    unique integer representation of each row (where each row is assumed to be a binary
    string).

    Note that with the addition of inds, this acts as a mask and will exclude any binary
    response to those columns not in inds

    Args:
        data (np.array, dtype=np.uint8) : Binary matrix whose rows are to be converted
        to a unique integer representation.

        inds (np.array, dtype=np.int8) : Array of column indexes to use to compute integer
        representation with.

        disease_cols (np.array, dtype="<U24") : Array of of disease names.
    """
    # Number of diseases of interest, set binary responses of columns those not in inds to 0
    max_n = (
        2**inds
    ).sum() + 1  # sum of 2 ^ each disease index= maximum number of binary combinations
    n_ind_diseases = inds.shape[0]
    n_obs, n_diseases = data.shape

    # Convert each row from binary representation to unique integer representation
    # subtracting the number of 0 responses (since 2**0 = 1) and then add to prevalence
    # array
    prev_arr = np.zeros((max_n), dtype=np.int64)
    for i in range(n_obs):
        ind_int = 0
        for j in range(n_ind_diseases):
            ind_int += data[i, inds[j]] * 2 ** inds[j]
        prev_arr[ind_int] += 1

    return prev_arr


##########################################################################################################
# 4. BUILD WORKLISTS
##########################################################################################################


def reduced_powerset(iterable, min_set=0, max_set=None):
    """
    This function computes the (potentially) reduced powerset
    of an iterable container of objects.

    By default, the function returns the full powerset of the
    iterable, including the empty set and the full container itself.
    The size of returned sets can be limited using the min_set and
    max_set optional arguments.

    Args:
        iterable  (iterable) : A container of objects for which to construct the (reduced) powerset.

        min_set (int) : The smallest size of set to include in the reduced powerset. Default 0.

        max_set (int) : The largest size of set to include in the reduced powerset. By default,
        sets up to len(iterable) areincluded.
    """
    # Default setting for max set to generate combinations from
    if max_set is None:
        max_set = len(iterable) + 1

    # Generate combinations
    return itertools.chain.from_iterable(
        itertools.combinations(iterable, r) for r in range(min_set, max_set)
    )


def compute_worklist(edge_list, n_diseases, shuffle=False):
    """
    Using a list of tuples where each tuple represents a hyperedges whose entries are nodes,
    compute the work list used to compute the edge weights.

    Args:
        edges (list) : List of tuples, each list element representing a unique hyperedge.

        n_diseases (int) : Number of diseases in dataset.

        shuffle (bool): Flag to shuffle worklist.
    """
    # Initialise dummy -1 vector, work list and number of hyperedges
    n_edges = len(edge_list)
    work_list = np.zeros((n_edges, n_diseases), dtype=np.int8)
    dummy_var = list(-1 * np.ones(n_diseases))

    # Loop over tuple of edges and fill work list
    for i, e in enumerate(edge_list):
        n_nodes = len(e)
        work_list[i] = list(e) + dummy_var[n_nodes:]

    # shuffle the work list
    if shuffle:
        reindex = np.arange(work_list.shape[0])
        np.random.shuffle(reindex)
        work_list = work_list[reindex]
        edge_list = np.array(edge_list, dtype="object")[reindex].tolist()

    return work_list


def comp_edge_worklists(hyperedge_arr, contribution_type="power", shuffle=False):
    """
    Given an array of hyperedges from some dataset and a weighting system ("power", "exclusive" or
    "progression", construct work and edge lists for compatability with numba. This function uses
    compute_worklist() above to allow compatability with Numba.

    Args:
        hyperedge_arr (np.array, dtype=np.uint8) : Array of hyperedges. Number of rows represents the
        number of hyperedges and the number of columns represents the number of diseases.

        contribution_type (str) : Type of weighting system, can either be "power", "exclusive" or "progression".

        shuffle (bool): Flag to shuffle worklist.
    """
    # Extract number of hyperedges and number of diseases
    n_hyperedges, n_diseases = hyperedge_arr.shape

    # Depending on if weight system is power set or exclusive
    if contribution_type == "power":
        # Compute list of all potential hyperedges including the power set of all unique hyperedges observed above
        edges = list(
            set().union(
                *[
                    list(
                        reduced_powerset(
                            np.where(i)[0],
                            min_set=1,
                            max_set=np.array([i.sum() + 1, n_diseases + 1])
                            .min()
                            .astype(np.int64),
                        )
                    )
                    for i in hyperedge_arr
                ]
            )
        )

    elif (contribution_type == "exclusive") or (contribution_type == "progression"):
        # Compute list of only observed hyperedges (this excludes all power set edges)
        edges = [tuple(np.where(row)[0]) for row in hyperedge_arr]

    # determine worklist for powerset edges and exclusive edges
    worklist = compute_worklist(edges, n_diseases, shuffle=shuffle)

    return worklist
