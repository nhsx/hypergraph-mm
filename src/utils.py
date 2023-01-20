import itertools
import math

import numba
import numpy as np

###############################################################################
# 1. CALCULATE MAXIMUM HYPEREDGES AND HYPERARCS
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def N_choose_k(n, k):
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
def N_deg_hyperarcs(n, d, b_hyperarcs=True):
    """
    Given an n-node directed hypergraph, how many d-degree hyperarcs are there?

    INPUTS:
    -----------------
        n (int) : Number of total nodes in directed hypergraph.

        d (int) : Degree of hyperarcs to count

        b_hyperarcs (bool) : Flag to tell function whether to only count
        B-hyperarcs or all hyperarc variants (B-, BF- and F-hyperarcs).
    """
    # Estimate n choose k using math.gamma supported by numba
    no_hyperedges = int(N_choose_k(n, d))
    if b_hyperarcs:
        no_hyperarcs = d
    else:
        no_hyperarcs = 0
        for i in range(1, d):
            # numerator = int(math.gamma(d+1))
            # denom = int(math.gamma(i+1)) * int(math.gamma(d-i+1))
            # no_i_hyp = int(numerator / denom)
            no_i_hyp = int(N_choose_k(d, i))
            no_hyperarcs += no_i_hyp

    return no_hyperedges * no_hyperarcs


@numba.njit(fastmath=True, nogil=True)
def N_max_hyperarcs(n_diseases, b_hyp=True):
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
    """
    # Initialise sum as number of nodes as to account for self-loops
    hyperarc_sum = n_diseases

    # Loop over hyperedge degrees
    for k in range(2, n_diseases + 1):

        # Estimate n choose k using math.gamma supported by numba
        comb = N_choose_k(n_diseases, k)
        # comb = math.comb(n_diseases, k) NOT IMPLEMENTED IN NUMBA

        # Count possible hyperarcs of hyperedge degree, depending on
        # if we only count B-hyperarcs or not
        if not b_hyp:
            hyperarc_sum += (2**k - 2) * comb
        else:
            hyperarc_sum += k * comb

    return int(hyperarc_sum)


@numba.njit(fastmath=True, nogil=True)
def N_max_hyperedges(n_diseases):
    """
    Given an n-node hypergraph, how many edges of degree 2 or more are there?

    INPUTS:
    -----------------
        n_diseases (int) : Number of total nodes in hypergraph.
    """
    # Total number of hyperedges of n disease node undirected hypergraph
    no_hyperedges = 2**n_diseases

    return int(no_hyperedges)


###############################################################################
# 2.BINARY ENCODING OF INDIVIDUALS
###############################################################################


@numba.njit(
    nogil=True,
    fastmath=True,
)
def binary(num, width=13):
    """
    JIT compiled function to convert an integer to a binary string of length
    specified by width. Note that this implementation is heavily based on the
    numpy.binary_repr implementation to compile with Numba

    INPUTS:
    --------------------
        num (integer) : Integer to convert to binary string

        width (integer) : Length of binary string
    """
    # Make sure width has enough bit size to store number
    # if 2**width < num:
    #    raise ValueError(f"Binwidth cannot store integer. Make sure
    # 2**width > {num}")

    # Initialise binary array of length width
    binary_arr = np.zeros((width), dtype=np.uint8)

    # If number is not 0, loop over width and repeatedly check if number of
    # divisible by 2 and accunulate binary responses
    if num != 0:
        for i in range(width):
            if num % 2 == 1:
                binary_arr[i] = 1
            else:
                binary_arr[i] = 0
            num = num // 2

    return binary_arr


@numba.njit(
    nogil=True,
    fastmath=True,
)
def num_2_disease(number, disease_cols):
    """
    Given an integer, convert to binary and relate it to the structure of
    diseases in the original binary matrix, outputting the binary array
    identifying which diseases the number represents, but also a list of the
    disease names

    INPUTS:
    ---------------
        number (int) : Number in [2**0, 2**n_diseases-1]

        disease_cols (np.array, dtype="<U24") : Array of of disease names.
    """
    # Convert dictionary disease keys to list and extract number of diseases
    n_diseases = disease_cols.shape[0]

    # Convert number to binary and tag on remaining zeros depending on how
    # large number is
    diseases_arr = binary(number, n_diseases)
    node_no = np.where(diseases_arr)[0].astype(np.int8)

    # Output string array of diseases the number represents
    diseases = disease_cols[node_no]

    return diseases_arr, node_no, diseases, len(diseases)


@numba.njit(fastmath=True, nogil=True)
def compute_bin_to_int(data):
    """
    For set of binary flag data, return the unique integer representation of
    each row (where each row is assumed to be a binary string).


    INPUTS:
    -----------------
        data (np.array, dtype=np.uint8) : Binary matrix whose rows are to be
        converted to a unique integer representation.
    """
    # Initialise array to store integer representations
    N_rows, N_cols = data.shape
    int_repr = np.empty(N_rows, dtype=np.int64)
    N_dis_arr = np.empty(N_rows, dtype=np.int64)

    # Convert each row from binary representation to unique integer
    # representation
    for i in range(N_rows):
        elem = data[i]
        hyperedge = elem[elem != -1].astype(np.int64)
        int_repr[i] = (2**hyperedge).sum()
        N_dis_arr[i] = hyperedge.shape[0]

    return int_repr, N_dis_arr


@numba.njit(fastmath=True, nogil=True)
def compute_integer_repr(data, inds, disease_cols):
    """
    For set of binary flag data and a subset of columns specified by inds,
    return the unique integer representation of each row (where each row is
    assumed to be a binary string).

    Note that with the addition of inds, this acts as a mask and will exclude
    any binary response to those columns not in inds

    INPUTS:
    -----------------
        data (np.array, dtype=np.uint8) : Binary matrix whose rows are to be
        converted to a unique integer representation.

        inds (np.array, dtype=np.int8) : Array of column indexes to use to
        compute integer representation with.

        disease_cols (np.array, dtype="<U24") : Array of of disease names.
    """
    # Number of diseases of interest, set binary responses of columns those not
    # in inds to 0
    max_n = (2**inds).sum() + 1
    n_ind_diseases = inds.shape[0]
    n_obs, n_diseases = data.shape

    # Convert each row from binary representation to unique integer
    # representation subtracting the number of 0 responses (since 2**0 = 1) and
    # then add to prevalence array
    prev_arr = np.zeros((max_n), dtype=np.int64)
    for i in range(n_obs):
        ind_int = 0
        for j in range(n_ind_diseases):
            ind_int += data[i, inds[j]] * 2 ** inds[j]
        prev_arr[ind_int] += 1

    return prev_arr


@numba.njit(fastmath=True, nogil=True)
def comp_pwset_prev(binmat, worklist, disease_cols):
    """
    Compute the prevalence array when using Power contribution for computing
    hyperedges

    INPUTS:
    -----------------
    binmat (np.array, dtype=np.uint8) : Binary flag matrix.

    worklist (np.array, dtype=np.int8) : Hyperedge worklist

    disease_cols (np.array, dtype="<U24") : Disease column titles
    """
    # Initialise hyperedge prevalence array
    n_diseases = disease_cols.shape[0]
    max_hyperedges = 2**n_diseases
    hyperedge_prev = np.zeros(max_hyperedges, dtype=np.int64)

    # Loop over worklist and apply
    for elem in worklist:
        hyperedge = elem[elem != -1].astype(np.int64)
        hyper_idx = (2**hyperedge).sum()
        pwset_prev = compute_integer_repr(binmat, hyperedge, disease_cols)[-1]
        hyperedge_prev[hyper_idx] = pwset_prev

    return hyperedge_prev


###############################################################################
# 3. SET MANIPULATION AND POWER SET GENERATION IN NUMBA
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def create_empty_list(value=0):
    """
    Build JIT-compiled empty integer list container since Numba doesn't like us
    specifying an empty list on its own.

    INPUTS:
    ----------------
        value (int/float/str) : Integer/float/string value to determine data
        type of list.
    """

    # Initialise a non-empty list of integers/floats and clear elements
    # return the container
    alist = numba.typed.List([value])
    alist.clear()

    return alist


@numba.njit(fastmath=True, nogil=True)
def create_set_union(arr1, arr2):
    """
    Compute the union of two lists by converting them to sets and taking union

    INPUTS:
    ---------------------
        arr1 (np.array, dtype=np.int8) : First array of integers of dtype
        np.int8.

        arr2 (np.array, dtype=np.int8) : Second list of integers of dtype
        np.int8.
    """
    # Form union of lists after converting to set
    union = set(arr1).union(set(arr2))

    return union


@numba.njit(fastmath=True, nogil=True)
def concatenate(arr1, arr2):
    """
    Compute the union of two lists by converting them to sets and taking union

    INPUTS:
    ---------------------
        arr1 (np.array, dtype=np.int8) : First array of integers of dtype
        np.int8.

        arr2 (np.array, dtype=np.int8) : Second list of integers of dtype
        np.int8.
    """
    # Form union of lists after converting to set
    n1 = arr1.shape[0]
    n2 = arr2.shape[0]
    n = n1 + n2
    output = np.empty((n), dtype=arr1.dtype)
    output[:n1] = arr1
    output[n1:] = arr2

    return output


@numba.njit(fastmath=True, nogil=True)
def generate_powerset(arr, full=False, rem_single=False):
    """
    JIT compiled version of creating the powerset from an array of type np.int8

    Function outputs a list of lists containing the power set of the specified
    n-tuple, ignoring the empty set and full set of elements in tuple.

    INPUTS:
    ----------------
        arr (np.array, dtype=np.int8) : Numpy array of dtype np.int8.

        full (bool): Flag to include last element of power set, i.e. all
        members of arr.
    """
    # Initialise empty lest in power_set as the "empty set"
    power_set = [np.zeros((0), dtype=arr.dtype)]
    n_inds = arr.shape[0]

    # Loop over elements in n-tuple and recursively build union of all subsets
    # in current power set, while accumulating more unique elements in power
    # set
    for element in arr:
        one_element_arr = np.array([element], dtype=arr.dtype)
        union = [concatenate(subarr, one_element_arr) for subarr in power_set]
        power_set += union

    # Depending on outputting final element of power set or not - always
    # excluding the empty set.
    final_elem = [-1, len(power_set)][int(full)]
    power_set = power_set[1:final_elem]

    # If removing single-elements
    single_idx_size = [0, n_inds][int(rem_single)]
    single_idx = np.array([2**i - 1 for i in range(n_inds)])[::-1]
    for i in range(single_idx_size):
        del power_set[single_idx[i]]

    # return power set, excluding empty set and full tuple as not needed
    return power_set


###############################################################################
# 4. BUILD WORKLISTS
###############################################################################


def reduced_powerset(iterable, min_set=0, max_set=None):
    """
    This function computes the (potentially) reduced powerset
    of an iterable container of objects.

    By default, the function returns the full powerset of the
    iterable, including the empty set and the full container itself.
    The size of returned sets can be limited using the min_set and
    max_set optional arguments.

    INPUTS:
    ----------
        iterable  (iterable) : A container of objects for which to construct
        the (reduced) powerset.

        min_set (int) : The smallest size of set to include in the reduced
        powerset. Default 0.

        max_set (int) : The largest size of set to include in the reduced
        powerset. By default, sets up to len(iterable) areincluded.
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
    Using a list of tuples where each tuple represents a hyperedges whose
    entries are nodes, compute the work list used to compute the edge weights.

    INPUTS:
    ---------------
        edges (list) : List of tuples, each list element representing a unique
        hyperedge.

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


def comp_edge_worklists(
    hyperedge_arr, contribution_type="power", shuffle=False
):
    """
    Given an array of hyperedges from some dataset and a weighting system
    ("power", "exclusive" or "progression", construct work and edge lists for
    compatability with numba. This function uses compute_worklist() above to
    allow compatability with Numba.

    INPUTS:
    ---------------
        hyperedge_arr (np.array, dtype=np.uint8) : Array of hyperedges. Number
        of rows represents the number of hyperedges and the number of columns
        represents the number of diseases.

        contribution_type (str) : Type of weighting system, can either be
        "power", "exclusive" or "progression".

        shuffle (bool): Flag to shuffle worklist.
    """
    # Extract number of hyperedges and number of diseases
    n_hyperedges, n_diseases = hyperedge_arr.shape

    # Depending on if weight system is power set or exclusive
    if contribution_type == "power":

        # Compute list of all potential hyperedges including the power set of
        # all unique hyperedges observed above
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

    elif (contribution_type == "exclusive") or (
        contribution_type == "progression"
    ):
        # Compute list of only observed hyperedges (this excludes all power
        # set edges)
        edges = [tuple(np.where(row)[0]) for row in hyperedge_arr]

    # determine worklist for powerset edges and exclusive edges
    worklist = compute_worklist(edges, n_diseases, shuffle=shuffle)

    return worklist


###############################################################################
# 5.1. BUILD PROGRESSION SET (EXCLUSIVE PROGRESSION)
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def compute_progset(ind_cond, ind_idx, undirected=False):
    """
    Construct disease progression set for an individual with an ordered
    array of diseases.

    ind_idx specified where in the ordered progression any 1-duplicates exist.
    In the case where duplicates are observed, the progression set will be
    constructed for an individual assuming a clean progression, and then any
    duplicates are constructed afterward by permuting those conditions which
    were observed at the same time.

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

    RETURNS:
    ------------------
        full_prog_set (np.array, dtype=np.int8) : Progression set for
        individuals with ordered conditions stored in ind_cond and any
        1-duplicates stored in ind_idx.
    """

    # Make copies of cond and idx arrays and work out maximum degree hyperarc
    # (excluding mortality) the individual contributes to
    ind_cond = ind_cond.copy()
    ind_idx = ind_idx.copy()
    n_diseases = ind_cond.shape[0]
    hyp_degree = n_diseases - np.sum(ind_cond == -1)

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
    prog_set_arr = np.zeros((len(prog_set_list), n_diseases), dtype=np.int8)
    for i, arr in enumerate(prog_set_list):
        prog_set_arr[i] = np.array(
            list(arr) + list(dummy_vec[: (n_diseases - len(arr))]),
            dtype=np.int8,
        )

    # Check if ind_index is -1. If not, individual has a duplicate
    if ind_idx[0] != -1:

        # If constructing undirected progressions then build this into model
        # through the mult variable, mult is used to determine number of extra
        # hyperarcs/hyperedges
        is_und = int(undirected)
        mult = [2, 1][is_und]

        # Check number of duplicates
        n_dupl = ind_idx.shape[0] - np.sum(ind_idx == -1)
        n_new_hyperarcs = (
            mult * n_dupl if ind_idx[0] != 0 else mult * n_dupl - 1
        )
        extra_progset = np.zeros((n_new_hyperarcs, n_diseases), dtype=np.int8)
        ind_indexes = ind_idx[:n_dupl] if n_new_hyperarcs > 0 else ind_idx[:0]

        # loop over indexes where 1-duplicates occurred
        j = 0
        for idx in ind_indexes:

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
            # can't use deg_prev as it indexes 2 before second condition,
            # i.e. -1'th condition, so manually construct, i.e. for permuted
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


###############################################################################
# 5.2. BUILD PROGRESSION SET (POWER SET PROGRESSION)
###############################################################################


@numba.njit(nogil=True, fastmath=True)
def compute_pwset_progset(ind_cond, ind_idx, undirected=False):
    """
    Construct disease progression set for an individual with an ordered
    array of disease indexes.

    ind_idx specified where in the ordered progression any 1-duplicates exist.
    In the case where duplicates are observed, the progression set will be
    constructed for an individual assuming a clean progression, and then any
    duplicates are constructed afterward by permuting those conditions which
    were observed at the same time. This last step is ignored if the individual
    has no 1-duplicates or if we seen undirected progression.

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

    RETURNS:
    ------------------
        full_prog_set (np.array, dtype=np.int8) : Progression set for
        individuals with ordered conditions stored in ind_cond and any
        1-duplicates stored in ind_idx.
    """
    # Make copies of cond and idx arrays and work out maximum degree hyperarc
    # (excluding mortality) the individual contributes to
    n_diseases = ind_cond.shape[0]
    ind_cond = ind_cond[ind_cond != -1].copy().astype(np.int8)
    hyp_degree = ind_cond.shape[0]
    dummy_vec = -1 * np.ones(shape=(n_diseases), dtype=np.int8)

    # If individual has 1 diseases
    if hyp_degree == 1:
        print("Individual only has 1 disease.")
        return

    # Create full powerset of progressions given ordered conditions set
    if hyp_degree > 2:
        prog_set_list = generate_powerset(ind_cond, full=True, rem_single=True)
    else:
        prog_set_list = [ind_cond]
    prog_set_arr = np.empty((len(prog_set_list), n_diseases), dtype=np.int8)
    for i, elem in enumerate(prog_set_list):
        deg = elem.shape[0]
        prog_set_arr[i] = np.array(
            list(elem) + list(dummy_vec[: (n_diseases - deg)]), dtype=np.int8
        )

    # Check if ind_index is -1. If not, individual has a duplicate
    if ind_idx[0] != -1:

        # If constructing undirected progressions then build this into model
        # through the mult variable, mult is used to determine number of extra
        # hyperarcs/hyperedges
        is_und = int(undirected)
        mult = [2, 1][is_und]

        # Check number of duplicates
        n_dupl = ind_idx.shape[0] - np.sum(ind_idx == -1)
        n_new_hyperarcs = (
            mult * n_dupl if ind_idx[0] != 0 else mult * n_dupl - 1
        )
        extra_progset = np.zeros((n_new_hyperarcs, n_diseases), dtype=np.int8)
        ind_indexes = ind_idx[:n_dupl] if n_new_hyperarcs > 0 else ind_idx[:0]

        # loop over indexes where 1-duplicates occurred
        j = 0
        for idx in ind_indexes:

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
            # can't use deg_prev as it indexes 2 before second condition,
            # i.e. -1'th condition, so manually construct, i.e. for permuted
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


###############################################################################
# 5.3. BUILD PROGRESSION SET (SIMPLE_PROGRESSION)
###############################################################################


@numba.njit(nogil=True, fastmath=True)
def compute_simple_progset(ind_cond, ind_idx, undirected=False):
    """
    Construct disease progression set for an individual with an ordered
    array of disease indexes.

    ind_idx specified where in the ordered progression any 1-duplicates exist.
    In the case where duplicates are observed, the progression set will be
    constructed for an individual assuming a clean progression, and then any
    duplicates are constructed afterward by permuting those conditions which
    were observed at the same time. This last step is ignored if the individual
    has no 1-duplicates or if we seen undirected progression.

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

    RETURNS:
    ------------------
        full_prog_set (np.array, dtype=np.int8) : Progression set for
        individuals with ordered conditions stored in ind_cond and any
        1-duplicates stored in ind_idx.
    """
    # Make copies of cond and idx array
    max_degree = 2
    n_diseases = ind_cond.shape[0]
    n_remaining = n_diseases - max_degree
    ind_cond = ind_cond.copy().astype(np.int8)
    ind_idx = ind_idx.copy()

    # Number of duplicates and maximum number degree hyperarc individua
    n_dupls = ind_idx.shape[0] - np.sum(ind_idx == -1)
    hyp_degree = ind_cond.shape[0] - np.sum(ind_cond == -1)

    # Number of clean progressions and number of progressions including
    # duplicates
    n_clean = np.arange(1, hyp_degree).sum()
    n_progs = n_clean + (1 - int(undirected)) * n_dupls

    # Intiialise progression set array and dummy vector
    prog_set_arr = np.empty((n_progs, n_diseases), dtype=np.int8)
    dummy_vec = -1 * np.ones(shape=(n_diseases), dtype=np.int8)

    # If individual has 1 diseases
    if ind_idx[0] == -2:
        print("Individual only has 1 disease.")
        return

    # Create progression set as if individual had a clean progression
    counter = 0
    for i in range(hyp_degree):
        for j in range(i + 1, hyp_degree):
            arr = ind_cond[np.array([i, j])]
            prog_set_arr[counter] = np.array(
                list(arr) + list(dummy_vec[:n_remaining]), dtype=np.int8
            )
            counter += 1

    # Check if ind_index is -1. If not, individual has a nonzero number of
    # 1-duplicates. This is easy to do as we can locate the 1-duplicate
    # conditions, flip them, and increment them to list
    if ind_idx[0] != -1 and not undirected:
        for i in range(n_dupls):
            dupl_idx = ind_idx[i]
            arr = ind_cond[dupl_idx : dupl_idx + 2][::-1]
            prog_set_arr[counter] = np.array(
                list(arr) + list(dummy_vec[:n_remaining]), dtype=np.int8
            )
            counter += 1

    return prog_set_arr
