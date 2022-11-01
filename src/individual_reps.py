import numba
import numpy as np

# Representing individuals
# Function to encode each individual's disease flags as a unique integer
# based on binary encoding


@numba.njit(nogil=True, fastmath=True)
def _compute_integer_repr(data, inds, disease_cols):
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


# Generate Numba-compiled power set


@numba.njit(nogil=True, fastmath=True)
def create_empty_list(value=0):
    """
    Build JIT-compiled empty integer list container since Numba doesn't like us
    specifying an empty list on its own.

    INPUTS:
    ----------------
        value (int/float) : Integer or float value to determine data type of
        list.
    """

    # Initialise a non-empty list of integers/floats and clear elements return
    # the container
    alist = numba.typed.List([value])
    alist.clear()

    return alist


@numba.njit(nogil=True, fastmath=True)
def _create_set_union(arr1, arr2):
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


@numba.njit(nogil=True, fastmath=True)
def _generate_powerset(arr, full=False):
    """
    JIT compiled version of creating the powerset from an array of type np.int8

    Function outputs a list of lists containing the power set of the specified
    n-tuple, ignoring the empty set and full set of elements in tup.

    INPUTS:
    ----------------
        arr (np.array, dtype=np.int8) : Numpy array of dtype np.int8.

        full (bool): Flag to include last element of power set, i.e. all
        members of arr.
    """
    # Initialise empty lest in power_set as the "empty set"
    power_set = [np.zeros((0), dtype=np.int8)]

    # Loop over elements in n-tuple and recursively build union of all subsets
    # in current power set, while accumulating more unique elements in power
    # set
    for element in arr:
        one_element_arr = np.array([element], dtype=np.int8)
        union = [
            np.array(
                list(_create_set_union(subarr, one_element_arr)), dtype=np.int8
            )
            for subarr in power_set
        ]
        power_set += union

    # Depending on outputting final element of power set or not - always
    # excluding the empty set.
    final_elem = [-1, len(power_set)][int(full)]

    # return power set, excluding empty set and full tuple as not needed
    return power_set[1:final_elem]
