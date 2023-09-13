import pytest  # noqa: F401
import numpy as np
import math

# import hypergraph-mm as hypmm
import os

# print(os.getcwd())
os.chdir("../")
from hypmm import build_model, utils, weight_functions, centrality_utils  # noqa: E402


# TODO: Test numpy max number of hyperedges/arcs calculation
# TODO: Directed Incidence matrix test
# TODO: Node weights?


def test_complete_sorensen_dice_coef():
    """
    Test the calculation of the edge weights when using the
    complete modified Sorensen-Dice Cofficient with a simple
    dataset including the binmat, conds_worklist and idx_worklist.

    The expected hyperedge weights have been calculated by
    hand and are stored in expected_weights.
    """
    simp_binmat = np.array(
        [[1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1]],
        dtype=np.int8,
    )

    simp_conds_worklist = np.array(
        [[0, 1, -1], [1, 2, -1], [2, -1, -1], [0, 1, 2]],
        dtype=np.int8,
    )

    simp_idx_worklist = np.array(
        [[-1, -1, -1], [-1, -1, -1], [-2, -1, -1], [-1, -1, -1]],
        dtype=np.int8,
    )
    colarr = np.array(["A", "B", "C"], dtype="<U24")

    expected_weights = {
        "A": (2 / 5),
        "B": (1 / 5),
        "C": (1 / 3),
        "A, B": (1 / 3),
        "B, C": (1 / 4),
        "A, B, C": (1 / 8),
    }

    _, weights, _ = build_model.compute_weights(  # hypmm.build_model.compute_weights(
        simp_binmat,
        simp_conds_worklist,
        simp_idx_worklist,
        colarr,
        "progression",
        weight_functions.modified_sorensen_dice_coefficient,
        # hypmm.weight_functions.modified_sorensen_dice_coefficient,
        utils.compute_progset,
        # hypmm.utils.compute_progset,  # progression (aggregate)
        dice_type=1,
        plot=False,
        ret_inc_mat=None,
    )

    hyperedge_weights = np.array(weights[0]["weight"])
    hyperedge_titles = np.array(weights[0]["disease set"])

    hypmm_weights = dict(zip(hyperedge_titles, hyperedge_weights))
    # make sure there are the right number of sets / weights
    assert len(hypmm_weights.values()) == len(expected_weights.values())

    # check each weight (value) for each hyperedge (key)
    for k in expected_weights:
        assert hypmm_weights[k] == expected_weights[k]


def test_powerset_sorensen_dice_coef():
    """
    Test the calculation of the edge weights when using the
    powerset Sorensen-Dice Cofficient with a simple
    dataset including the binmat, conds_worklist and idx_worklist.

    The expected hyperedge weights have been calculated by
    hand and are stored in expected_weights.
    """
    simp_binmat = np.array(
        [[1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1]],
        dtype=np.int8,
    )

    simp_conds_worklist = np.array(
        [[0, 1, -1], [1, 2, -1], [2, -1, -1], [0, 1, 2]],
        dtype=np.int8,
    )

    simp_idx_worklist = np.array(
        [[-1, -1, -1], [-1, -1, -1], [-2, -1, -1], [-1, -1, -1]],
        dtype=np.int8,
    )
    colarr = np.array(["A", "B", "C"], dtype="<U24")

    expected_weights = {
        "A": (2 / 4),
        "B": (3 / 4),
        "C": (3 / 4),
        "A, B": (2 / 5),
        "B, C": (1 / 3),
        "A, B, C": (1 / 8),
    }

    _, weights, _ = build_model.compute_weights(  # hypmm.build_model.compute_weights(
        simp_binmat,
        simp_conds_worklist,
        simp_idx_worklist,
        colarr,
        "progression",
        weight_functions.modified_sorensen_dice_coefficient,
        # hypmm.weight_functions.modified_sorensen_dice_coefficient,
        utils.compute_progset,  # hypmm.utils.compute_progset,
        # progression (aggregate)
        dice_type=0,
        plot=False,
        ret_inc_mat=None,
    )

    hyperedge_weights = np.array(weights[0]["weight"])
    hyperedge_titles = np.array(weights[0]["disease set"])

    hypmm_weights = dict(zip(hyperedge_titles, hyperedge_weights))
    # make sure there are the right number of sets / weights
    assert len(hypmm_weights.values()) == len(expected_weights.values())

    # check each weight (value) for each hyperedge (key)
    for k in expected_weights:
        assert hypmm_weights[k] == expected_weights[k]


def test_iterate_eigencentrality_vector():
    """
    Test the single iteration of the Chebyshev algorithm which
    calculates [M W M^T - diag(M W M^T)]v for incidence matrix M as a
    E x N matrix, weight matrix W as an E x E matrix and vector v as an
    N x 1 vector.
    """
    # Reproducible random incidence matrix,
    # weight vector (representing diagonal of weight matrix)
    # and iteration vector v.
    N_nodes = 4
    N_edges = 5
    seed = 42

    # Incidence matrix
    np.random.seed(seed)
    inc_mat = np.random.randint(low=0, high=2, size=(N_nodes, N_edges), dtype=np.uint8)
    inc_mat = np.unique(inc_mat, axis=1)
    inc_mat = inc_mat[:, inc_mat.sum(axis=0) > 0]
    inc_mat = inc_mat[inc_mat.sum(axis=1) > 0]
    N, M = inc_mat.shape

    # Weight and iteration vectors
    np.random.seed(seed)
    weights = (
        np.random.random(size=M)
        .reshape(
            -1,
        )
        .astype(np.float64)
    )
    np.random.seed(seed + 1)
    vector = (
        np.random.random(size=N)
        .reshape(
            -1,
        )
        .astype(np.float64)
    )

    # Construct [M W M^T - diag(M W M^T)]v using numpy
    MWMt = inc_mat.dot(np.diag(weights)).dot(inc_mat.T)
    A = MWMt - np.diag(np.diag(MWMt))
    expected = A.dot(vector)

    # Apply iterate_eigencentrality_vector()
    result = centrality_utils.iterate_eigencentrality_vector(inc_mat, weights, vector)

    # Check all values are close between result and expected
    assert np.allclose(result, expected)


def test_matrix_mult():
    """
    Test simple matrix multiplication from
    centrality_utils.matrix_mult() as it has been
    numbified.
    """
    # Generate random rectangular matrices
    E, N = 1000, 10
    seed = 42

    # Matrices M and W
    np.random.seed(seed)
    M = np.random.random(size=(E, N)).astype(np.float64)
    np.random.seed(seed)
    W = np.random.random(size=(E)).astype(np.float64)

    # Matrix multiply with numpy
    expected = np.diag(W).dot(M)
    result = centrality_utils.matrix_mult(M, W)

    # Check all values are close between result and expected
    assert np.allclose(result, expected)


def test_iterate_pagerank_vector():
    """
    Test single iteration of Chebyshev iteration for
    computing PageRank, i.e. this time we enforce row-wise
    stochasticity in the input matrix P, and our iteration vector v
    must induce a probability space, therefore vP must also be a
    row-wise sotchastic matrix.
    """
    # Generate random probability transition matrix
    N = 5
    seed = 42
    np.random.seed(seed)
    ptm = np.random.random(size=(N, N))
    ptm /= ptm.sum(axis=1)

    # Generate random vector representing probability space
    np.random.seed(seed + 1)
    vector = np.random.random(size=N)
    vector /= vector.sum()

    # Generate numpy expected
    expected = vector.dot(ptm)
    result = centrality_utils.iterate_pagerank_vector(ptm, vector)

    # Check all values are close between result and expected
    assert np.allclose(result, expected)


def test_N_choose_k():
    """
    Test numbified nCk formula
    """
    # Generate expected
    tolerance = 1e-6
    N = 10
    k = 2
    expected = math.comb(N, k)

    # Generate output
    result = utils.N_choose_k(N, k)

    assert expected - result < tolerance


def test_simple_hyperarc_weights():
    """
    Test the calculation of the hyperarc and
    hyperedge weights with very simple progressions
    hyperarcs, using the complete modified Sorensen-Dice
    Cofficient.

    The expected hyperedge weights have been calculated by
    hand and are stored in expected_weights.
    """
    simp_binmat = np.array(
        [[1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 1]],
        dtype=np.int8,
    )

    simp_conds_worklist = np.array(
        [[0, 1, -1], [1, 2, -1], [2, -1, -1], [0, 1, 2]],
        dtype=np.int8,
    )

    simp_idx_worklist = np.array(
        [[-1, -1, -1], [-1, -1, -1], [-2, -1, -1], [-1, -1, -1]],
        dtype=np.int8,
    )
    colarr = np.array(["A", "B", "C"], dtype="<U24")

    expected_weights = {
        "A -> A": (0),
        "B -> B": (0),
        "C -> C": (1 / 3),
        "A -> B": (1 / 3),
        "B -> C": (1 / 4),
        "A, B -> C": (1 / 8),
    }

    _, weights, _ = build_model.compute_weights(  # hypmm.build_model.compute_weights(
        simp_binmat,
        simp_conds_worklist,
        simp_idx_worklist,
        colarr,
        "progression",
        weight_functions.modified_sorensen_dice_coefficient,
        utils.compute_progset,
        dice_type=1,
        plot=False,
        ret_inc_mat=None,
    )

    hyperarc_df = weights[1]
    hyperarc_titles = np.array(weights[1]["progression"])

    # make sure there are the right number of sets / weights
    assert len(hyperarc_titles) == len(expected_weights.values())

    # check each expected weight for each hyperarc
    for title in hyperarc_titles:
        result = hyperarc_df[hyperarc_df.progression == title]
        expected = expected_weights[title]
        assert result.weight.iloc[0] == expected


def test_sibling_hyperarc_weights():
    """
    Test the calculation of the hyperarc and
    hyperedge weights when there are known sibling
    hyperarcs, using the complete modified Sorensen-Dice
    Cofficient.

    The expected hyperedge weights have been calculated by
    hand and are stored in expected_weights.
    """
    simp_binmat = np.array(
        [[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]],
        dtype=np.int8,
    )

    simp_conds_worklist = np.array(
        [[0, 1, -1], [1, 0, -1], [1, 2, 0], [0, 1, 2]],
        dtype=np.int8,
    )

    simp_idx_worklist = np.array(
        [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        dtype=np.int8,
    )
    colarr = np.array(["A", "B", "C"], dtype="<U24")

    expected_hyperedge_weights = {
        "A": (2 / 7),
        "B": (1 / 4),
        "C": (0),
        "A, B": (1 / 3),
        "B, C": (1 / 5),
        "A, B, C": (1 / 5),
    }

    expected_hyperarc_weights = {
        "A -> A": (0),
        "B -> B": (0),
        "C -> C": (0),
        "A -> B": (2 / 9),
        "B -> A": (1 / 9),
        "B -> C": (1 / 5),
        "B, C -> A": (1 / 10),
        "A, B -> C": (1 / 10),
    }

    _, weights, _ = build_model.compute_weights(
        simp_binmat,
        simp_conds_worklist,
        simp_idx_worklist,
        colarr,
        "progression",
        weight_functions.modified_sorensen_dice_coefficient,
        utils.compute_progset,
        dice_type=1,
        plot=False,
        ret_inc_mat=None,
    )

    hyperarc_df = weights[1]
    hyperedge_df = weights[0]
    hyperarc_titles = hyperarc_df["progression"]
    hyperedge_titles = hyperedge_df["disease set"]

    # make sure there are the right number of sets / weights
    assert len(hyperedge_titles) == len(expected_hyperedge_weights.values())
    assert len(hyperarc_titles) == len(expected_hyperarc_weights.values())

    # check each hyperedge weight
    for title in hyperedge_titles:
        result = hyperedge_df[hyperedge_df["disease set"] == title]
        expected = expected_hyperedge_weights[title]
        assert result.weight.iloc[0] == expected

    # check each hyperarc weight
    for title in hyperarc_titles:
        result = hyperarc_df[hyperarc_df.progression == title]
        expected = expected_hyperarc_weights[title]
        assert result.weight.iloc[0] == expected


def test_duplicate_hyperarc_weights():
    """
    Test the calculation of the edge weights when using the
    complete modified Sorensen-Dice Cofficient with a simple
    dataset including the binmat, conds_worklist and idx_worklist.

    The expected hyperedge weights have been calculated by
    hand and are stored in expected_weights.

    Here, we have 4 individuals and 3 diseases,
    with the following progressions:
    A -> B
    B -> A
    B -> C -> A
    A -> B -> C

    Except the 2nd and 4th progressions have
    duplicates at indexes 0 and 1, i.e. there is now
    uncertainty in the progressions such that
    A -> B
    B <-> A (i.e. A -> B or B <- A)
    B -> C -> A
    A -> B <-> C (i.e. A -> B -> C or A -> C -> B)

    These must be accounted for in the weight
    calculation, but dividing contribution made
    from these individuals for the progressions at
    each degree where a duplicate was observed.

    For the 4th individual, the progressions of
    degree 2 and 3 are now {A -> B, A -> C}
    and {A, B -> C, A, C -> B}, whose contributions
    are now halved per set.
    """
    simp_binmat = np.array(
        [[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]],
        dtype=np.int8,
    )

    simp_conds_worklist = np.array(
        [[0, 1, -1], [1, 0, -1], [1, 2, 0], [0, 1, 2]],
        dtype=np.int8,
    )

    simp_idx_worklist = np.array(
        [[-1, -1, -1], [0, -1, -1], [-1, -1, -1], [1, -1, -1]],
        dtype=np.int8,
    )
    colarr = np.array(["A", "B", "C"], dtype="<U24")

    expected_hyperedge_weights = {
        "A": (2.5 / 7.5),
        "B": (1.5 / 7.0),
        "C": (0),
        "A, B": (2.5 / 8.5),
        "A, C": (0.5 / 5),
        "B, C": (1 / 4.5),
        "A, B, C": (1 / 5),
    }

    expected_hyperarc_weights = {
        "A -> A": (0),
        "B -> B": (0),
        "C -> C": (0),
        "A -> B": (2 / 2.5) * (2.5 / 8.5),
        "B -> A": (0.5 / 2.5) * (2.5 / 8.5),
        "A -> C": (0.5 / 5),
        "B -> C": (1 / 4.5),
        "B, C -> A": (1 / 2) * (1 / 5),
        "A, B -> C": (0.5 / 2) * (1 / 5),
        "A, C -> B": (0.5 / 2) * (1 / 5),
    }

    _, weights, _ = build_model.compute_weights(
        simp_binmat,
        simp_conds_worklist,
        simp_idx_worklist,
        colarr,
        "progression",
        weight_functions.modified_sorensen_dice_coefficient,
        utils.compute_progset,
        dice_type=1,
        plot=False,
        ret_inc_mat=None,
    )

    hyperarc_df = weights[1]
    hyperedge_df = weights[0]
    hyperarc_titles = hyperarc_df["progression"]
    hyperedge_titles = hyperedge_df["disease set"]

    # make sure there are the right number of sets / weights
    assert len(hyperedge_titles) == len(expected_hyperedge_weights.values())
    assert len(hyperarc_titles) == len(expected_hyperarc_weights.values())

    # check each hyperedge weight
    for title in hyperedge_titles:
        result = hyperedge_df[hyperedge_df["disease set"] == title]
        expected = expected_hyperedge_weights[title]
        assert result.weight.iloc[0] == expected

    # check each hyperarc weight
    for title in hyperarc_titles:
        result = hyperarc_df[hyperarc_df.progression == title]
        expected = expected_hyperarc_weights[title]
        assert result.weight.iloc[0] == expected
