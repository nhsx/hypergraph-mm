import pytest
import numpy as np

# import hypergraph-mm as hypmm
import os

# print(os.getcwd())
os.chdir("../")
from hypmm import build_model, utils, weight_functions


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
        weight_functions.modified_sorensen_dice_coefficient,  # hypmm.weight_functions.modified_sorensen_dice_coefficient,
        utils.compute_progset,  # hypmm.utils.compute_progset,  # progression (aggregate)
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
        "A": (2 / 2),
        "B": (1 / 1),
        "C": (1 / 1),
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
        weight_functions.modified_sorensen_dice_coefficient,  # hypmm.weight_functions.modified_sorensen_dice_coefficient,
        utils.compute_progset,  # hypmm.utils.compute_progset,  # progression (aggregate)
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
