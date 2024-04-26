import pytest  # noqa: F401
import numpy as np

from hypmmm import build_model, weight_functions  # noqa: E402


# TODO: Test numpy max number of hyperedges/arcs calculation
# TODO: Directed Incidence matrix test
# TODO: Node weights?


def test_complete_sorensen_dice_coef_mort():
    """
    Test the calculation of the edge weights when using the
    complete modified Sorensen-Dice Cofficient with a simple
    dataset including the binmat, conds_worklist and idx_worklist.

    The expected hyperedge weights have been calculated by
    hand and are stored in expected_weights.
    """
    simp_binmat = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        dtype=np.int8,
    )

    simp_conds_worklist = np.array(
        [[0, -1, -1], [1, -1, -1], [2, -1, -1], [0, 1, 2]],
        dtype=np.int8,
    )

    simp_idx_worklist = np.array(
        [[-2, -1, -1], [-2, -1, -1], [-2, -1, -1], [-1, -1, -1]],
        dtype=np.int8,
    )

    colarr = np.array(["A", "B", "C"], dtype="<U24")

    end_prog = np.array([1, 1, 1, 1], dtype=np.int8)

    mort_type = 1
    end_type = mort_type

    expected_weights = {
        "A": (1 / 3),
        "B": (1 / 5),
        "C": (1 / 4),
        "A, MORT": (1 / 4),
        "B, MORT": (1 / 3),
        "C, MORT": (1 / 3),
        "A, B": (1 / 6),
        "A, B, C": (1 / 7),
        "A, B, C, MORT": (1 / 10),
    }

    _, weights, _ = build_model.compute_weights(
        simp_binmat,
        simp_conds_worklist,
        simp_idx_worklist,
        colarr,
        "progression",
        weight_functions.modified_sorensen_dice_coefficient,
        end_prog,
        dice_type=1,
        end_type=end_type,
        plot=False,
        ret_inc_mat=True,
        sort_weights=False,
    )

    print(weights[0])

    hyperedge_weights = np.array(weights[0]["weight"])
    hyperedge_titles = np.array(weights[0]["disease set"])

    hypmm_weights = dict(zip(hyperedge_titles, hyperedge_weights))
    # make sure there are the right number of sets / weights
    assert len(hypmm_weights.values()) == len(expected_weights.values())

    # check each weight (value) for each hyperedge (key)
    for k in expected_weights.keys():
        assert hypmm_weights[k] == expected_weights[k]
