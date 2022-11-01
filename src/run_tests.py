from build_directed_model import _compute_directed_model
from hyperarc_weights import _comp_hyperarc_weight
from hyperedge_weights import (
    _comp_overlap_coeff,
    _modified_dice_coefficient_comp,
    _modified_dice_coefficient_pwset,
)
from individual_reps import _compute_integer_repr, _generate_powerset
from utils import _setup_weight_comp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import string
import time as t
from tabulate import tabulate

# Run test case to compute hyperedge and hyperarc weights


def _run_test_case(
    test_binmat,
    test_conds_worklist,
    test_idx_worklist,
    end_prog,
    contribution_type,
    weight_function,
    plot=True,
):
    """
    This function will take an example dataset of individuals with arbitrary
    diagnoses, whose condition sets are ordered and duplicates are specified.

    The function will then build the prevalence arrays, incidence matrix and
    compute the edge weights for the hyperedges and hyperarcs.

    The function allows specification of which weight function to use and which
    contribution type to use, i.e. exclusive, progression or power set based.
    In future the function will handle mortality as a variable to consider
    also.

    INPUTS:
    -------------
        test_binmat (np.array, dtype=np.uint8) : Binary flag matrix whose test
        observations are represented by rows and diseases depresented by
        columns.

        test_conds_worklist (np.array, dtype=np.int8) : For each individual,
        the worklist stores the ordered set of conditions via their columns
        indexes. For compatability with Numba, once all conditions specified
        per individual, rest of row is a stream if -1's.

        test_idx_worklist (np.array, dtype=np.int8) : Worklist to specify any
        duplicates. For each individuals, if -2 then they only have 1
        condition, if -1 then they have no duplicates and if they have positive
        integers indexing test_conds_worklist then it specifies which
        conditions have a duplicate which needs taken care of.

        end_prog (np.array, dtype=np.int8) : Array storing end of progression
        information on individuals. If an array of -1's, then excluding
        progression end information.

        contribution_type (str) : Type of contribution to hyperedges each
        individual has, i.e. can be "exclusive", "power" or "progression".

        weight_function (func) : Numba-compiled weight function, will be
        version of the overlap coefficient and modified sorensen-dice
        coefficient. The args argument are the optional argument specified for
        the weight function.

        plot (bool) : Flag to plot hyperedge and hyperarc weights.
    """
    # Number of observations and columns and set up end of progressions
    N_obs, N_cols = test_binmat.shape
    colarr = np.asarray(list(string.ascii_uppercase[:N_cols]), dtype="<U24")
    incl_end = 1 if end_prog[0] != -1 else 0
    n_end = end_prog.sum()
    end_type = 0

    # Build incidence matrix, prevalence arrays and hyperarc worklists
    print(
        """Building directed incidence matrix, work list and
         prevalence arrays..."""
    )
    st = t.time()
    output = _compute_directed_model(
        test_binmat,
        test_conds_worklist,
        test_idx_worklist,
        mortality=end_prog,
        mortality_type=end_type,
    )

    (
        inc_mat,
        hyperarc_worklist,
        hyperarc_prev,
        hyperedge_prev,
        node_prev,
    ) = output
    print(f"Completed in {round(t.time()-st,2)} seconds.")
    print(inc_mat.shape)

    # Compute exclusive count of hyperedges of individuals from snapshot at
    # end of PoA
    columns_idxs = np.arange(N_cols).astype(np.int8)
    hyperedge_denom = _compute_integer_repr(test_binmat, columns_idxs, colarr)

    # Depending on function type, define denominator array
    if weight_function == _comp_overlap_coeff:
        denom_prev = hyperedge_prev.copy()
    else:
        denom_prev = hyperedge_denom.copy()

    # Depending on contribution type, specify prevalence arrays for
    # hyperedge/hyperarc weights.
    pwset = _generate_powerset(columns_idxs, full=True)
    hyperarc_num_prev = hyperedge_prev.copy()
    if contribution_type == "exclusive":
        hyperedge_num_prev = hyperedge_denom.copy()
    elif contribution_type == "progression":
        hyperedge_num_prev = hyperedge_prev.copy()

    # If power set contribution, build hyperedge array by looping over all
    # hyperedges and using _compute_integer_repr(elem), taking the last
    # element of the resulting array as the power set prevalence for elem
    elif contribution_type == "power":
        hyperedge_num_prev = np.zeros_like(hyperedge_prev, dtype=np.int8)
        for i, elem in enumerate(pwset):
            hyperedge_num_prev[i + 1] = _compute_integer_repr(
                test_binmat, elem, colarr
            )[-1]

    # Build edge weights. If power set contribution and using overlap
    # coefficient or power set dice coefficient then compute single disease set
    # weights as the proportion of individuals with the disease out of all
    # individuals.
    if weight_function == _modified_dice_coefficient_pwset:
        hyp_cols = list(colarr).copy()
        hyper_weights = list(test_binmat.sum(axis=0) / test_binmat.shape[0])
        pwset = [elem for elem in pwset if elem.shape[0] != 1]
    elif (
        weight_function == _comp_overlap_coeff
        and contribution_type != "exclusive"
    ):
        hyp_cols = list(colarr).copy()
        hyper_weights = list(test_binmat.sum(axis=0) / test_binmat.shape[0])
        pwset = [elem for elem in pwset if elem.shape[0] != 1]
    else:
        hyp_cols = []
        hyper_weights = []

    # Loop over elements in pwset
    for elem in pwset:
        hyper_col = colarr[elem]
        w = weight_function(
            test_binmat, elem, colarr, hyperedge_num_prev, denom_prev
        )[0]
        hyper_weights.append(w)
        hyp_cols.append(", ".join(hyper_col))

    # Build dataframe of hyperedge weights
    hyperedge_weights = pd.DataFrame(
        {"disease set": hyp_cols, "weight": hyper_weights}
    ).sort_values(by=["weight"], ascending=False)

    # Plot hyperedge weights
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(x="weight", y="disease set", data=hyperedge_weights, ax=ax)
        ax.set_title("Edge Weights", fontsize=18)
        ax.set_ylabel("Disease Set", fontsize=18)
        ax.set_yticklabels(list(ax.get_yticklabels()), fontsize=15)
        # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=15)
        ax.set_xlabel("Weight", fontsize=18)

    # Build hyperarc weights
    # Set up variables for edge weights, nodes, node weights, palette, etc.
    # dependent on whether mortality is used
    output = _setup_weight_comp(
        colarr,
        colarr,
        test_binmat,
        node_prev,
        hyperarc_prev,
        incl_end,
        end_type,
        n_died=n_end,
    )

    (
        (colarr, nodes, test_dict),
        col_progs,
        edge_weights,
        node_weights,
        palette,
    ) = output

    # Initialise disease progression names and weights and add self-edges to
    # hyperarc worklist only if we're using the complete dice coefficient,
    # otherwise use what has been defined above in _setup_weight_comp()
    if weight_function == _modified_dice_coefficient_comp:
        edge_weights = []
        col_progs = []
        self_edge_worklist = np.array(
            [[i] + (N_cols - 1) * [-1] for i in range(N_cols)], dtype=np.int8
        )
        hyperarc_worklist = np.concatenate(
            [self_edge_worklist, hyperarc_worklist], axis=0
        )

    # Loop over hyperarcs, compute weight and append weight and disease
    # progression title to respective lists
    for hyperarc in hyperarc_worklist:
        hyperarc = hyperarc[hyperarc != -1]
        hyperarc_cols = colarr[hyperarc]
        weight, denom = _comp_hyperarc_weight(
            test_binmat,
            hyperarc,
            weight_function,
            hyperarc_prev,
            hyperarc_num_prev,
            colarr,
            hyperedge_num_prev,
            denom_prev,
        )
        edge_weights.append(weight)

        # If self-edge hyperarc weight, define string title differently
        if hyperarc.shape[0] == 1:
            col_progs.append(f"{hyperarc_cols[-1]} -> {hyperarc_cols[-1]}")
        else:
            col_progs.append(
                ", ".join(hyperarc_cols[:-1]) + " -> " + hyperarc_cols[-1]
            )

    # Build dataframe of hyperedge weights
    hyperarc_weights = pd.DataFrame(
        {"progression": col_progs, "weight": edge_weights}
    ).sort_values(by=["weight"], ascending=False)

    # If plotting edge and node weights
    node_weights = pd.DataFrame({"node": nodes, "weight": node_weights})
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(x="weight", y="progression", data=hyperarc_weights, ax=ax)
        ax.set_title("Edge Weights", fontsize=18)
        ax.set_ylabel("Disease Progression", fontsize=18)
        ax.set_yticklabels(list(ax.get_yticklabels()), fontsize=15)
        # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=15)
        ax.set_xlabel("Weight", fontsize=18)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(
            x="weight", y="node", data=node_weights, ax=ax, palette=palette
        )
        ax.axvline(x=1 / 2, ymin=0, ymax=10, c="r", linestyle="--")
        ax.set_title("Node Weights", fontsize=18)
        ax.set_yticklabels(list(ax.get_yticklabels()), fontsize=15)
        # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=15)
        ax.set_ylabel("Tail/Head Nodes", fontsize=18)
        ax.set_xlabel("Weight", fontsize=18)

    # return hyperedge_weights, hyperarc_weights, node_weights
    print()
    print(
        tabulate(
            hyperedge_weights, headers="keys", tablefmt="psql", showindex=False
        )
    )
    print()
    print(
        tabulate(
            hyperarc_weights, headers="keys", tablefmt="psql", showindex=False
        )
    )
    print()
    print(
        tabulate(
            node_weights, headers="keys", tablefmt="psql", showindex=False
        )
    )
