import numpy as np
import pandas as pd

from hypmmm import centrality
from hypmmm import build_model
from hypmmm import create_figures


##########################################################################################################
# 1. REMOVE CORRECTED PAGERANK
##########################################################################################################


def compute_corrected_pagerank(merged_df, trans_type):
    """
    Calculate the normalised Remove Corrected PageRank using the no mortality
    and mortality PageRanks.
    Get the sum of the non-mort and mort type 1 for each disease.
    Then normalise this across all the diseases, ALIVE and MORT.

    Args:
        merged_df (df): Dataframe of no mortality and mortality PageRanks.
        trans_type (str): 'suc' for successor columns, 'pred' for predecessor columns.
    Returns:
        list: Remove corrected PageRank for either the successors or predecessors.

    """
    com_PageRank = list()

    for idx in range(0, len(merged_df)):
        if trans_type == "suc":
            total = (
                merged_df.loc[idx, "No Mort Suc PageRank"]
                + merged_df.loc[idx, "Mort 1 Suc PageRank"]
            )
        if trans_type == "pred":
            total = (
                merged_df.loc[idx, "No Mort Pred PageRank"]
                + merged_df.loc[idx, "Mort 1 Pred PageRank"]
            )

        com_PageRank.append(total)

    norm_PageRank = [(v / sum(n)) for n in [com_PageRank] for v in n]
    return norm_PageRank


def calc_remove_pagerank(
    n_diseases,
    binmat,
    conds_worklist,
    idx_worklist,
    contribution_type,
    weight_function,
    complete_denom,
    end_prog,
    plot,
):
    """
    Calculate remove corrected PageRank using the PageRank calculated
    when mortality is and isn't included.
    Args:
        n_diseases (int) : Number of diseases included.

        binmat (np.array, dtype=np.uint8) : Binary flag matrix whose test observations
        are represented by rows and diseases depresented by columns.

        conds_worklist (np.array, dtype=np.int8) : For each individual, the worklist storesx
        the ordered set of conditions via their columns indexes. For compatability with Numba, once
        all conditions specified per individual, rest of row is a stream if -1's.

        idx_worklist (np.array, dtype=np.int8) : Worklist to specify any duplicates. For each
        individuals, if -2 then they only have 1 condition, if -1 then they have no duplicates and
        if they have positive integers indexing conds_worklist then it specifies which
        conditions have a duplicate which needs taken care of.

        contribution_type (str) : Type of contribution to hyperedges each individual has, i.e. can be
        "exclusive", "power" or "progression".

        weight_function (func) : Numba-compiled weight function, will be version of the overlap coefficient
        and modified sorensen-dice coefficient.

        complete_denom (bool) :

        end_prog (numpy array) : Binary vector of whether mortality occured in analysis period
        (1 = Mortality, 0 = Alive).

        plot (bool) : Flag to plot hyperedge weights, hyperarc weights and transition matrices for mortality type 1.

    Returns:
        dataframe : Dataframe containing PageRank for no mortality, mortality type 1 and remove corrected mortality.
        numpy array : mort1_inc_mat_tail
        numpy array : mort1_inc_mat_head
        numpy array : mort1_hyperarc_weights
        numpy array : mort1_node_weights
        numpy array : mort1_hyperarc_titles

    """
    ##########################################################################################################
    # FIRST CALCULATE PAGERANK WITHOUT MORTALITY
    ##########################################################################################################
    print("*" * 20, "\nComputing PageRank without Mortality...")
    no_mort_end_prog = -1 * np.ones(binmat.shape[0], dtype=np.int8)  # Ignores mortality

    no_mort_inc_mat, no_mort_weights, no_mort_mort_colarr = build_model.compute_weights(
        binmat,
        conds_worklist,
        idx_worklist,
        None,
        contribution_type,
        weight_function,
        no_mort_end_prog,
        dice_type=complete_denom,
        end_type=0,
        plot=False,
        ret_inc_mat=True,
        sort_weights=False,
    )

    # no_mort_hyperedge_weights = np.array(no_mort_weights[0]["weight"])
    # no_mort_hyperedge_titles = np.array(no_mort_weights[0]["disease set"])

    no_mort_hyperarc_weights = np.array(no_mort_weights[1]["weight"])
    no_mort_progression_titles = np.array(no_mort_weights[1]["progression"])

    no_mort_node_weights = np.array(no_mort_weights[2]["weight"])
    # no_mort_node_titles = np.array(no_mort_weights[2]["node"])

    no_mort_output = build_model.setup_vars(
        no_mort_inc_mat,
        n_diseases,
        no_mort_hyperarc_weights,
        no_mort_progression_titles,
        no_mort_node_weights,
        mort_type=None,
    )
    (
        no_mort_inc_mat_data,
        no_mort_hyperarc_data,
        no_mort_node_weights,
        no_mort_node_degs,
        no_mort_edge_degs,
    ) = no_mort_output

    no_mort_inc_mat_tail, no_mort_inc_mat_head = no_mort_inc_mat_data

    no_mort_edge_weights, no_mort_hyperarc_titles = no_mort_hyperarc_data

    no_mort_node_degree_tail, no_mort_node_degree_head = no_mort_node_degs
    no_mort_edge_degree_tail, no_mort_edge_degree_head = no_mort_edge_degs

    # SUCCESSOR PAGERANK
    no_mort_inpt = (
        (no_mort_inc_mat_tail, no_mort_inc_mat_head),
        (no_mort_edge_weights, no_mort_node_weights),
        no_mort_node_degs,
        no_mort_edge_degree_tail,
    )

    no_mort_node_pagerank = centrality.pagerank_centrality(
        no_mort_inpt,
        rep="standard",
        typ="successor",
        tolerance=1e-10,
        max_iterations=1000,
        is_irreducible=True,
        weight_resultant=False,
        random_seed=None,
        eps=0.00001,
    )

    no_mort_all_node_sc_evc = pd.DataFrame(
        {"Disease": no_mort_mort_colarr, "No Mort Suc PageRank": no_mort_node_pagerank}
    )

    # no_mort_succ_order = (
    #     no_mort_all_node_sc_evc.sort_values(by="No Mort Suc PageRank", ascending=False)
    #     .reset_index(drop=True)
    #     .Disease
    # )
    no_mort_succ_node_sc_evc = (
        no_mort_all_node_sc_evc.sort_values(by="No Mort Suc PageRank", ascending=False)
        .reset_index(drop=True)
        .round({"No Mort Suc PageRank": 3})
    )

    # PREDECESSOR PAGERANK
    # node_pagerank = centrality.pagerank_centrality(
    #     no_mort_inpt,
    #     rep="standard",
    #     typ="predecessor",
    #     tolerance=1e-5,
    #     max_iterations=1000,
    #     is_irreducible=True,
    #     weight_resultant=True,
    #     random_seed=None,
    # )

    no_mort_all_node_pr_evc = pd.DataFrame(
        {"Disease": no_mort_mort_colarr, "No Mort Pred PageRank": no_mort_node_pagerank}
    )

    # no_mort_pred_order = (
    #     no_mort_all_node_pr_evc.sort_values(by="No Mort Pred PageRank", ascending=False)
    #     .reset_index(drop=True)
    #     .Disease
    # )
    no_mort_pred_node_pd_evc = (
        no_mort_all_node_pr_evc.sort_values(by="No Mort Pred PageRank", ascending=False)
        .reset_index(drop=True)
        .round({"No Mort Pred PageRank": 3})
    )

    no_mort_pagerank = no_mort_pred_node_pd_evc.merge(no_mort_succ_node_sc_evc)

    ##########################################################################################################
    # NEXT CALCULATE PAGERANK WITH MORTALITY (TYPE 1)
    ##########################################################################################################
    print("*" * 20, "\nComputing PageRank with Mortality Type 1...")
    mort1_inc_mat, mort1_weights, mort_colarr = build_model.compute_weights(
        binmat,
        conds_worklist,
        idx_worklist,
        None,
        contribution_type,
        weight_function,
        end_prog,
        dice_type=complete_denom,
        end_type=1,
        plot=plot,
        ret_inc_mat=True,
        sort_weights=False,
    )

    # mort1_hyperedge_weights = np.array(mort1_weights[0]["weight"])
    # mort1_hyperedge_titles = np.array(mort1_weights[0]["disease set"])

    mort1_hyperarc_weights = np.array(mort1_weights[1]["weight"])
    mort1_progression_titles = np.array(mort1_weights[1]["progression"])

    mort1_node_weights = np.array(mort1_weights[2]["weight"])
    # mort1_node_titles = np.array(mort1_weights[2]["node"])

    mort1_output = build_model.setup_vars(
        mort1_inc_mat,
        n_diseases,
        mort1_hyperarc_weights,
        mort1_progression_titles,
        mort1_node_weights,
        mort_type=1,
    )
    (
        mort1_inc_mat_data,
        mort1_hyperarc_data,
        mort1_node_weights,
        mort1_node_degs,
        mort1_edge_degs,
    ) = mort1_output

    mort1_inc_mat_tail, mort1_inc_mat_head = mort1_inc_mat_data

    mort1_edge_weights, mort1_hyperarc_titles = mort1_hyperarc_data

    mort1_node_degree_tail, mort1_node_degree_head = mort1_node_degs
    mort1_edge_degree_tail, mort1_edge_degree_head = mort1_edge_degs

    if plot:
        create_figures.suc_trans_matrix(
            mort1_inc_mat_tail,
            mort1_inc_mat_head,
            mort1_edge_weights,
            mort1_node_degree_tail,
            mort_colarr,
        )

        create_figures.pred_trans_matrix(
            mort1_inc_mat_tail,
            mort1_inc_mat_head,
            mort1_edge_weights,
            mort1_node_degree_head,
            mort1_edge_degree_tail,
            mort_colarr,
        )

    # SUCCESSOR PAGERANK

    mort1_inpt = (
        (mort1_inc_mat_tail, mort1_inc_mat_head),
        (mort1_edge_weights, mort1_node_weights),
        mort1_node_degs,
        mort1_edge_degree_tail,
    )
    mort1_node_pagerank = centrality.pagerank_centrality(
        mort1_inpt,
        rep="standard",
        typ="successor",
        tolerance=1e-10,
        max_iterations=1000,
        is_irreducible=True,
        weight_resultant=False,
        random_seed=None,
        eps=0.00001,
    )

    mort1_all_node_sc_evc = pd.DataFrame(
        {"Disease": mort_colarr, "Mort 1 Suc PageRank": mort1_node_pagerank}
    )

    # mort1_succ_order = (
    #     mort1_all_node_sc_evc.sort_values(by="Mort 1 Suc PageRank", ascending=False)
    #     .reset_index(drop=True)
    #     .Disease
    # )
    mort1_succ_node_sc_evc = (
        mort1_all_node_sc_evc.sort_values(by="Mort 1 Suc PageRank", ascending=False)
        .reset_index(drop=True)
        .round({"Mort 1 Suc PageRank": 3})
    )

    # PREDECESSOR PAGERANK

    mort1_node_pagerank = centrality.pagerank_centrality(
        mort1_inpt,
        rep="standard",
        typ="predecessor",
        tolerance=1e-5,
        max_iterations=1000,
        is_irreducible=True,
        weight_resultant=True,
        random_seed=None,
    )

    mort1_all_node_pr_evc = pd.DataFrame(
        {"Disease": mort_colarr, "Mort 1 Pred PageRank": mort1_node_pagerank}
    )

    # mort1_pred_order = (
    #     mort1_all_node_pr_evc.sort_values(by="Mort 1 Pred PageRank", ascending=False)
    #     .reset_index(drop=True)
    #     .Disease
    # )
    mort1_pred_node_pd_evc = (
        mort1_all_node_pr_evc.sort_values(by="Mort 1 Pred PageRank", ascending=False)
        .reset_index(drop=True)
        .round({"Mort 1 Pred PageRank": 3})
    )

    mort1_pagerank = mort1_pred_node_pd_evc.merge(mort1_succ_node_sc_evc)

    ##########################################################################################################
    # MERGE THE MORT AND NO-MORT PAGERANKS TO GET THE REMOVE CORRECTED PAGERANK
    ##########################################################################################################
    print("*" * 20, "\nComputing Remove Corrected PageRank...")
    merged_df = pd.merge(no_mort_pagerank, mort1_pagerank, on="Disease", how="outer")
    merged_df = merged_df.fillna(0)

    norm_suc_pagerank = compute_corrected_pagerank(merged_df, "suc")
    merged_df["Corrected Suc PageRank"] = norm_suc_pagerank

    norm_pred_pagerank = compute_corrected_pagerank(merged_df, "pred")
    merged_df["Corrected Pred PageRank"] = norm_pred_pagerank

    return (
        merged_df,
        mort1_inc_mat_tail,
        mort1_inc_mat_head,
        mort1_hyperarc_weights,
        mort1_node_weights,
        mort1_hyperarc_titles,
    )
