import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

from hypmmm import centrality


def pagerank_scatter(suc_col, pred_col, dis_col, file_name=None):
    """
    Create a scatterplot comparing successor PageRank to predecessor PageRank
    to observe whether diseases are more likely to be a successor or predecessor.

    Args:
        suc_col (df col) : PageRank successor column from dataframe.

        pred_col (df col) : PageRank predecessor column from dataframe.

        dis_col (df_col) : Column of dataframe with disease labels.

        file_name (string) : Location and name of file.

    Return:
        plt figure : Scatterplot of successor and predecessor PageRank

    """
    fig = plt.figure()

    # Triangles
    x = [-0.01, 0.99, -0.01]
    y = [-0.01, 0.99, 0.99]

    max_len = max(pred_col.max(), suc_col.max())

    plt.fill(x, y, color="orange", alpha=0.1)
    plt.text(0.001, max_len + 0.04, "Predecessor", color="orange")

    plt.fill(y, x, color="green", alpha=0.1)
    plt.text(max_len + 0.03, 0.001, "Successor", color="green", rotation=90)

    plt.scatter(x=suc_col, y=pred_col, s=800, c=range(0, len(dis_col)), alpha=0.5)

    for x, y, label in zip(suc_col, pred_col, dis_col):
        plt.text(x, y, label, va="center", ha="center")

    plt.xlim(-0.01, max_len + 0.05)
    plt.ylim(-0.01, max_len + 0.05)
    plt.plot(
        (-0.01, max_len + 0.05),
        (-0.01, max_len + 0.05),
        ls="--",
        color="blue",
    )
    plt.text(
        max_len - 0.05,
        max_len - 0.04,
        "Transitive",
        ha="center",
        va="center",
        rotation=45,
        color="blue",
    )
    plt.xlabel = "Successor PageRank"
    plt.ylabel = "Predecessor PageRank"

    plt.show()
    if file_name is not None:
        fig.savefig(file_name)


def dis_prog_paths(rank_progressions, max_d, top_n, prog_idx_lists):
    """
    Create a scatterplot which shows the top n most common disease progressions for each degree.
    Hyperarcs which could be successors or predecessors are joined via lines.

    Args:
        rank_progressions (pd dataframe) : Dataframe with cols: index, Disease, Degree, Eigenvector Centrality.

        max_d (int) : Maximum hyperarc degree (how many nodes max).

        top_n (int) : How many top progressions to show.

        prog_idx_lists (list of lists) : List of lists of progressions.

    Returns:
        plt figure : Scatterplot of top disease/mortality progressions connected if progression
                     can occur between hyperarcs.
    """
    linestyle_lsts = list(matplotlib.lines.lineStyles.keys())
    prog_colours = list(plt.get_cmap("nipy_spectral")(np.linspace(0, 1, max_d)))[:-1]

    sq_s = 10
    fig, ax = plt.subplots(1, 1, figsize=(sq_s, sq_s))
    sns.scatterplot(
        x="Degree",
        y="index",
        hue="Disease",
        s=200,
        marker="o",
        data=rank_progressions,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Hyperarc Degree", fontsize=15)
    ax.set_ylabel("Progression", fontsize=15)
    ax.set_title("Important Disease Progression Pathways", fontsize=15)
    ax.set_xticks(np.arange(1, max_d + 1))
    ax.set_yticks(np.arange(top_n))
    ax.set_xticklabels(np.arange(1, max_d + 1), fontsize=15)
    ax.set_yticklabels(np.arange(1, top_n + 1), fontsize=15)
    ax.axis([0, max_d + 1, top_n, -1])
    ax.grid("on")

    for i, dis in enumerate(rank_progressions.Disease):
        row = rank_progressions.iloc[i]
        deg, idx = (row["Degree"], row["index"])
        n_dis = len(dis)
        plt.annotate(
            dis,
            (deg - (0.1 / 10) * (n_dis / 2), idx - 0.15),
            fontsize=int(1.25 * sq_s),
            rotation=40,
        )

    for i, deg_lst in enumerate(prog_idx_lists):
        for j, idx_list in enumerate(deg_lst):
            for idx in idx_list:
                plt.plot(
                    [i + 1, i + 2],
                    [j, idx],
                    linestyle=linestyle_lsts[j % len(linestyle_lsts)],
                    c=prog_colours[i],
                    linewidth=2,
                )

    plt.show()
    return plt


def suc_trans_matrix(
    inc_mat_tail,
    inc_mat_head,
    edge_weights,
    node_degree_tail,
    mort_colarr,
    file_name=None,
):
    """
    Create a colour-coded matrix which gives the probability of transitioning from one node to
    another. Here we give the successor transition matrix.

    Args:
        inc_mat_tail (np array) : Incidence matrix of tails of hyperarcs.

        inc_mat_head (np array) : Incidence matrix of heads of hyperarcs.

        edge_weights (list) : List of hyperedge weights.

        node_degree_tail (list) : List of the node degree of the hyperarc tails.

        mort_colarr (list) : List of the names of the diseases and mortality.

        file_name (str) : Directory and name of where to store figure. If None, then does not save.

    Returns:
        sns figure : Matrix of size = number of disease nodes + number of mortality nodes.
    """

    fig = plt.figure()
    P_node = centrality.comp_ptm_std(
        inc_mat_tail,
        inc_mat_head,
        edge_weights,
        node_degree_tail,
        None,
        "successor",
        eps=0.00001,
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(P_node, cmap="inferno", ax=ax, annot=True, annot_kws={"fontsize": 8})
    ax.set_yticklabels(mort_colarr, fontsize=10, rotation=0)
    ax.set_xticklabels(mort_colarr, fontsize=10, rotation=45)
    ax.set_title(
        "Transition Matrix for Random Node Walk (Successor Detection)",
        fontsize=18,
        pad=20,
    )
    ax.set(xlabel="End Node", ylabel="Start Node")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.show()
    if file_name is not None:
        fig.savefig(file_name)


def pred_trans_matrix(
    inc_mat_tail,
    inc_mat_head,
    edge_weights,
    node_degree_head,
    edge_degree_tail,
    mort_colarr,
    file_name=None,
):
    """
    Create a colour-coded matrix which gives the probability of transitioning from one node to
    another. Here we give the predecessor transition matrix.

    Args:
        inc_mat_tail (np array) : Incidence matrix of tails of hyperarcs.

        inc_mat_head (np array) : Incidence matrix of heads of hyperarcs.

        edge_weights (list) : List of hyperedge weights.

        node_degree_tail (list) : List of the node degree of the hyperarc tails.

        mort_colarr (list) : List of the names of the diseases and mortality.

        file_name (str) : Directory and name of where to store figure. If None, then does not save.

    Returns:
        sns figure : Matrix of size = number of disease nodes + number of mortality nodes.
    """

    fig = plt.figure()
    P_node = centrality.comp_ptm_std(
        inc_mat_tail,
        inc_mat_head,
        edge_weights,
        node_degree_head,
        edge_degree_tail,
        rep="predecessor",
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(P_node, cmap="inferno", ax=ax, annot=True, annot_kws={"fontsize": 8})
    ax.set_yticklabels(mort_colarr, fontsize=10, rotation=0)
    ax.set_xticklabels(mort_colarr, fontsize=10, rotation=45)
    ax.set_title(
        "Transition Matrix for Random Node Walk (Predeccessor Detection)",
        fontsize=18,
        pad=20,
    )
    ax.set(xlabel="Start Node", ylabel="End Node")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.show()
    if file_name is not None:
        fig.savefig(file_name)
