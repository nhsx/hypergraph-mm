import datetime as dt
import os
import sys
import time as t

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import seaborn as sns

import centrality_utils
import utils
import weight_functions

###############################################################################
# 1. CREATE DIRECTED DATA
###############################################################################


def filter_data(
    data,
    columns,
    all_columns,
    rem_healthy=False,
    exclusive=False,
    rem_resbreaks=False,
    date_index=None,
    verbose=True,
):
    """
    Using user-specified flags, filter individuals according to condition
    dates, residential breaks and disease exclusivity.

    INPUTS:
    -----------------------
        data (pd.DataFrame) : Dataframe of individuals as rows and demographic,
        death and disease columns. All disease entries must be datetime
        instances.

        columns (list) : List of disease strings used to subset particular
        diseases in data.

        all_columns (list) : List of all diseases.

        rem_healthy (bool) : Flag to remove individuals with none of the
        diseases specified in columns.

        exclusive (bool) : Flag to only include individuals that only ever had
        a distinct subset of the diseases in columns, i.e. anyone who ever had
        other diseases not mentioned in columns are excluded.

        rem_resbreaks (bool) : Flag to remove individuals who moved from Wales,
        i.e. those tagged with "Residency Break".

        date_index (3-tuple) : 3-tuple of integers to represent year, month
        and day for cut-off when filtering individuals. This will remove
        individuals who have any condition dates prior to the date specified.

        verbose (bool) : Flag to print out progress information during and
        summary statistics after creating the dataset.

    RETURNS:
    ----------------------
        data_df (pd.DataFrame) : Filtered disease date-time DataFrame of
        individuals.

        demo_df (pd.DataFrame) : Filtered demographic DataFrame of individuals.
    """
    # Slice disease dataframe to only include FIRST observational dates and
    # create demographic dataframe allowing option to remove individuals who
    # moved during cohort coverage. Rename columns accordingly
    # Difference between all charlson diseases and diseases specified
    diff_diseases = list(set(all_columns) - set(columns))
    data_df = data[[f"FIRST_{col}" for col in columns]]
    N_obs = data_df.shape[0]
    data_df.columns = columns

    # Extract demographic data
    death_cols = ["DOD", "ADDE_DOD", "ADDE_DOD_REG_DT", "COHORT_END_DATE"]
    demo_cols = [
        "GNDR_CD",
        "AGE_AT_INCEPTION",
        "WIMD2011_QUINTILE_INCEPTION",
        "COHORT_END_DESC",
        "COHORT_END_DATE",
    ]
    demo_df = data[demo_cols + death_cols[:-1]]
    demo_names = ["SEX", "AGE", "DEPR", "COHORT_END_DESC", "COHORT_END_DATE"]

    # Remove those who left cohort due to residential break
    if rem_resbreaks:
        if verbose:
            print(
                (
                    "Removing individuals who left cohort intermittently "
                    "for a residency break..."
                )
            )
        demo_df = demo_df.loc[data["COHORT_END_DESC"] != "Residency break"].reset_index(
            drop=True
        )
        data_df = data_df.loc[data["COHORT_END_DESC"] != "Residency break"].reset_index(
            drop=True
        )
        if verbose:
            N_removed = N_obs - demo_df.shape[0]
            print(
                (
                    f"Removed {N_removed} ({round(100*N_removed/N_obs,1)}%) "
                    "individuals."
                )
            )

    # If date string specified, filter data by removing individuals who have
    # any observed condition date earlier than date specified
    N_obs = demo_df.shape[0]
    if date_index is not None:
        y, m, d = date_index
        time_threshold = dt.date(y, m, d)
        if verbose:
            date_string = str(d) + "/" + str(m) + "/" + str(y)
            print(
                ("\nRemoving individuals whose condition dates < " f"{date_string}...")
            )
        date_bool = np.all((data_df > time_threshold) | (pd.isnull(data_df)), axis=1)
        data_df = data_df.loc[date_bool].reset_index(drop=True)
        demo_df = demo_df.loc[date_bool].reset_index(drop=True)
        if verbose:
            N_removed = N_obs - demo_df.shape[0]
            print(
                (
                    f"Removed {N_removed} ({round(100*N_removed/N_obs,1)}%) "
                    "individuals."
                )
            )

    # Convert to binary numpy array and remove those without any conditions
    # specified in columns
    # This is only flagged if specifically rem_healthy = True or if diseases
    # specified is a subset of all disease nodes
    N_obs = demo_df.shape[0]
    data_binmat = np.array(data_df.astype(bool).astype(int))
    if rem_healthy or diff_diseases != []:
        if verbose:
            print(
                (
                    "\nRemoving individuals without any condition in disease "
                    "nodes specified..."
                )
            )
        dir_ind = np.where(data_binmat.sum(axis=1) > 0)[0]
        data_df = data_df.loc[dir_ind].reset_index(drop=True)
        demo_df = demo_df.loc[dir_ind].reset_index(drop=True)
        data_binmat = data_binmat[dir_ind]
        if verbose:
            N_removed = N_obs - demo_df.shape[0]
            print(
                (
                    f"Removed {N_removed} ({round(100*N_removed/N_obs,1)}%) "
                    "individuals."
                )
            )

    # If only including individuals that only ever had some subset of those
    # diseases in columns
    N_obs = data_binmat.shape[0]
    if exclusive:
        if verbose:
            print(
                (
                    "\nRemoving individuals whose final multimorbidity set "
                    "are a subset of those diseases provided..."
                )
            )

        # If not all diseases specified then nobody is excluded, otherwise
        # look at binary flag data for all non-specified diseases and remove
        # anyone who has been observed to have any of these non-specified
        # diseases.
        if diff_diseases != []:
            diff_columns = [f"FIRST_{col}" for col in diff_diseases]
            all_data_binmat = np.array(data[diff_columns].astype(bool).astype(int))[
                dir_ind
            ]
            exc_ind = np.where(all_data_binmat.sum(axis=1) == 0)[0]
            data_df = data_df.loc[exc_ind].reset_index(drop=True)
            demo_df = demo_df.loc[exc_ind].reset_index(drop=True)
            data_binmat = data_binmat[exc_ind]

        if verbose:
            N_removed = N_obs - demo_df.shape[0]
            print(
                (
                    f"Removed {N_removed} ({round(100*N_removed/N_obs,1)}%) "
                    "individuals."
                )
            )

    # Remove unnecessary columns from demographic DataFrame and rename columns
    demo_df = demo_df[demo_cols]
    demo_df.columns = demo_names

    return data_df, demo_df, data_binmat


def process_individuals(data_arr, data_binmat, verbose=False):
    """
    Work out each individual's ordered condition set and identify which
    individuals have a single condition, a clean progression of multiple
    conditions or a progression of multiple conditions with duplicates.

    INPUTS:
    -----------------------
        data_arr (np.array, dtype=datetime64) : Array whose rows represent
        individuals and column entries represent dates where conditions were
        observed.

        data_binmat (np.array, dtype=np.uint8) : Binary flag matrix of
        data_arr.

        verbose (bool) : Level of verbosity.

    RETURNS:
    ---------------
        data_df (pd.DataFrame) : DataFrame of DateTime instances for
        individuals, all non-NA entries represent the first date a disease was
        observed by the healthcare system.

        demo_df (pd.DataFrame) : Demographic data on the individuals.

        ALL_OBS (np.array, dtype=np.uint8) :  Same as data_df but converted to
        binary and a numpy array.

        ALL_CONDS (np.array, dtype=np.int8) : Ordered condition indexes
        (according to column index of ALL_OBS).

        ALL_IDX (np.array, dtype=np.int8) : indexes of ALL_CONDS to represent
        1-duplicates. Rows of -1 represent clean progressions for individuals
        and a -2 in the first column represent those individuals with only 1
        condition.

        ALL_OGIDX (np.array, dtype=np.int64) : Original indices of individuals
        from input DataFrame for reference.
    """

    # Initialise containers for individuals with clean progression and those
    # with duplicate progression, the duration between disease observations
    # and order of observed conditions
    if verbose:
        print("\nCategorising individual progressions...")
    clean_traj, clean_cond = [], []
    dupl_traj, dupl_dur, n_dupls, dupl_cond = [], [], [], []
    single_traj, single_cond = [], []

    # Organise individuals into lists where they either had a clean
    # observational disease progression, a disease progression where some
    # conditions were observed at the same time, or those who only had a
    # single condition observed through their time interacting with
    # healthcare system
    for i, obs in enumerate(data_arr):
        # Locate which conditions were observed, extract number of conditions
        # and remove None instances
        ind_cond_idx = np.where(~pd.isnull(obs))[0]
        ind_ndisease = len(ind_cond_idx)
        ind_cond_dates = obs[ind_cond_idx]

        # Check if individual ha single disease or not
        if ind_ndisease > 1:
            # Sort dates and diseases
            sort_date_idx = np.argsort(ind_cond_dates)
            sort_dates = ind_cond_dates[sort_date_idx]
            sort_cond = ind_cond_idx[sort_date_idx]

            # Compute observational disease date difference
            unique_diseases = np.unique(sort_dates)
            n_unique_diseases = unique_diseases.shape[0]
            disease_diffs = (sort_dates[1:] - sort_dates[:-1]).astype(int)

            # If any of these differences are 0, then individual appended to
            # duplicate disease progression list. Otherwise, add individual to
            # clean disease progression list
            if n_unique_diseases != ind_ndisease:
                dupl_traj.append(i)
                n_dupls.append(ind_ndisease - n_unique_diseases)
                dupl_dur.append(disease_diffs)
                dupl_cond.append(sort_cond)
            else:
                clean_traj.append(i)
                clean_cond.append(sort_cond)
        else:
            # If individual has only 1 condition
            single_traj.append(i)
            single_cond.append(ind_cond_idx)

    # Convert single to array
    single_traj = np.asarray(single_traj)
    n_single_inds = single_traj.shape[0]
    single_obs_binmat = data_binmat[single_traj]

    # Convert clean to array
    clean_traj = np.asarray(clean_traj)
    n_clean_inds = clean_traj.shape[0]
    clean_obs_binmat = data_binmat[clean_traj]

    if verbose:
        print("Processing individuals with duplicates...")

    # Convert to array
    dupl_traj = np.asarray(dupl_traj)
    n_all_dupl_inds = dupl_traj.shape[0]

    # Work out where durations between First observations of conditions are 0
    # (i.e. duplications) in individual's disease prgression and split by
    # n-duplicate, n = 1,..., 5
    dupl_seq_loc = [np.where(lst == 0)[0] for lst in dupl_dur]
    dupl_seq_split = [
        np.array([seq for seq in dupl_seq_loc if len(seq) == i], dtype=np.int8)
        for i in np.unique(n_dupls)
    ]
    dupl_seq_idx = [
        [i for i, seq in enumerate(dupl_seq_loc) if len(seq) == j]
        for j in np.unique(n_dupls)
    ]

    # Using the index split dup_seq_idx, perform the same split for the
    # individuals' conditions and durations
    dupl_cond_split = [[dupl_cond[idx] for idx in lst] for lst in dupl_seq_idx]

    # We will always keep the 1-duplicates
    dupl_obs_ogidx = dupl_traj[dupl_seq_idx[0]]
    dupl_obs_binmat = data_binmat[dupl_obs_ogidx]
    dupl_obs_conds = dupl_cond_split[0]
    dupl_obs_idx = list(dupl_seq_split[0])

    # Loop through remaining individuals to extract those with multiple
    # 1-duplicates
    for i in range(1, max(n_dupls)):

        # Take difference of locations where duplicates were observed
        dup_seq = dupl_seq_split[i]
        dup_seq_diff = np.diff(dup_seq, axis=1)

        # Find individuals who only have 1-duplicates, i.e. the location of
        # their duplicates are not consecutive or else that would mean their
        # duplicate would be of multiple diseases
        dup_seq_1dupls = np.argwhere((dup_seq_diff != 1).all(axis=1))[:, 0]

        # If detected any, extract observation index, binary flag, ordered
        # condition and indexes of 1-duplicates
        if dup_seq_1dupls.sum() > 0:
            ind_1dupls_og = dupl_traj[
                np.array([dupl_seq_idx[i][j] for j in dup_seq_1dupls])
            ]
            obs_1dupls_binmat = data_binmat[ind_1dupls_og]
            conds_1dupls = [dupl_cond_split[i][j] for j in dup_seq_1dupls]
            condidx_1dupls = list(dupl_seq_split[i][dup_seq_1dupls].astype(np.int8))

            dupl_obs_ogidx = np.concatenate([dupl_obs_ogidx, ind_1dupls_og], axis=0)
            dupl_obs_binmat = np.concatenate(
                [dupl_obs_binmat, obs_1dupls_binmat], axis=0
            )
            dupl_obs_conds += conds_1dupls
            dupl_obs_idx += condidx_1dupls

    n_1dupl_inds = dupl_obs_binmat.shape[0]
    ALL_CONDS = single_cond + clean_cond + dupl_obs_conds
    ALL_IDX = (
        n_single_inds * [np.array([-2], dtype=np.int8)]
        + n_clean_inds * [np.array([-1], dtype=np.int8)]
        + dupl_obs_idx
    )
    ALL_OBS = np.concatenate(
        [single_obs_binmat, clean_obs_binmat, dupl_obs_binmat], axis=0
    )
    ALL_OGIDX = np.concatenate([single_traj, clean_traj, dupl_obs_ogidx], axis=0)

    if verbose:
        print(f"\nNumber of individuals with single disease: {n_single_inds}")
        print(f"Number of individuals with clean progressions: {n_clean_inds}")
        print(
            ("Number of individuals with duplicate progressions: " f"{n_all_dupl_inds}")
        )
        print(
            (
                "    Individuals with duplicates valid for analyses: "
                f"{n_1dupl_inds} ({round(100*n_1dupl_inds/n_all_dupl_inds,1)}"
                "%)"
            )
        )

    return ALL_CONDS, ALL_IDX, ALL_OBS, ALL_OGIDX


def create_data(
    data,
    columns,
    all_columns,
    rem_healthy=False,
    exclusive=False,
    rem_resbreaks=False,
    date_index=None,
    verbose=True,
):
    """
    Given the Pandas DataFrame of individuals with information on their
    multimorbidity, demographics and recorded death/end-of-cohort entries,
    construct dataset of individuals for a directed hypergraph model.

    INPUTS:
    -------------------
        data (pd.DataFrame) : Dataframe of individuals as rows and demographic,
        death and disease columns. All disease entries must be datetime
        instances.

        columns (list) : List of disease strings used to subset particular
        diseases in data.

        all_columns (list) : List of all disease strings.

        date_index (3-tuple) : 3-tuple of integers to represent year, month
        and day for cut-off when filtering individuals. If None, then don't
        filter.

        rem_healthy (bool) : Flag to remove individuals with none of the
        diseases specified in columns.

        rem_resbreaks (bool) : Flag to remove individuals who moved from Wales,
        i.e. those tagged with "Residency Break"

        exclusive (bool) : Flag to only include individuals that only ever had
        a distinct subset of the diseases in columns, i.e. anyone who ever had
        other diseases not mentioned in columns are excluded.

        verbose (bool) : Flag to print out progress information during and
        summary statistics after creating the dataset

    RETURNS:
    ---------------
        data_df (pd.DataFrame) : DataFrame of DateTime instances for
        individuals, all non-NA entries represent the first date a disease was
        observed by the healthcare system.

        demo_df (pd.DataFrame) : Demographic data on the individuals.

        ALL_OBS (np.array, dtype=np.uint8) :  Same as data_df but converted to
        binary and a numpy array.

        ALL_CONDS (np.array, dtype=np.int8) : Ordered condition indexes
        (according to column index of ALL_OBS).

        ALL_IDX (np.array, dtype=np.int8) : indexes of ALL_CONDS to represent
        1-duplicates. Rows of -1 represent clean progressions for individuals
        and a -2 in the first column represent those individuals with only 1
        condition.

        ALL_OGIDX (np.array, dtype=np.int64) : Original indices of individuals
        from input DataFrame for reference.
    """
    # store original observations and disease number
    N_original = data.shape[0]
    N_obs, N_cols = data.shape
    N_diseases = len(columns)

    # Print out diagnostics during runtime
    if verbose:
        st = t.time()
        print(f"Building dataset with diseases:\n{columns}")
        print("\nFiltering dataset according to dataset specifications...")

    # Filter data according to dataset specifications
    data_df, demo_df, data_binmat = filter_data(
        data,
        columns,
        all_columns,
        rem_healthy,
        exclusive,
        rem_resbreaks,
        date_index,
        verbose,
    )

    # Convert dates to numpy type datetime64 and extract number of diseases
    # and observations
    data_arr = np.array(data_df).astype("datetime64[D]")

    # process individuals to build ordered condition sets and duplicates
    output = process_individuals(data_arr, data_binmat, verbose)
    ALL_CONDS, ALL_IDX, ALL_OBS, ALL_OGIDX = output

    # Convert ALL_IDX and ALL_CONDS to their worklists
    if verbose:
        print("\nComputing worklists...")
    ALL_CONDS = utils.compute_worklist(ALL_CONDS, N_diseases)
    ALL_IDX = utils.compute_worklist(ALL_IDX, N_diseases // 2)
    output = (ALL_OBS, ALL_CONDS, ALL_IDX, ALL_OGIDX)

    # DataFrame output for demographics and disease flags
    data_df = data_df.iloc[ALL_OGIDX].reset_index(drop=True)
    demo_df = demo_df.iloc[ALL_OGIDX].reset_index(drop=True)
    all_data_df = (demo_df, data_df)

    # Output information on dataset
    New_obs = ALL_OBS.shape[0]
    if verbose:
        en = t.time()
        elapsed = np.round(en - st, 2)
        print(f"\nCompleted in {elapsed} seconds.")
        print(f"Original number of individuals: {N_original}.")
        print(
            (
                f"Total individuals available for analysis: {New_obs} "
                f"({round(100*New_obs / N_original,1)}% of original dataset)."
            )
        )
        print(f"Number of disease nodes: {N_diseases}")

    return all_data_df, output


###############################################################################
# 2. BUILD DIRECTED MODEL
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def compute_directed_model(
    data, ordered_cond, ordered_idx, progression_func, progression
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
    disease populations. The 2-dimensional numpy array hyperarc_prev orders
    the columns as head node entries and the rows as different tail node
    combinations. Both arrays are of type np.float64.

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

        progression (int) : This is the type of progression system used. If 0
        then we build the progression set by aggregating each ordered
        condition, if 1 we build the power set progression set, if 2 we build
        a simple directed graph, i.e. only hyperarcs of degree 2.

    RETURNS:
    --------------------
        inc_mat (2d np.array, dtype=np.int8) : Signed, directed incidence
        matrix which store tail members of a hyperarc as -1's in a row and the
        head member as 1. Can be split into tail- and head-incidence matrices

        hyperarc_worklist (2d np.array, dtype=np.int8) : Hyperarcs, ordered as
        they are seen in inc_mat, stored as the disease index followed by a
        stream if -1's for compatability with numba.

        hyperarc_prev (2d np.array, dtype=np.float64) : Prevalence array for
        hyperarcs. Entry [i,j] is the prevalence for the hyperarc whose tail
        members binary encoding is i and whose head member disease index is j.

        hyperedge_prev (1d np.array, dtype=np.float64) : Prevalence array for
        corresponding parent hyperedges of the hyperarcs. Entry [i] is the
        prevalence for the hyperedge whose disease set's binary encoding is
        equal to i.

        node_prev (1d np.array, dtype=np.float64) : Prevalence array for the
        nodes. The prevalence for a disease node is split into it's tail- and
        head-components and prevalences, for cample entries [i] and
        [i+n_diseases] is the node prevalence for the tail- and head-component
        of D_i, respectively.
    """
    # INITIALISATION OF PREVALENCE ARRAYS, MORTALITY, INCIDENCE MATRICES,
    # ETC.

    # Number of diseases and observations
    n_diseases = data.shape[1]
    n_obs = data.shape[0]

    # deduce maximum number of hyperarcs and number of hyperedges, depending
    # on the progression type
    # If building directed hypergraph model
    if progression != 2:
        max_hyperarcs = utils.N_max_hyperarcs(n_diseases, b_hyp=True)
        max_hyperedges = utils.N_max_hyperedges(n_diseases)
        simple_flag = 0

    # If building simple directed graph model
    else:
        max_hyperarcs = 2 * round(utils.N_choose_k(n_diseases, 2))
        max_hyperedges = 2 ** (n_diseases - 1) + 2 ** (n_diseases - 2) + 1
        simple_flag = 1

    # Initialise hyperarc work list, hyperarc and node prevalence arrays and
    # directed incidence matrix. Dummy vector used to fill row of incidence
    # matrix as initialised as empty
    hyperedge_prev = np.zeros((max_hyperedges), dtype=np.float64)
    hyperarc_worklist = np.empty(shape=(max_hyperarcs, n_diseases), dtype=np.int8)
    hyperarc_prev = np.zeros((max_hyperedges, n_diseases), dtype=np.float64)
    node_prev = np.zeros((2 * n_diseases), dtype=np.float64)
    pop_prev = np.zeros(n_diseases, dtype=np.int64)
    inc_mat = np.empty(shape=(max_hyperarcs, n_diseases), dtype=np.int8)
    dummy_vec = np.zeros(n_diseases, dtype=np.int8)

    # Loop over each individuals binary flag vector representing undirected
    # multimorbidity set
    n_row = 0
    for ii in range(n_obs):

        # EXTRACT INDIVIDUAL'S CONDITION SET, OBSERVED ORDERING, DUPLICATES
        # AND INCREMENT SINGLE-SET POPULATIONS

        # Select binary realisations of individual, order of morbidities
        # representing disease progression, potential indices of duplicates
        ind_cond = ordered_cond[ii]
        ind_idx = ordered_idx[ii]

        # # Add individual to prevalence counter for single condition sets and
        # tail/head node count
        n_ind_cond = ind_cond[ind_cond != -1].shape[0]
        node_weight = n_ind_cond - 1
        for c in ind_cond[:n_ind_cond]:
            pop_prev[c] += 1

        # Check if individual only has 1 disease, if not, then continue to
        # deduce their hyperarcs. If they are, move to next individual
        min_indidx = ind_idx[0]
        if min_indidx != -2:

            # COMPUTE DISEASE NODE AND SINGLE-PROGRESSION PREVALENCE

            # If individual doesn't have a 2-duplicate at end of progression
            # then contribution remains unitary.
            max_indidx = ind_idx.max()
            if max_indidx != n_ind_cond - 1:
                for i in range(n_ind_cond - 1):
                    node_prev[ind_cond[i]] += 1.0
                node_prev[n_diseases + ind_cond[n_ind_cond - 1]] += node_weight

            # If individual has duplicate at end of progression then
            # contribution to last two diseases is halved to take into account
            # hyperarcs where these diseases have swapped their tail and
            # head role
            else:
                for i in range(n_ind_cond - 2):
                    node_prev[ind_cond[i]] += 1.0
                node_prev[ind_cond[n_ind_cond - 2 : n_ind_cond]] += 0.5
                node_prev[n_diseases + ind_cond[n_ind_cond - 2 : n_ind_cond]] += (
                    node_weight / 2
                )

            # First disease individual had contributes to single-set hyperedge
            # prevalence. If duplicate exists at beginning of progression, then
            # hyperedge prevalence for single-set disease needs halved for
            # first and second condition .
            which_min = int(min_indidx == 0)
            start_dupl = [1, 2][which_min]
            hyp_cont = [1.0, 0.5][which_min]
            hyp_idx = ind_cond[:start_dupl].astype(np.int64)
            hyperedge_prev[2**hyp_idx] += hyp_cont

            # COMPUTE PROGRESSION SET

            # Compute progression set based on ordering
            prog_set = progression_func(ind_cond, ind_idx, undirected=False)
            n_prog_obs = len(prog_set)

            # Work out number of conditions in each progression element of
            # prog_set and hyperarc contribution factor
            n_conds_prog = prog_set.shape[1] - (prog_set == -1).sum(axis=1)
            arc_cont_dirhyp = 1 / np.bincount(n_conds_prog)[2:]
            arc_cont_simple = np.bincount(prog_set[:, 0])

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

                # For directed hypergraph, contribution is split equally for
                # each same-degree hyperarc, i.e. each individual contributes
                # a total of 1.0 per degree
                # For directed graph, contribution is split equally for every
                # same-tail hyperarc, i.e. each individual contributes a total
                # of 1.0 per progression with the same tail
                contribution = [
                    arc_cont_dirhyp[n_conds - 2],
                    1 / arc_cont_simple[elem[0]],
                ][simple_flag]

                # IF UNOBSERVED PROGRESSION HYPERARC, ADD TO INCIDENCE
                # MATRIX AND HYPERARC WORKLIST OTHERWISE, CONTINUE TO
                # INCREMENT CONTRIBUTIONS

                # Check if hyperarc has been seen before, if not then it
                # should still be 0 and needs to be added to incidence matrix
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
                hyperedge_prev[bin_headtail] += contribution
                hyperarc_prev[bin_tail, head_node] += contribution

        # If individual only has 1 disease, then half contribution to head and
        # tail disease node and contribute to single-set disease prevalence
        else:

            # CONTRIBUTE PREVALENCE FOR INDIVIDUALS WITH ONLY 1 CONDITIONS

            # If the individual only had one condition add prevalence to top
            # row of hyperarc
            single_cond = ind_cond[0]
            hyperarc_prev[0, single_cond] += 1.0
            hyperedge_prev[2**single_cond] += 1.0
            node_prev[single_cond] += 1.0
            node_prev[n_diseases + single_cond] += 1.0

    return (
        inc_mat[:n_row],
        hyperarc_worklist[:n_row],
        hyperarc_prev,
        hyperedge_prev,
        node_prev,
        pop_prev,
    )


###############################################################################
# 4. BUILD MODEL AND COMPUTE EDGE WEIGHTS
###############################################################################


def compute_weights(
    binmat,
    conds_worklist,
    idx_worklist,
    colarr,
    contribution_type,
    weight_function,
    progression_function,
    dice_type=1,
    verbose=True,
    ret_inc_mat=False,
    plot=True,
    save_images=(False, None),
    plt_dis_prog=None,
):
    """
    This function will take an example dataset of individuals with arbitrary
    diagnoses, whose condition sets are ordered and duplicates are specified.

    The function will then build the prevalence arrays, incidence matrix and
    compute the edge weights for the hyperedges and hyperarcs.

    The function allows specification of which weight function to use and which
    contribution type to use, i.e. exclusive, progression or power set based.

    INPUTS:
    -------------
        binmat (np.array, dtype=np.uint8) : Binary flag matrix whose test
        observations are represented by rows and diseases depresented by
        columns.

        conds_worklist (np.array, dtype=np.int8) : For each individual, the
        worklist stores the ordered set of conditions via their columns
        indexes. For compatability with Numba, once all conditions specified
        per individual, rest of row is a stream if -1's.

        idx_worklist (np.array, dtype=np.int8) : Worklist to specify any
        duplicates. For each individuals, if -2 then they only have 1
        condition, if -1 then they have no duplicates and if they have positive
        integers indexing conds_worklist then it specifies which conditions
        have a duplicate which needs taken care of.

        colarr (np.array, dtype="<U24") : Numpy array of disease column titles

        contribution_type (str) : Type of contribution to hyperedges each
        individual has, i.e. can be "exclusive" or "progression".

        weight_function (func) : Numba-compiled weight function, will be
        version of the overlap coefficient and modified sorensen-dice
        coefficient.

        progression_function (func) : Numba-compiled progression function,
        dictating exactly what hyperarcs, and therefore hyperedges, an
        individual contributes to in the graph. Is either
        utils.compute_progset, utils.compute_single_progset,
        utils.compute_pwset_progset. The first used exclusive progressiion,
        the second constructs a digraph and the third follows a power set
        progression.

        dice_type (int) : Type of Sorensen-Dice coefficient used, either 1
        (Complete) or 0 (Power).

        plot (bool) : Flag to plot hyperedge and hyperarc weights.

        ret_inc_mat (bool) : Flag to return the tail and head incidence
        matrices.

        verbose (bool) : Flag to print updates on computing incidence matrix,
        prevalence arrays and weights.

        save_images (2-tuple) : Save hyperedge, hyperarc and node weight
        graphs. First element of tuple is boolean flag, second element is file
        path to save images.

        plt_dis_prog (str) : Flag to plot hyperedge-hyperarc plot but from a
        seed condition, i.e. only plot top 28 edges edges whose tail includes a
        disease. If None, plots top 28.
    """
    # Number of observations and columns and set up end of progressions
    N_obs, N_diseases = binmat.shape

    if verbose:
        print("Building directed incidence matrix and prevalence arrays...")

    # Depending on progression function, specify type in compute_directed_model
    if progression_function == utils.compute_progset:
        progression = 0
    elif (
        progression_function == utils.compute_pwset_progset
        or contribution_type == "power"
    ):
        progression = 1
    elif progression_function == utils.compute_simple_progset:
        progression = 2

    # Build incidence matrix, prevalence arrays and hyperarc worklists
    st = t.time()
    output = compute_directed_model(
        binmat, conds_worklist, idx_worklist, progression_function, progression
    )
    (
        inc_mat,
        hyperarc_worklist,
        hyperarc_prev,
        hyperedge_prev,
        node_prev,
        pop_prev,
    ) = output

    if verbose:
        print(f"Completed in {round(t.time()-st,2)} seconds.")
        print(f"Incidence matrix shape: {inc_mat.shape}")

    # Set up variables for building weights for hyperedges, hyperarc and nodes,
    # dependent on whether mortality is used
    output = weight_functions.setup_weight_comp(
        colarr,
        colarr,
        binmat,
        node_prev,
        hyperarc_prev,
        hyperarc_worklist,
        weight_function,
        dice_type,
    )
    string_output, hyperarc_output, node_weights, palette = output
    (
        hyperarc_progs,
        hyperarc_weights,
        hyperarc_worklist,
        hyperarc_counter,
    ) = hyperarc_output
    colarr, nodes, dis_dict = string_output

    # Build DataFrame for node weights
    node_weights_df = pd.DataFrame({"node": nodes, "weight": node_weights})

    # Depending on contribution type, specify prevalence array for hyperedges
    # and the hyperedge array to loop through. If "exclusive" just take unique
    # rows of the original input binmat. Prevalence array comes from granular
    # split-and-count of individuals of their final multimorbidity set.
    hyperarc_num_prev = hyperedge_prev.copy()

    columns_idxs = np.arange(N_diseases).astype(np.int8)
    hyperedge_denom = utils.compute_integer_repr(binmat, columns_idxs, colarr)

    if contribution_type == "exclusive":
        hyperedge_num_prev = hyperedge_denom.copy()
        hyperedge_arr = np.unique(binmat, axis=0)

    # If "progression" we take the unique, absolute rows of the directed
    # incidence matrix, concatenated with the self-edges, as these cannot be
    # represented in the directed incidence matrix. Prevalence array comes
    # from compute_directed_model() assuming "progression" contribution.
    elif contribution_type == "progression":
        hyperedge_num_prev = hyperedge_prev.copy()
        hyperedge_arr = np.unique(np.abs(inc_mat), axis=0)
        selfedge_arr = np.eye(N_diseases)
        hyperedge_arr = np.concatenate([selfedge_arr, hyperedge_arr], axis=0)

    # If "power" hyperedge prevalence is computed at runtime inside
    # weight_function, but for numba compatability, we create a zero array for
    # hyperedge_num_prev
    elif contribution_type == "power":
        hyperedge_arr = np.unique(binmat, axis=0)
        hyperedge_num_prev = np.zeros_like(hyperedge_prev, dtype=np.float64)

    # Build worklist of hyperedges, their unique integer representation and
    # their disease set string
    hyperedge_worklist = utils.comp_edge_worklists(
        hyperedge_arr, contribution_type, shuffle=False
    )
    hyperedge_indexes, hyperedge_N = utils.compute_bin_to_int(hyperedge_worklist)
    hyperedge_cols = np.asarray(
        list(
            map(
                lambda col: ", ".join(np.sort(col)),
                map(lambda row: colarr[row[row != -1]], hyperedge_worklist),
            )
        )
    )

    # Sort hyperedges by degree
    sort_hyps = np.argsort(hyperedge_N)
    hyperedge_worklist = hyperedge_worklist[sort_hyps]
    hyperedge_indexes = hyperedge_indexes[sort_hyps]
    hyperedge_N = hyperedge_N[sort_hyps]
    hyperedge_cols = hyperedge_cols[sort_hyps]

    # Depending on function type, define denominator array. If Overlap
    # coefficient we need single-disease total population sizes, i.e. we can
    # use the hyperedge_prev we get from compute_directed_model()
    if weight_function == weight_functions.comp_overlap_coeff:
        denom_prev = pop_prev.copy()

    # If "exclusive" contribution
    elif contribution_type == "exclusive":
        denom_prev = hyperedge_denom.copy()

    # If "power" contribution, need to build prevalence array if we're using
    # the reformulation of the Sorrensen-Dice coefficient
    elif (
        contribution_type == "power"
        and weight_function == weight_functions.modified_sorensen_dice_coefficient
    ):
        hyperedge_num_prev = utils.comp_pwset_prev(binmat, hyperedge_worklist, colarr)
        denom_prev = hyperedge_num_prev.copy()

    # If using the modified sorensen-dice coefficient, this denominator will
    # weight hyperedge/hyperarc prevalences relative to other prevalences of
    # similar disease sets
    else:
        denom_prev = hyperedge_prev.copy()

    # BUILD HYPEREDGE WEIGHTS
    if verbose:
        print("\nComputing hyperedge weights...")
        st = t.time()

    # Updated modified sorensen-dice coefficient combined Power and Complete
    # here
    if weight_function == weight_functions.modified_sorensen_dice_coefficient:
        hyperedge_weights = weight_function(
            hyperedge_worklist,
            hyperedge_N,
            hyperedge_indexes,
            hyperedge_num_prev,
            denom_prev,
            dice_type,
        )

        # If Power dice then add single set disease weights manually
        if dice_type == 0:
            hyperedge_weights[:N_diseases] = binmat.sum(axis=0) / binmat.shape[0]

    # Otherwise, If using overlap coefficient or older versions of the
    # Sorensen-Dice coefficients
    else:
        # Initialise hyperedge weights
        hyperedge_weights = np.empty_like(hyperedge_indexes, dtype=np.float64)
        if weight_function == weight_functions.modified_dice_coefficient_comp:
            hyperedge_counter = 0

        # If Overlap of Power Dice, compute single disease set edge weights as
        # the proportion of individuals with the disease out of all
        # individuals.
        else:
            hyperedge_weights[:N_diseases] = binmat.sum(axis=0) / binmat.shape[0]
            hyperedge_counter = N_diseases

        # Compute hyperedge weights
        hyperedge_weights = weight_functions.compute_hyperedge_weights(
            binmat,
            hyperedge_weights,
            hyperedge_worklist,
            colarr,
            hyperedge_num_prev,
            denom_prev,
            weight_function,
            contribution_type,
            hyperedge_counter,
        )

    # Build dataframe of hyperedge weights and sort if specified
    hyperedge_indexes = np.asarray(hyperedge_indexes, dtype=np.int64)
    hyperedge_weights = np.asarray(hyperedge_weights, dtype=np.float64)
    hyperedge_weights_df = pd.DataFrame(
        {"disease set": hyperedge_cols, "weight": hyperedge_weights}
    )
    hyperedge_weights_df = hyperedge_weights_df[hyperedge_weights_df.weight > 0]

    if verbose:
        print(f"Completed in {round(t.time()-st,2)} seconds.")
        print("\nComputing hyperarc weights...")
        st = t.time()

    # BUILD HYPERARC WEIGHTS
    output = weight_functions.compute_hyperarc_weights(
        hyperarc_weights,
        hyperarc_progs,
        hyperarc_worklist,
        colarr,
        hyperarc_prev,
        hyperarc_num_prev,
        hyperedge_weights,
        hyperedge_indexes,
        hyperarc_counter,
    )
    hyperedge_weights, hyperarc_progs = output

    # Build dataframe of hyperedge weights and sort if specified
    hyperarc_weights_df = pd.DataFrame(
        {"progression": hyperarc_progs, "weight": hyperarc_weights}
    )

    if verbose:
        print(f"Completed in {round(t.time()-st,2)} seconds.")

    # Plot hyperedge, hyperarc and node weights
    if plot:
        n = 28
        sorted_hyperarc_weights_df = hyperarc_weights_df.sort_values(
            by=["weight"], ascending=False
        )
        sorted_hyperedge_weights_df = hyperedge_weights_df.sort_values(
            by=["weight"], ascending=False
        )

        # Just hyperedges
        hyperedge_fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(
            x="weight",
            y="disease set",
            data=sorted_hyperedge_weights_df.iloc[:n],
            ax=ax,
        )
        # ax.set_title("Hyperedge Weights", fontsize=18)
        ax.set_ylabel("Disease Set", fontsize=15)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=15)
        ax.set_xlabel("Weight", fontsize=15)
        hyperedge_fig.tight_layout(pad=0)

        # Just hyperarcs
        hyperarc_fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(
            x="weight",
            y="progression",
            data=sorted_hyperarc_weights_df.iloc[:n],
            ax=ax,
        )
        # ax.set_title("Hyperarc Edge Weights", fontsize=18)
        ax.set_ylabel("Disease Progression", fontsize=15)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=15)
        ax.set_xlabel("Weight", fontsize=15)
        hyperarc_fig.tight_layout(pad=0)

        # Build dataframe of top hyperarc weights to plot and their parent
        # hyperedge weights
        edge_arc_df = pd.DataFrame(
            columns=[
                "progression",
                "hyperarc_weight",
                "disease_set",
                "hyperedge_weight",
                "hyperedge_index",
            ]
        )

        hyperarc_progs = sorted_hyperarc_weights_df["progression"]

        # plt_dis_prog allows user to specify if hyperarc-hyperedge plot only
        # shows top hyperarcs and their parents which include diseases
        # specified in string tuple plt_dis_prog. If None, then plots top 28
        # from all diseases.
        if plt_dis_prog is None:
            top_hyperarcs_df = sorted_hyperarc_weights_df.iloc[:n]
        else:
            # There is the option here to filter top hyperarcs by either the
            # diseases in plt_dis_prog being in the tail only (currently
            # commented out) or allowing them to be in either the tail or head
            # of hyperarcs.
            dis = plt_dis_prog
            # Only in tail
            # title_prog_idx = [
            #   i for i, p in enumerate(hyperarc_progs) if all(
            #       [True if d in p.split(" -> ")[0] else False for d in dis]
            #   )
            # ]

            # In tail and head
            title_prog_idx = [
                i
                for i, p in enumerate(hyperarc_progs)
                if all([True if d in p else False for d in dis])
            ]
            top_hyperarcs_df = (sorted_hyperarc_weights_df.iloc[title_prog_idx]).iloc[
                :n
            ]

        # Initialise the hyperedge counter to fill edge_arc_df with hyperarcs
        # and their parents.
        # Note: the code below is messy and there could well be a much smarter
        # way of doing this, but I somehow decided to do it in an
        # overcomplicated manner...
        hyperedge_counter = 0
        for idx, hyp_arc in top_hyperarcs_df.iterrows():

            # Loop over top hyperarcs according to plt_dis_prog
            # specifications, in each hyperarc extract tail diseases and
            # head disease.
            prog, weight = hyp_arc
            prog_split = prog.split(" -> ")
            prog_tail = list(prog_split[0].split(", "))
            prog_head = prog_split[1]

            # If hyperarc is NOT a self-transition, build hyperedge disease
            # set string title
            if prog_head not in prog_split[0]:
                hyperedge = ", ".join(np.sort(prog_tail + [prog_head]))
                hyperarc_title = ", ".join(prog_tail + [prog_head])

            # Otherwise, the head disease WILL be the same as the tail disease
            # and so hyperedge set is just the disease title,
            # i.e. prog_tail[0] (as prog_tail is a list)
            else:
                hyperedge = prog_tail[0]
                hyperarc_title = prog_tail[0]

            # From the sorted hyperedge weight dataframe, extract the relevant
            # parent. I have a Try-Except block here in case no hyperedge
            # parent is found. This should technically not happen.
            hyperedge_info = sorted_hyperedge_weights_df[
                sorted_hyperedge_weights_df["disease set"] == hyperedge
            ]
            if hyperedge_info.shape[0] != 0:
                edge_weight = hyperedge_info.weight.iloc[0]
                hyperedge_sorted = hyperedge_info["disease set"].iloc[0]
            else:
                edge_weight = 0.0
                hyperedge_sorted = ""

            # If populated edge_arc_df dataframe already, check to see if
            # parent has been seen before by processing a sibling hyperarc in a
            # previous iteration
            if edge_arc_df.disease_set.shape[0] > 0:
                sorted_sets = np.array(
                    [", ".join(np.sort(s.split(", "))) for s in edge_arc_df.disease_set]
                )
                check_edge = edge_weight == np.array(list(edge_arc_df.hyperedge_weight))
                check_dis = hyperedge_sorted == sorted_sets

            # If first entry in edge_arc_df, then no need to check (as it
            # throws an error if we do), so manually set the bools to False
            else:
                sorted_sets = edge_arc_df.disease_set
                check_edge = np.array([False])
                check_dis = np.array([False])

            # check_bool checks the edge weight is the same and so is the
            # disease set of the parent hyperedge to make
            check_bool = check_edge & check_dis
            # If parent hyperedge seen before, extract the dataframe-relative
            # row index for appending the child hyperarc and its parent easily
            # to edge_arc_df
            if np.any(check_bool):
                hyperedge_idx = edge_arc_df[check_bool]["hyperedge_index"].iloc[0]
            # Otherwise, this is a new hyperarc and corresponding parents, so
            # incremement hyperedge_counter
            else:
                hyperedge_idx = hyperedge_counter
                hyperedge_counter += 1

            # Append hyperarc child and hyperedge sibling to edge_arc_df
            row = pd.DataFrame(
                dict(
                    progression=prog,
                    hyperarc_weight=weight,
                    disease_set=hyperarc_title,
                    hyperedge_weight=edge_weight,
                    hyperedge_index=hyperedge_idx,
                ),
                index=[idx],
            )
            edge_arc_df = pd.concat([edge_arc_df, row], axis=0)

        # Superimpose top hyperarcs onto their parent hyperedges
        arc_palette = np.array(
            plt.get_cmap("nipy_spectral")(
                np.linspace(0.1, 0.9, np.unique(edge_arc_df.hyperedge_index).shape[0])
            )
        )
        palette_idxs = np.array(edge_arc_df.hyperedge_index, dtype=np.int8)
        hyphyp_fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        sns.barplot(
            x="hyperarc_weight",
            y="progression",
            data=edge_arc_df,
            ax=ax,
            palette=arc_palette[palette_idxs],
        )
        ax.set_ylabel("Disease Progression (Hyperarc)", fontsize=18)
        ax.set_yticklabels(edge_arc_df.progression, fontsize=15)
        # ax.set_title("Superimposed Hyperedge/Hyperarc Weights", fontsize=18)
        ax.set_xlabel("Hyperedge (shaded)/Hyperarc (solid) Weight", fontsize=18)
        ax1 = ax.twinx()
        sns.barplot(
            x="hyperedge_weight",
            y="disease_set",
            data=edge_arc_df,
            ax=ax1,
            alpha=0.35,
            palette=arc_palette[palette_idxs],
        )
        ax.grid("on")
        if plt_dis_prog is None:
            ax1.set_yticklabels(edge_arc_df.disease_set, fontsize=15)
            ax1.set_ylabel("Disease Set (Hyperedge)", fontsize=18)
        else:
            ax1.set_yticklabels([], fontsize=15)
            ax1.set_ylabel(None, fontsize=18)
        ax.tick_params(labelsize=15)
        hyphyp_fig.tight_layout(pad=0)

        # Node weights
        if node_weights_df.shape[0] <= 30:
            figsize = (10, 8)
        else:
            figsize = (12, 15)
        node_fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.barplot(x="weight", y="node", data=node_weights_df, ax=ax, palette=palette)
        ax.axvline(x=1 / 2, ymin=0, ymax=10, c="r", linestyle="--")
        # ax.set_title("Node Weights", fontsize=18)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=15)
        ax.set_ylabel("Tail/Head Nodes", fontsize=15)
        ax.set_xlabel("Weight", fontsize=15)
        node_fig.tight_layout(pad=0)

        if save_images[0]:
            if save_images[1] is None:
                fpath = sys.path[0]
            else:
                fpath = save_images[1]
            hyperedge_fig.savefig(os.path.join(fpath, "hyperedge_weights.png"))
            hyperarc_fig.savefig(os.path.join(fpath, "hyperarc_weights.png"))
            hyphyp_fig.savefig(os.path.join(fpath, "hyperedge_arc_weights.png"))
            node_fig.savefig(os.path.join(fpath, "node_weights.png"))

    # If returning incidence matrix
    if ret_inc_mat:
        output = inc_mat, (
            hyperedge_weights_df,
            hyperarc_weights_df,
            node_weights_df,
        )
    elif ret_inc_mat is None:
        output = (
            inc_mat,
            (hyperedge_weights_df, hyperarc_weights_df, node_weights_df),
            (hyperedge_prev, hyperarc_prev, node_prev),
        )
    else:
        output = hyperedge_weights_df, hyperarc_weights_df, node_weights_df

    return output


###############################################################################
# 5. ORGANISE VARIABLES FOR DOWNSTREAM ANALYSIS
###############################################################################


def setup_vars(inc_mat, N_diseases, hyperarc_weights, hyperarc_titles, node_weights):
    """
    This function organises the directed incidence matrix into it's tail and
    head components, taking into account the inclusion of mortality and
    self-edges. Node and edge, tail and head degrees are calculated and then
    all data is ordered according to hyperarc degree.
    """
    # Number of edges and nodes in directed hypergraph and convert node and
    # edge weights to arrays
    N_edges, N_nodes = inc_mat.shape
    edge_weights = np.asarray(hyperarc_weights, dtype=np.float64)
    hyperarc_titles = np.asarray(hyperarc_titles)
    node_weights = np.asarray(node_weights, dtype=np.float64)

    # Tail incidence matrix are where entries in inc_mat are negative
    inc_mat_tail = inc_mat.T.copy()
    inc_mat_tail[inc_mat_tail > 0] = 0
    inc_mat_tail = np.abs(inc_mat_tail)

    # Head incidence matrix are where entries in inc_mat are positive
    inc_mat_head = inc_mat.T.copy()
    inc_mat_head[inc_mat_head < 0] = 0

    # If mort_type is None then create self-loops (self-edges)
    selfedge_component = np.eye(N_nodes)[:N_diseases]

    # Concatenate self-edge/mortality hyperarcs to tail and head incidence
    # matrices
    inc_mat_tail = np.concatenate([selfedge_component.T, inc_mat_tail], axis=1)
    inc_mat_head = np.concatenate([selfedge_component.T, inc_mat_head], axis=1)

    # Calculate tail and head, node and edge degrees
    node_degs, edge_degs = centrality_utils.degree_centrality(
        inc_mat_tail, inc_mat_head, edge_weights, None
    )
    node_degree_tail, node_degree_head = node_degs
    edge_degree_tail, edge_degree_head = edge_degs

    # Compute edge degree valencies to use to sort data
    edge_valency = centrality_utils.degree_centrality(
        inc_mat_tail, inc_mat_head, edge_weights, None
    )[1][0]
    sort_edges = np.argsort(edge_valency)

    # Sort data
    inc_mat_tail = inc_mat_tail[:, sort_edges]
    inc_mat_head = inc_mat_head[:, sort_edges]
    edge_degree_tail = edge_degree_tail[sort_edges]
    edge_degree_head = edge_degree_head[sort_edges]
    edge_weights = edge_weights[sort_edges]
    hyperarc_titles = hyperarc_titles[sort_edges]

    # Format output
    edge_degs = (edge_degree_tail, edge_degree_head)
    output = (
        (inc_mat_tail, inc_mat_head),
        (edge_weights, hyperarc_titles),
        node_weights,
        node_degs,
        edge_degs,
    )

    return output
