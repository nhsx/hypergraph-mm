import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import time as t
import matplotlib.pyplot as plt
import numba
import string

from hypmmm import utils, weight_functions, centrality_utils


##########################################################################################################
# 1. CREATE DIRECTED DATA
##########################################################################################################
def create_data(
    data,
    columns,
    all_columns,
    date_index=None,
    verbose=True,
    rem_resbreaks=True,
    incl_mortality=False,
    exclusive=False,
):
    """
    Given the Pandas DataFrame of individuals with information on their multimorbidity, demographics and
    recorded death/end-of-cohort entries, construct dataset of individuals for a directed hypergraph model.

    Args:
        data (pd.DataFrame) : Dataframe of individuals as rows and demographic, death and disease columns.
        All disease entries must be datetime instances.

        columns (list) : List of disease strings used to subset particular diseases in data.

        all_columns (list) : List of all disease strings.

        date_index (3-tuple) : 3-tuple of integers to represent year, month and day for cut-off when
        filtering individuals. If None, then don't filter.

        verbose (bool) : Flag to print out progress information during and summary statistics after
        creating the dataset

        rem_resbreaks (bool) : Flag to remove individuals who moved from Wales, i.e. those
        tagged with "Residency Break".

        incl_mortality (bool) : Flag to include a mortality node to directed hypergraph.

        exclusive (bool) : Flag to only include individuals that only ever had a distinct subset of
        the diseases in columns, i.e. anyone who ever had other diseases not mentioned in columns
        are excluded.

    Returns:
        data_df (pd.DataFrame) : DataFrame of DateTime instances for individuals, all non-NA entries
        represent the first date a disease was observed by the healthcare system.

        demo_df (pd.DataFrame) : Demographic data on the individuals.

        ALL_OBS (np.array, dtype=np.uint8) :  Same as data_df but converted to binary and a numpy array.

        ALL_CONDS (np.array, dtype=np.int8) : Ordered condition indexes (according to column index of ALL_OBS).

        ALL_IDX (np.array, dtype=np.int8) : indexes of ALL_CONDS to represent 1-duplicates. Rows of -1 represent
        clean progressions for individuals and a -2 in the first column represent those individuals with only 1
        condition.

        ALL_OGIDX (np.array, dtype=np.int64) : Original indices of individuals from input DataFrame for reference.
    """
    if verbose:
        st = t.time()
        print(f"Building dataset with diseases:\n{columns}")
        print()
        print("Filtering dataset according to diseases specified...")

    # Slice disease dataframe to only include FIRST observational dates and create demographic dataframe
    # allowing option to remove individuals who moved during cohort coverage. Rename columns accordingly
    data_df = data[[f"FIRST_{col}" for col in columns]]
    original_obs = data_df.shape[0]
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
        print(
            "Removing individuals who left cohort intermittently for a residency break..."
        )
        demo_df = demo_df.loc[data["COHORT_END_DESC"] != "Residency break"].reset_index(
            drop=True
        )
        data_df = data_df.loc[data["COHORT_END_DESC"] != "Residency break"].reset_index(
            drop=True
        )

    # If date string specified, filter data by removing individuals who have any observed condition
    # date earlier than date specified
    if date_index is not None:
        print("Removing individuals whose condition dates < {date_index}...")
        y, m, d = date_index
        time_threshold = dt.date(y, m, d)
        date_bool = np.all((data_df > time_threshold) | (pd.isnull(data_df)), axis=1)
        data_df = data_df.loc[date_bool].reset_index(drop=True)
        demo_df = demo_df.loc[date_bool].reset_index(drop=True)

    # Convert to binary numpy array and remove those without any conditions specified in columns
    data_binmat = np.array(data_df.astype(bool).astype(int))
    dir_ind = np.where(data_binmat.sum(axis=1) > 0)[0]
    data_df = data_df.loc[dir_ind].reset_index(drop=True)
    demo_df = demo_df.loc[dir_ind].reset_index(drop=True)
    data_binmat = data_binmat[dir_ind]

    # If only including individuals that only ever had some subset of those diseases in columns
    n_exclusive = data_binmat.shape[0]
    n_inclusive = data_binmat.shape[0]
    if exclusive:
        print(
            "Removing individuals whose final multimorbidity set are a subset of those diseases provided..."
        )

        # Difference between all charlson diseases and diseases specified
        diff_diseases = list(set(all_columns) - set(columns))

        # If not all diseases specified then nobody is excluded, look at binary flag data for all
        # non-specified diseases and remove anyone who has been observed to have any of these
        # non-specified diseases.
        if diff_diseases != []:
            diff_columns = [f"FIRST_{col}" for col in diff_diseases]
            all_data_binmat = np.array(data[diff_columns].astype(bool).astype(int))[
                dir_ind
            ]
            exc_ind = np.where(all_data_binmat.sum(axis=1) == 0)[0]
            data_df = data_df.loc[exc_ind].reset_index(drop=True)
            demo_df = demo_df.loc[exc_ind].reset_index(drop=True)
            data_binmat = data_binmat[exc_ind]
            n_exclusive = data_binmat.shape[0]

    # If including mortality as a node, then sort all death columns
    if incl_mortality:
        print("Sorting individuals with recorded deaths...")
        all_death_merge = pd.DataFrame(index=np.arange(demo_df.shape[0]))
        for col in death_cols:
            # Extract individuals with death column entries
            if col != "COHORT_END_DATE":
                deathcol_df = demo_df[~pd.isnull(demo_df[col])][col]
            else:
                deathcol_df = demo_df[demo_df["COHORT_END_DESC"] == "Died"][col]

            # Outer merge for each death column
            all_death_merge = all_death_merge.merge(
                deathcol_df, how="outer", left_index=True, right_index=True
            )

        # Remove individuals without any death entries, convert to DateTime and then take maximum across columns
        all_death_merge = (
            all_death_merge[~np.all(pd.isnull(all_death_merge), axis=1)]
            .apply(pd.to_datetime)
            .max(axis=1)
        )

        # Create DateTime Series of individuals and the maximum condition dates
        death_data_df = data_df.loc[np.array(all_death_merge.index)]
        all_death_dis_dates = pd.to_datetime(
            [max(list(filter(None, row))) for row in np.array(death_data_df)]
        )

        # Compute difference between maximum condition date and death date and identify those nonnegative entries
        death_dis_diff = all_death_merge - all_death_dis_dates
        death_dis_diff_idx = np.where((death_dis_diff / np.timedelta64(1, "D")) > 0)[0]
        n_deaths = death_dis_diff_idx.shape[0]

        # Select those dates of death which are greater than all condition dates
        death_dates = all_death_merge.iloc[death_dis_diff_idx]
        demo_df["MORTALITY_DATE"] = death_dates
        demo_df["MORTALITY"] = np.zeros(demo_df.shape[0], dtype=np.uint8)
        demo_df.loc[~pd.isnull(demo_df["MORTALITY_DATE"]), "MORTALITY"] = 1
        demo_cols.append("MORTALITY")
        demo_cols.append("MORTALITY_DATE")
        demo_names.append("MORTALITY")
        demo_names.append("MORTALITY_DATE")

    # Remove unnecessary columns from demographic DataFrame and rename columns
    demo_df = demo_df[demo_cols]
    demo_df.columns = demo_names

    # Convert dates to numpy type datetime64 and extract number of diseases and observations
    data_arr = np.array(data_df).astype("datetime64[D]")
    n_diseases = data_arr.shape[1]
    n_obs = data_arr.shape[0]

    # Initialise containers for individuals with clean progression and those with duplicate
    # progression, the duration between disease observations and order of observed conditions
    if verbose:
        print("Categorising individual progressions...")
    clean_traj = []
    clean_dur = []
    clean_cond = []

    dupl_traj = []
    dupl_dur = []
    n_dupls = []
    dupl_cond = []

    single_traj = []
    single_cond = []

    # Organise individuals into lists where they either had a clean observational disease progression,
    # a disease progression where some conditions were observed at the same time, or those who only
    # had a single condition observed through their time interacting with healthcare system
    for i, obs in enumerate(data_arr):
        # Locate which conditions were observed, extract number of conditions and remove None instances
        ind_cond_idx = np.where(~pd.isnull(obs))[0]
        ind_ndisease = len(ind_cond_idx)
        ind_cond_dates = obs[ind_cond_idx]

        # Check if individual had single disease or not
        if ind_ndisease > 1:
            # Sort dates and diseases
            sort_date_idx = np.argsort(ind_cond_dates)
            sort_dates = ind_cond_dates[sort_date_idx]
            sort_cond = ind_cond_idx[sort_date_idx]

            # Compute observational disease date difference
            unique_diseases = np.unique(sort_dates)
            n_unique_diseases = unique_diseases.shape[0]
            disease_diffs = (sort_dates[1:] - sort_dates[:-1]).astype(int)

            # If any of these differences are 0, then individual appended to duplicate
            # disease progression list. Otherwise, add individual to clean disease
            # progression list
            if n_unique_diseases != ind_ndisease:
                dupl_traj.append(i)
                n_dupls.append(ind_ndisease - n_unique_diseases)
                dupl_dur.append(disease_diffs)
                dupl_cond.append(sort_cond)
            else:
                clean_traj.append(i)
                clean_dur.append(disease_diffs)
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

    # Work out where durations between First observations of conditions are 0 (i.e. duplications) in individual's
    # disease prgression and split by n-duplicate, n = 1,..., 5
    dupl_seq_loc = [np.where(lst == 0)[0] for lst in dupl_dur]
    dupl_seq_split = [
        np.array([seq for seq in dupl_seq_loc if len(seq) == i], dtype=np.int8)
        for i in np.unique(n_dupls)
    ]
    dupl_seq_idx = [
        [i for i, seq in enumerate(dupl_seq_loc) if len(seq) == j]
        for j in np.unique(n_dupls)
    ]

    # Using the index split dup_seq_idx, perform the same split for the individuals'
    # conditions and durations
    dupl_cond_split = [[dupl_cond[idx] for idx in lst] for lst in dupl_seq_idx]

    # We will always keep the 1-duplicates
    dupl_obs_ogidx = dupl_traj[dupl_seq_idx[0]]
    dupl_obs_binmat = data_binmat[dupl_obs_ogidx]
    dupl_obs_conds = dupl_cond_split[0]
    dupl_obs_idx = list(dupl_seq_split[0])

    # Loop through remaining individuals to extract those with multiple 1-duplicates
    for i in range(1, max(n_dupls)):
        # Take difference of locations where duplicates were observed
        dup_seq = dupl_seq_split[i]
        dup_seq_diff = np.diff(dup_seq, axis=1)

        # Find individuals who only have 1-duplicates, i.e. the location of their duplicates
        # are not consecutive or else that would mean their duplicate would be of multiple diseases
        dup_seq_1dupls = np.argwhere((dup_seq_diff != 1).all(axis=1))[:, 0]

        # If detected any, extract observation index, binary flag, ordered condition and indexes
        # of 1-duplicates
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

    if verbose:
        print("Combining individuals with clean and duplicate progressions...")
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

    # Convert ALL_IDX and ALL_CONDS to their worklists
    if verbose:
        print("Computing worklists...")
    ALL_CONDS = utils.compute_worklist(ALL_CONDS, n_diseases)
    # -1's added to the end of the array in utils.compute_worklist()
    ALL_IDX = utils.compute_worklist(ALL_IDX, 4)  # n_diseases//2)
    output = (ALL_OBS, ALL_CONDS, ALL_IDX, ALL_OGIDX)

    # DataFrame output for demographics and disease flags
    all_data_df = (
        demo_df.iloc[ALL_OGIDX].reset_index(drop=True),
        data_df.iloc[ALL_OGIDX].reset_index(drop=True),
    )

    # Output information on dataset
    if verbose:
        en = t.time()
        elapsed = np.round(en - st, 2)
        print()
        print(f"Completed in {elapsed} seconds.")
        print(f"Original number of individuals: {original_obs}")
        print(f"Number of diseases: {n_diseases}")
        if exclusive:
            print(
                f"Number of individuals with 1 or more disease in columns specified: {n_inclusive}"
            )
            print(
                "Number of these individuals which *only* had some subset of diseases in columns "
                f"specified: {n_exclusive}"
            )
        else:
            print(f"Number of individuals with 1 or more disease in {columns}: {n_obs}")
        if incl_mortality:
            print(
                f"Number of these individuals with a valid recorded date of death: {n_deaths}"
            )
        print(f"    Number of individuals with single disease: {n_single_inds}")
        print(f"    Number of individuals with clean progressions: {n_clean_inds}")
        print(
            f"    Number of individuals with duplicate progressions: {n_all_dupl_inds}"
        )
        print(f"        Individuals with duplicates valid for analyses: {n_1dupl_inds}")
        print(
            f"Total individuals available for analysis: {n_single_inds + n_clean_inds+n_1dupl_inds} "
            f"({int(100*(n_single_inds+n_clean_inds+n_1dupl_inds)/n_obs)}% of "
            f"{n_single_inds+n_clean_inds+n_all_dupl_inds})"
        )
        print(
            f"This represents {int(100*(n_single_inds+n_clean_inds+n_1dupl_inds)/original_obs)}% of the original "
            "dataset"
        )

    return all_data_df, output


##########################################################################################################
# 2. BUILD PROGRESSION SETS FOR INDIVIDUALS
##########################################################################################################


@numba.njit(fastmath=True, nogil=True)
def compute_progset(ind_cond, ind_idx, undirected=False, mort_flag=-1, mort_type=0):
    """
    Construct disease progression set for an individual with an ordered
    array of diseases.

    ind_idx specified where in the ordered progression any 1-duplicates exist.
    In the case where duplicates are observed, the progression set will be constructed for an
    individual assuming a clean progression, and then any duplicates are constructed
    afterward by permuting those conditions which were observed at the same time.

    Note that this function permits the inclusion of mortality using the mort_flag variable.
    If -1, then exclude mortality alltogether, if 0 then individual still alive by the end of cohort PoA.
    If 1 then individual died. Extra hyperarcs are created to represent progression to death/survival,
    taking account of where a duplicate is observed for the last conditions for an individual.

    Note further we have implemented multiple ways of account for mortality using the mort_type
    variable. If mort_type=0 we have 1 mortality node to represent progression to death. If mort_type=1
    then we have a single node for death and a single node to represent the individual still alive by the
    end of the cohort PoA. If mort_type=2 then we have a mortality node for each disease to represent tagging
    the individuals final observed disease prior to their death. This type has no alive node. If mort_type=3,
    we have a single alive node and a mortality node fo each disease node. If mort_type=4, we have a mortality
    node for each degree hyperarc representing the number of conditions obtained prior to death. This type has
    no alive node. If mort_type=5, we have a mortality node for each degree and an alive node.

    Arg:
        ind_cond (np.array, dtype=np.int8) : Numpy array of integers representing order of
        observed conditions.

        ind_idx (np.array, dtype=np.int8) : Numpy array of integers representing index
        of ordered conditions where a 1-duplicate has occurred. If array contains -1,
        individual is assumed to have a clean disease progression

        undirected (bool) : Flag to specify whether progression set is producing undirected
        progressions, i.e. where duplicates don't care about hyperarc ordering of tail and head.

        mort_flag (int) : Integer flag for whether individual died. -1 ignores mortality, 0 is alive and
        1 is dead.

        mort_type (int) : Type or mortality setup. Integer between and including 0 and 5.

    Returns:
        full_prog_set (np.array, dtype=np.int8) : Progression set for individuals with ordered conditions
        stored in ind_cond and any 1-duplicates stored in ind_idx.
    """
    # Make copies of cond and idx arrays and work out maximum degree hyperarc (excluding mortality) the
    # individual contributes to
    ind_cond = ind_cond.copy()
    ind_idx = ind_idx.copy()
    hyp_degree = ind_cond.shape[0] - np.sum(ind_cond == -1)

    # Work out maximum duplicate, and check if duplicate is at the end of individual progression
    max_dupl_idx = ind_idx.max()
    end_dupl_flag = int(max_dupl_idx == hyp_degree - 2)

    # Instead of IF statements, select options from list to speed runtime, i.e. incl_mort
    # is a flag to include mortality (where mort_flag = 0 or 1).
    incl_mort = [1, 1, 0][mort_flag]
    alive_node = [0, 1][mort_type]

    # incl_alive_node is required to setting correct indices in prog_set. This will be
    # flagged only if the individual died, or if the individual lived and we have an aive node
    incl_alive_node = int(alive_node + mort_flag > 0)

    # Number of duplicates to deal with during creation of individual's progressions
    # Number of diseases, incremented by one if we're using mortality.
    n_dupl = ind_idx.shape[0] - np.sum(ind_idx == -1)
    n_disease_only = ind_cond.shape[0]
    n_diseases = n_disease_only + incl_mort

    # create dummy array to use to build clean progression set
    dummy_vec = -1 * np.ones(shape=(n_diseases), dtype=np.int8)

    # Create progression set as if individual had a clean progression
    prog_set_list = [ind_cond[:j].astype(np.int8) for j in range(2, hyp_degree + 1)]
    prog_set_arr = np.zeros(
        (len(prog_set_list) + incl_alive_node, n_diseases), dtype=np.int8
    )
    for i, arr in enumerate(prog_set_list):
        prog_set_arr[i] = np.array(
            list(arr) + list(dummy_vec[: (n_diseases - len(arr))]), dtype=np.int8
        )

    # If including mortality, tag on last progression which will represent whether the individual
    # lived or died by the end of the cohort. Second condition is for whether to include alive node.
    if incl_alive_node == 1:
        last_prog = list(prog_set_list[-1])
        last_disease = last_prog[-1]
        mortality_list = [
            [-1, n_disease_only],
            [n_disease_only, n_disease_only + 1],
            [-1, n_disease_only + last_disease],
            [n_disease_only, n_disease_only + 1 + last_disease],
            [-1, n_disease_only + hyp_degree - 1],
            [n_disease_only, n_disease_only + 1 + hyp_degree - 1],
        ][mort_type]
        mort_node = mortality_list[mort_flag]
        mort_dummy = list(dummy_vec[: n_diseases - 1 - hyp_degree])
        prog_set_arr[-1] = np.array(last_prog + [mort_node] + mort_dummy, dtype=np.int8)
        ind_idx[n_dupl] = [-1, hyp_degree - 1][end_dupl_flag]

    # Check if ind_index[0] is -1. If not, individual has a duplicate
    if ind_idx[0] != -1:
        # If constructing undirected progressions then build this into model through the mult
        # variable, mult is used to determine number of extra hyperarcs/hyperedges
        is_und = int(undirected)
        mult = [2, 1][is_und]

        # Check number of duplicates to find out number of new hyperarcs
        # which_mort is 1 when incl_mort=end_dupl_flag, i.e. if including mortality and their last duplicate index was
        # at the end of their progression.
        # Can also be described as which_mort = (1-m)*e + (1-e)*m (e = end_dupl_flag, m = incl_mort).
        which_mort = [incl_mort, 1 - incl_mort][end_dupl_flag]
        if ind_idx[0] == 0:
            const = 1 + (end_dupl_flag * incl_mort) * (is_und - 1)
            n_new_hyperarcs = (
                mult * n_dupl - const - incl_mort * (1 - incl_alive_node)
            )  # Additional term is whether we have alive node or not
        else:
            n_new_hyperarcs = mult * n_dupl - is_und * (1 - which_mort)

        # Given number of duplicates, specify exactly how many 1-duplicates to loop over and initialise array for extra
        # hyperarcs
        ind_indexes = (
            ind_idx[: n_dupl + end_dupl_flag * incl_mort * incl_alive_node]
            if n_new_hyperarcs > 0
            else ind_idx[:0]
        )
        extra_progset = np.zeros((n_new_hyperarcs, n_diseases), dtype=np.int8)

        # loop over indexes where 1-duplicates occurred
        j = 0
        for idx in ind_indexes:
            # If including mortality and we have reached the duplicate index which requires the swapping of the last 2
            # conditions before tagging on the mortality node. Can only happen when incl_mort == 1 so mortality_node is
            # known.
            if idx == hyp_degree - 1:
                last_disease = last_prog[-2]
                mortality_list = [
                    [-1, n_disease_only],
                    [n_disease_only, n_disease_only + 1],
                ][mort_type]
                mort_node = mortality_list[mort_flag]
                extra_progset[mult * j - 1] = np.array(
                    last_prog[:-2] + last_prog[-2:][::-1] + [mort_node] + mort_dummy,
                    dtype=np.int8,
                )
                j += 1
                continue

            # Store first condition, the swapped pair of conditions to be permuted
            # and the second element of this pair
            deg0 = ind_cond[0]
            deg1 = ind_cond[idx : idx + 2][::-1].astype(np.int8)
            deg2 = ind_cond[idx + 1]

            # If 1-duplicate at beginning, append the swapped pair of conditions if computing
            # progression set for directed hypergraph, otherwise skip
            if idx == 0 and not undirected:
                extra_progset[j] = np.array(
                    list(deg1) + list(dummy_vec[: (n_diseases - 2)]), dtype=np.int8
                )
                j += 1

            # If 1-duplicate swaps the second and third conditions, then we can't use deg_prev as it
            # indexes 2 before second condition, so manually construct, i.e. for 1-duplicate progression
            # example A, (B -> C) we require the additional hyperarcs deg_alt1 = A -> C and
            # deg_alt2 = A, C -> B, etc.
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

            # Otherwise, we can use deg_prev to ensure we add on permutations after known progression prior to
            # 1-duplicate
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
            (prog_set_arr.shape[0] + extra_progset.shape[0], n_diseases), dtype=np.int8
        )
        full_prog_set[: prog_set_arr.shape[0]] = prog_set_arr
        full_prog_set[prog_set_arr.shape[0] :] = extra_progset
    else:
        full_prog_set = prog_set_arr

    return full_prog_set


##########################################################################################################
# 3. BUILD DIRECTED MODEL
##########################################################################################################


@numba.njit(fastmath=True, nogil=True)
def compute_directed_model(
    data, ordered_cond, ordered_idx, mortality=None, mortality_type=0
):
    """
    Compute prevalence for all hyperarcs and hyperedge and build the incidence matrix
    negative entries represent tail nodes of the hyperarc and positive entries represent
    the head node of the hyperarc.

    This requires not only the known binary flag matrix of individuals and their multimorbidity
    set, but also an ordering of their observed disease progression via ordered_cond as well as
    information on whether certain conditions for the same individual were first observed
    on the same episode during interaction with the healthcare system.

    Prevalence is stored in two numpy arrays, the 1-dimensional numpy array hyperedge_prev stores
    parent hyperedge prevalence as well as single-set disease populations. The 2-dimensional
    numpy array hyperarc_prev orders the columns as head node entries and the rows as different
    tail node combinations. Both arrays are of type np.float64.

    Mortality type can be 0 or 1.

    Args:
        data (np.array, dtype=np.uint8) : Binary array representing observations as rows
        and their conditions as columns.

        ordered_cond (np.array, dtype=np.int8) : Numpy array of integers representing order of
        observed conditions.

        ordered_idx (np.array, dtype=np.int8) : Numpy array of integers representing index
        of ordered conditions where a 1-duplicate has occurred. If array contains -1,
        individual is assumed to have a clean disease progression.

        mortality (np.array, dtype=np.uint8) : If None, then mortality is not used. If numpy
        binary array then mortality is introduced into hypergraph set up.

        mortality_type (int) : Type or mortality setup. Integer including 0 and 1.

    Returns:
        inc_mat (2d np.array, dtype=np.int8) : Signed, directed incidence matrix which store tail members of
        a hyperarc as -1's in a row and the head member as 1. Can be split into tail- and head-incidence matrices

        hyperarc_worklist (2d np.array, dtype=np.int8) : Hyperarcs, ordered as they are seen in inc_mat, stored as
        the disease index followed by a stream if -1's for compatability with numba.

        hyperarc_prev (2d np.array, dtype=np.float64) : Prevalence array for hyperarcs. Entry [i,j] is the prevalence
        for the hyperarc whose tail members binary encoding is i and whose head member disease index is j.

        hyperedge_prev (1d np.array, dtype=np.float64) : Prevalence array for corresponding parent hyperedges of the
        hyperarcs. Entry [i] is the prevalence for the hyperedge whose disease set's binary encoding is equal to i.

        node_prev (1d np.array, dtype=np.float64) : Prevalence array for the nodes. The prevalence for a disease node
        is split into it's tail- and head-components and prevalences, for cample entries [i] and [i+n_diseases] is the
        node prevalence for the tail- and head-component of D_i, respectively.
    """

    # INITIALISATION OF PREVALENCE ARRAYS, MORTALITY, INCIDENCE MATRICES, ETC.

    # Number of diseases and observations
    n_diseases = data.shape[1]
    n_obs = data.shape[0]

    # Setup for using mortality
    incl_mort = 0 if mortality[0] == -1 else 1
    n_morts = [1, 2][mortality_type]
    n_mort_nodes = incl_mort * n_morts
    alive_node = [0, 1][mortality_type]

    # deduce maximum number of hyperarcs and number of hyperedges
    mort_type = [None, mortality_type][incl_mort]
    max_hyperarcs = utils.N_max_hyperarcs(n_diseases, b_hyp=True, mort_type=mort_type)
    max_disease_hyperedges = 2**n_diseases
    max_total_hyperedges = 2 ** (n_diseases + incl_mort * (n_morts - 1)) + incl_mort * (
        (2 ** (np.arange(n_diseases))).sum() + 1
    )

    # Initialise hyperarc work list, hyperarc and node prevalence arrays and directed incidence matrix.
    # Dummy vector used to fill row of incidence matrix as initialised as empty
    hyperedge_prev = np.zeros((max_total_hyperedges), dtype=np.float64)
    hyperarc_worklist = np.empty(
        shape=(max_hyperarcs, n_diseases + incl_mort), dtype=np.int8
    )
    hyperarc_prev = np.zeros(
        (max_disease_hyperedges, n_diseases + n_mort_nodes), dtype=np.float64
    )
    node_prev = np.zeros((2 * n_diseases + n_mort_nodes), dtype=np.float64)
    pop_prev = np.zeros(n_diseases + n_mort_nodes, dtype=np.int64)
    inc_mat = np.empty(shape=(max_hyperarcs, n_diseases + n_mort_nodes), dtype=np.int8)
    dummy_vec = np.zeros(n_diseases + n_mort_nodes, dtype=np.int8)

    # Loop over each individuals binary flag vector representing undirected
    # multimorbidity set
    n_row = 0
    for ii in range(n_obs):
        # EXTRACT INFORMATION ON INDIVIDUAL

        # Select binary realisations of individual, order of morbidities
        # representing disease progression, potential indices of duplicates
        # look at each row/patient individually:
        # ind_binmat = data[ii]
        ind_cond = ordered_cond[ii]
        ind_idx = ordered_idx[ii]

        # Obtain information on whether individual died. Is -1 if excluding mortality
        # incl_alive_node is required for ensuring whether we add prevalence to mortality
        # hyperarc or not. This will be flagged only if the individual died, or if the
        # individual lived and we have an alive node
        ind_mort = mortality[
            ii
        ]  # goes through mortality column to determine if individual alive or mort, 0 = alive
        incl_alive_node = incl_mort * int(
            alive_node + ind_mort > 0
        )  # binary of whether to include alive nodes or not

        # Select node weight to increment node prevalence. For the case where we have mortality
        # for each degree, let node_weight be unitary, otherwise mortality node for higher degrees
        # will be significantly larger than lower degrees.
        n_ind_cond = ind_cond[ind_cond != -1].shape[
            0
        ]  # number of conditions individual has
        node_weight = n_ind_cond - 1  # num of conditions - 1
        mort_node_weight = [n_ind_cond, 1][
            mortality_type // 4
        ]  # this is the same as the inds num of conditions

        # Add individual to population prevalence counter of single diseases
        for c in ind_cond[:n_ind_cond]:
            pop_prev[c] += 1

        # If mortality type 1 is included count the number of people who are alive and mort
        if mortality_type == 1:
            pop_prev[-1] += ind_mort
            if ind_mort == 0:
                pop_prev[-2] += 1

        # Check if individual only has 1 disease, if not, then continue to deduce their
        # hyperarcs. If they are, move to next individual.
        # Checking whether individuals have duplciate progressions
        min_indidx = ind_idx[0]  # -2 means only one disease present
        if min_indidx != -2:
            # COMPUTE HYPEREDGE PREVALENCE OF FIRST OBSERVED DISEASE INDIVIDUAL OBTAINED

            # First disease individual had contributes to single-set hyperedge prevalence. If duplicate exists
            # at beginning of progression, then hyperedge prevalence for single-set disease needs halved for
            # first and second condition.
            which_min = int(min_indidx == 0)
            start_dupl = [1, 2][which_min]
            hyp_cont = [1.0, 0.5][which_min]
            hyp_idx = ind_cond[:start_dupl].astype(np.int64)
            hyperedge_prev[2**hyp_idx] += hyp_cont

            # COMPUTE DISEASE/MORTALITY NODE PREVALENCE
            # Add prevalence to mortality node. First select the potential alive-index and dead-index
            # based on the mort_type and then increment node prevalence for that node index (based on ind_mort).
            # If not including mortality, add 0 prevalence to 0th index of node_prev so that we do nothing.
            surv_mort_list = [
                [0, 2 * n_diseases],
                [2 * n_diseases, 2 * n_diseases + 1],
            ][mortality_type]
            death_idx = surv_mort_list[ind_mort]

            # If individual has a clean progression or if the duplicate ISN'T at end of progression
            # then contribution don't need to alter contribution.
            max_indidx = ind_idx.max()
            if min_indidx == -1 or max_indidx != n_ind_cond - 1:
                fin_cond = ind_cond[n_ind_cond - 1]
                for i in range(n_ind_cond - 1):
                    node_prev[ind_cond[i]] += 1.0
                node_prev[n_diseases + fin_cond] += node_weight
                node_prev[incl_alive_node * death_idx] += (
                    incl_alive_node * mort_node_weight
                )

            # If individual has duplicate at end of progression then contribution to last two diseases
            # is halved to take into account hyperarcs where these diseases have swapped their tail and
            # head role
            else:
                dupl_nodes = ind_cond[n_ind_cond - 2 : n_ind_cond]
                for i in range(n_ind_cond - 2):
                    node_prev[ind_cond[i]] += 1.0
                node_prev[dupl_nodes] += 0.5
                node_prev[n_diseases + dupl_nodes] += node_weight / 2

                # This flag is only called when we the individual died and we have a mortality node for every disease,
                # as a 1-duplicate at the end of their progressions means incrementing the prevalence of each mortality
                # node of the two diseases in the 1-duplicate. Every other option available sets incl_dupl = 0 so we
                # only use death_idx to update the prevalence arrays, i.e. node_idx = death_idx when
                # mort_type != 2or3 and ind_mort = 0.
                incl_dupl = [[0, 0], [0, 0]][ind_mort][mortality_type]
                scale = [1.0, 0.5][incl_dupl]
                node_idx = incl_dupl * dupl_nodes[: incl_dupl + 1] + death_idx
                node_prev[incl_alive_node * node_idx] += (
                    incl_alive_node * mort_node_weight * scale
                )

            # COMPUTE PROGRESSION SET

            # Compute progression set based on ordering
            prog_set = compute_progset(
                ind_cond,
                ind_idx,
                undirected=False,
                mort_flag=ind_mort,
                mort_type=mortality_type,
            )

            # Keep track of elements looping through progression set
            n_prog_obs = len(prog_set)
            prog_bin_int = np.zeros(n_prog_obs, dtype=np.int64)

            # Work out number of conditions in each progression element of prog_set
            n_conds_prog = prog_set.shape[1] - (prog_set == -1).sum(axis=1)

            # LOOP OVER OBSERVED PROGRESSIONS AND INCREMENT CONTRIBUTIONS TO HYPERARCS AND PARENT HYPEREDGES

            # Loop over individual's progression set
            for jj in range(n_prog_obs):
                # Extract progression set element
                elem = prog_set[jj]
                n_conds = n_conds_prog[jj]

                # Work out binary integer mappings for hyperarc (bin_tail, head_node) and
                # parent hyperedge (bin_tail + head_node)
                bin_tail = 0
                for i in range(
                    n_conds - 1
                ):  # based on the number of conditions in a progression set
                    bin_tail += (
                        2 ** elem[i]
                    )  # loops through which conditions occur (except last)
                head_node = elem[n_conds - 1]
                bin_headtail = bin_tail + 2**head_node

                # Work out if contribution needs to be halved to take into account any duplicates
                hyperedge_cont = 1.0
                hyperarc_cont = 1.0
                if bin_headtail in prog_bin_int:
                    hyperedge_cont = 0.0
                if (n_conds == n_conds_prog).sum() == 2:
                    hyperarc_cont = 0.5

                # Add element of progression set to prog_bin_int
                prog_bin_int[jj] = bin_headtail

                # IF UNOBSERVED PROGRESSION HYPERARC, ADD TO INCIDENCE MATRIX AND HYPERARC WORKLIST
                # OTHERWISE, CONTINUE TO INCREMENT CONTRIBUTIONS

                # Check if hyperarc has been seen before, if not then it should still be 0
                # and needs to be added to incidence matrix
                if hyperarc_prev[bin_tail, head_node] == 0:
                    # Add hyperarc to worklist
                    hyperarc_worklist[n_row] = elem

                    # Update incidence matrix
                    inc_mat[n_row] = dummy_vec

                    for i in range(n_conds - 1):
                        inc_mat[n_row, elem[i]] = -1

                    inc_mat[n_row, elem[n_conds - 1]] = 1

                    n_row += 1

                # Initialise prevalence for this hyperarc and also the parent hyperedge
                # using contribution from individual
                hyperedge_prev[bin_headtail] += hyperedge_cont
                hyperarc_prev[bin_tail, head_node] += hyperarc_cont

        # If individual only has 1 disease, then half contribution to head and tail disease node
        # and contribute to single-set disease prevalence
        else:
            # Extract only condition individual had, and add to hyperedge prevalence as this was their
            # first condition
            single_cond = ind_cond[0]

            # List of lists specifying which the index of mortality/survival to select based on mort_type
            surv_mort_list = [
                [n_diseases, 2 * n_diseases],
                [2 * n_diseases, 2 * n_diseases + 1],
            ][mortality_type]

            death_idx = surv_mort_list[ind_mort]

            # CONTRIBUTE PREVALENCE FOR INDIVIDUALS WITH ONLY 1 CONDITIONS

            # If the individual only had one condition and then died, then prevalence is contributed
            # at the last row of the array, while if they lived then it is contributed at
            # first row of array
            # INCLUDE SELF_LOOP FROM DISEASES
            if alive_node == 0:
                hyperarc_prev[
                    -1 * incl_mort * ind_mort, single_cond
                ] += 1.0  # row, column

            # BELOW ADDS WHEN SINGLE DISEASE IS OBSERVED WITHOUT A FINAL STATE, DISEASE -> ALIVE
            if (alive_node == 1) and (ind_mort == 0):
                alive_col = n_diseases  # assuming starts with 0

                # If only one condition, then assume alive is final state
                # Work out binary integer mappings for hyperarc (bin_tail, single_cond) and
                # parent hyperedge (bin_tail + single_cond)
                bin_tail = (
                    2**single_cond
                )  # Gets the index for the single_condition (row)

                # Check if hyperarc has been seen before, if not then it should still be 0
                # and needs to be added to incidence matrix
                if hyperarc_prev[bin_tail, alive_col] == 0:
                    ind_dis_vector = dummy_vec.copy()  # individual disease vector
                    ind_dis_vector[single_cond] = -1
                    ind_dis_vector[alive_col] = 1

                    # Add hyperarc to worklist
                    hyperarc_worklist_row = np.empty(
                        shape=(n_diseases + incl_mort), dtype=np.int8
                    )
                    hyperarc_worklist_row.fill(-1)
                    hyperarc_worklist_row[0] = single_cond
                    hyperarc_worklist_row[1] = alive_col

                    hyperarc_worklist[n_row] = hyperarc_worklist_row

                    # Update incidence matrix
                    inc_mat[n_row] = ind_dis_vector
                    n_row += 1

                # Initialise prevalence for this hyperarc and also the parent hyperedge
                # using contribution from individual
                hyperedge_prev[
                    bin_tail + 2 ** (n_diseases)
                ] += 1  # add prevalence to single disease with ALIVE
                hyperedge_prev[bin_tail] += 1  # add prevalence to single disease

                hyperarc_prev[bin_tail, alive_col] += 1
                node_prev[n_diseases * 2] += 1.0  # alive node given an extra prevalence

            # BELOW ADDS WHEN SINGLE DISEASE IS OBSERVED WITH MORT, DISEASE -> MORT
            if (alive_node == 1) and (ind_mort == 1):
                mort_col = n_diseases + 1  # assuming starts with 0
                bin_tail = (
                    2**single_cond
                )  # Gets the index for the single_condition (row)

                # Check if hyperarc has been seen before, if not then it should still be 0
                # and needs to be added to incidence matrix
                if hyperarc_prev[bin_tail, mort_col] == 0:
                    ind_dis_vector = dummy_vec.copy()  # individual disease vector
                    ind_dis_vector[single_cond] = -1
                    ind_dis_vector[mort_col] = 1
                    # Add hyperarc to worklist
                    hyperarc_worklist_row = np.empty(
                        shape=(n_diseases + incl_mort), dtype=np.int8
                    )
                    hyperarc_worklist_row.fill(-1)
                    hyperarc_worklist_row[0] = single_cond
                    hyperarc_worklist_row[1] = mort_col

                    hyperarc_worklist[n_row] = hyperarc_worklist_row

                    # Update incidence matrix
                    inc_mat[n_row] = ind_dis_vector
                    n_row += 1

                # Initialise prevalence for this hyperarc and also the parent hyperedge
                # using contribution from individual
                hyperedge_prev[
                    bin_tail + 2 ** (mort_col)
                ] += 1  # add prevalence to single disease with MORT
                hyperedge_prev[bin_tail] += 1  # add prevalence to single disease

                hyperarc_prev[bin_tail, mort_col] += 1
                node_prev[-1] += 1.0  # mort node given an extra prevalence

            node_prev[single_cond] += 1.0  # single condition head given prevalence

    return (
        inc_mat[:n_row],
        hyperarc_worklist[:n_row],
        hyperarc_prev,
        hyperedge_prev,
        node_prev,
        pop_prev,
    )


##########################################################################################################
# 4. BUILD MODEL AND COMPUTE EDGE WEIGHTS
##########################################################################################################


def compute_weights(
    binmat,
    conds_worklist,
    idx_worklist,
    colarr,
    contribution_type,
    weight_function,
    end_prog,
    dice_type=1,
    end_type=0,
    plot=True,
    ret_inc_mat=False,
    sort_weights=True,
):
    """
    This function will take a dataset of individuals and their condition flags,
    two arrays specifying the ordering of their condition sets and existence of
    1-duplicates.

    The function will then build the prevalence arrays, incidence matrix and
    compute the edge weights for the hyperedges and hyperarcs.

    The function allows specification of which weight function to use and which
    contribution type to use, i.e. exclusive, progression or power set based.

    The function also allows input of mortality through end_prog and the mortality
    type, through end_type. If end_type=0, then 1 mortality node for death, if
    end_type=1, then 1 mortality node and 1 survival node. If end_type=2, then
    a mortality node for each disease node. If end_type=3, we have a mortality node
    for each disease node and a single alive node. If end_type=4, we have a mortality
    node for each degree hyperarc representing the number of conditions obtained prior
    to death. This type has no alive node. If end_type=5, we have a mortality node for
    each degree and an alive node.

    Args:
        binmat (np.array, dtype=np.uint8) : Binary flag matrix whose test observations
        are represented by rows and diseases depresented by columns.

        conds_worklist (np.array, dtype=np.int8) : For each individual, the worklist storesx
        the ordered set of conditions via their columns indexes. For compatability with Numba, once
        all conditions specified per individual, rest of row is a stream if -1's.

        idx_worklist (np.array, dtype=np.int8) : Worklist to specify any duplicates. For each
        individuals, if -2 then they only have 1 condition, if -1 then they have no duplicates and
        if they have positive integers indexing conds_worklist then it specifies which
        conditions have a duplicate which needs taken care of.

        colarr (np.array, dtype="<U24") : Numpy array of disease column titles

        contribution_type (str) : Type of contribution to hyperedges each individual has, i.e. can be
        "exclusive", "power" or "progression".

        weight_function (func) : Numba-compiled weight function, will be version of the overlap coefficient
        and modified sorensen-dice coefficient.

        end_prog (np.array, dtype=np.int8) : Array storing end of progression information on
        individuals. If an array of -1's, then excluding progression end information.

        dice_type (int) : Type of Sorensen-Dice coefficient used, either 1 (Complete) or 0 (Power).

        end_type (int) : Type of end-of-progression setup. Must be an integer between 0 and 5.

        plot (bool) : Flag to plot hyperedge and hyperarc weights.

        ret_inc_mat (bool) : Flag to return the tail and head incidence matrices.

        sort_weights (bool) : Flag to sort weights before returning them.
    """
    # Number of observations and columns
    N_obs, N_diseases = binmat.shape
    columns_idxs = np.arange(N_diseases).astype(np.int8)
    if colarr is None:
        colarr = np.asarray(list(string.ascii_uppercase[:N_diseases]), dtype="<U24")
    else:
        colarr = np.asarray(colarr, dtype="<U24")

    # Set up end of progression information
    incl_end = 1 if end_prog[0] != -1 else 0
    # alive_node = [0, [0, 1][end_type]][
    #    incl_end
    # ]  # is 1 if alive node is included in mort_type
    n_morts = [0, [1, 2][end_type]][incl_end]  # tells us the number of mortality nodes
    n_end = end_prog.sum()  # number of mortality end states for all individuals
    # n_alive = N_obs - n_end  # number of alive end states for all individuals
    N_cols = N_diseases + incl_end * n_morts  # num of different nodes

    print("Building directed incidence matrix and prevalence arrays...")
    # Hyperedge denominator used in the modified sorensen-dice coefficient calculation.
    # This looks only at the diseases (not mort or alive)
    # Outputs an array of len(all possible disease combos) with the num of times each disease occurs.
    hyperedge_denom = utils.compute_integer_repr(binmat, columns_idxs, colarr).astype(
        np.float64
    )

    # Build incidence matrix, prevalence arrays and hyperarc worklists
    st = t.time()

    output = compute_directed_model(
        binmat,
        conds_worklist,
        idx_worklist,
        mortality=end_prog,
        mortality_type=end_type,
    )
    (
        inc_mat,
        hyperarc_worklist,
        hyperarc_prev,
        hyperedge_prev,
        node_prev,
        pop_prev,
    ) = output

    print(f"Completed in {round(t.time()-st,2)} seconds.")

    # Set up variables for building weights for hyperedges, hyperarc and nodes, dependent on
    # whether mortality is used
    output = weight_functions.setup_weight_comp(
        colarr,
        colarr,
        binmat,
        node_prev,
        hyperarc_prev,
        hyperarc_worklist,
        weight_function,
        dice_type,
        incl_end,
        end_type,
        n_end,
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

    # Depending on contribution type, specify prevalence array for hyperedges and the hyperedge array
    # to loop through. If "exclusive" just take unique rows of the original input binmat. Prevalence array
    # comes from granular split-and-count of individuals of their final multimorbidity set.
    hyperarc_num_prev = hyperedge_prev.copy()
    if contribution_type == "exclusive":
        hyperedge_num_prev = hyperedge_denom.copy()
        denom_prev = hyperedge_num_prev.copy()
        if incl_end:
            binmat = np.concatenate([binmat, end_prog[:, np.newaxis]], axis=1)
        hyperedge_arr = np.unique(binmat, axis=0)

    # If "progression" we take the unique, absolute rows of the directed incidence matrix, prevalence array
    # comes from compute_directed_model() assuming "progression" contribution.
    elif contribution_type == "progression":
        hyperedge_num_prev = hyperedge_prev.copy()
        denom_prev = hyperedge_num_prev.copy()
        hyperedge_arr = np.unique(np.abs(inc_mat), axis=0)

        # need to add single diseases to worklist for them to be counted in the power set calculation.
        selfedge_arr = np.eye(N_diseases, N_cols, k=0)
        hyperedge_arr = np.concatenate([selfedge_arr, hyperedge_arr], axis=0)

    # If "power" hyperedge, generate worklist by taking unique entries of binmat,
    # concatenating death vector to hyperedge_arr to include mortality
    elif contribution_type == "power":
        hyperedge_num_prev = np.zeros_like(hyperedge_prev)
        if incl_end:
            binmat = np.concatenate([binmat, end_prog[:, np.newaxis]], axis=1)
        hyperedge_arr = np.unique(binmat, axis=0)

    # Build worklist of hyperedges, their unique integer representation and their disease set string
    hyperedge_worklist = utils.comp_edge_worklists(
        hyperedge_arr, contribution_type, shuffle=False
    )

    hyperedge_indexes, hyperedge_N = utils.compute_bin_to_int(
        hyperedge_worklist
    )  # hyperedge_N is the degree

    hyperedge_cols = np.asarray(
        list(
            map(
                lambda col: ", ".join(col),
                map(lambda row: colarr[row[row != -1]], hyperedge_worklist),
            )
        )
    )

    # Sort hyperedges by degree
    sort_hyps = np.argsort(hyperedge_N)
    hyperedge_worklist = hyperedge_worklist[sort_hyps]  # sorts based on node degree
    hyperedge_indexes = hyperedge_indexes[sort_hyps]
    hyperedge_N = hyperedge_N[sort_hyps]
    hyperedge_cols = hyperedge_cols[sort_hyps]

    # If Overlap coefficient we need single-disease total population sizes, i.e. we can use
    # the pop_prev we get from compute_directed_model()
    if weight_function == weight_functions.comp_overlap_coeff:
        denom_prev = pop_prev.copy()

    # BUILD HYPEREDGE WEIGHTS
    print("\nComputing hyperedge weights...")
    st = t.time()
    # Updated modified sorensen-dice coefficient combined Power and Complete here
    if weight_function == weight_functions.modified_sorensen_dice_coefficient:
        hyperedge_weights = weight_function(
            hyperedge_worklist,
            hyperedge_N,
            hyperedge_indexes,
            hyperedge_num_prev,
            denom_prev,
            dice_type,
        )

    # Build dataframe of hyperedge weights and sort if specified
    hyperedge_indexes = np.asarray(hyperedge_indexes, dtype=np.int64)
    hyperedge_weights = np.asarray(hyperedge_weights, dtype=np.float64)
    hyperedge_weights_df = pd.DataFrame(
        {"disease set": hyperedge_cols, "weight": hyperedge_weights}
    )

    hyperedge_weights_df = hyperedge_weights_df[hyperedge_weights_df.weight > 0]
    if sort_weights:
        hyperedge_weights_df = hyperedge_weights_df.sort_values(
            by=["weight"], ascending=False
        )

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
        end_type,
        hyperarc_counter,
    )

    hyperarc_weights, hyperarc_progs = output

    # Build dataframe of hyperedge weights and sort if specified
    hyperarc_weights_df = pd.DataFrame(
        {"progression": hyperarc_progs, "weight": hyperarc_weights}
    )
    if sort_weights:
        hyperarc_weights_df = hyperarc_weights_df.sort_values(
            by=["weight"], ascending=False
        )

    print(f"Completed in {round(t.time()-st,2)} seconds.")

    # Plot hyperedge, hyperarc and node weights
    if plot:
        n = 28

        # Just hyperedges
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(
            x="weight", y="disease set", data=hyperedge_weights_df.iloc[:n], ax=ax
        )
        ax.set_title("Hyperedge Weights", fontsize=18)
        ax.set_ylabel("Disease Set", fontsize=15)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=15)
        ax.set_xlabel("Weight", fontsize=15)

        # Just hyperarcs
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(
            x="weight", y="progression", data=hyperarc_weights_df.iloc[:n], ax=ax
        )
        ax.set_title("Hyperarc Edge Weights", fontsize=18)
        ax.set_ylabel("Disease Progression", fontsize=15)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=15)
        ax.set_xlabel("Weight", fontsize=15)

        # Build dataframe of top hyperarc weights to plot and their parent hyperedge weights
        edge_arc_df = pd.DataFrame(
            columns=[
                "progression",
                "hyperarc_weight",
                "disease_set",
                "hyperedge_weight",
                "hyperedge_index",
            ]
        )
        top_hyperarcs_df = hyperarc_weights_df.iloc[:n]
        hyperedge_counter = 0
        for idx, hyp_arc in top_hyperarcs_df.iterrows():
            prog, weight = hyp_arc
            prog_split = prog.split(" -> ")
            prog_tail = list(prog_split[0].split(", "))
            prog_head = prog_split[1]
            if prog_head not in prog_split[0]:
                if "MORT" in prog_head:
                    if len(prog_tail) == 1:
                        hyperedge = prog_tail[0]
                        hyperarc_title = f"{prog_tail[0]}, {prog_head}"
                    else:
                        hyperedge = ", ".join(prog_tail + [prog_head])
                        hyperarc_title = ", ".join(prog_tail + [prog_head])

                elif "ALIVE" in prog_head:
                    if len(prog_tail) == 1:
                        hyperedge = prog_tail[0]
                        hyperarc_title = f"{prog_tail[0]}, ALIVE"
                    else:
                        hyperedge = ", ".join(prog_tail + [prog_head])
                        hyperarc_title = ", ".join(prog_tail + [prog_head])
                else:
                    hyperedge = ", ".join(np.sort(prog_tail + [prog_head]))
                    hyperarc_title = ", ".join(prog_tail + [prog_head])
            else:
                hyperedge = prog_tail[0]
                hyperarc_title = hyperedge

            hyperedge_info = hyperedge_weights_df[
                hyperedge_weights_df["disease set"] == hyperedge
            ]
            try:
                edge_weight = hyperedge_info.weight.iloc[0]
            except:  # Noqa: E722
                edge_weight = 0.0

            if np.any(edge_arc_df.hyperedge_weight == edge_weight):
                hyperedge_idx = edge_arc_df[
                    edge_arc_df.hyperedge_weight == edge_weight
                ]["hyperedge_index"].iloc[0]
            else:
                hyperedge_idx = hyperedge_counter
                hyperedge_counter += 1
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(
            x="hyperarc_weight",
            y="progression",
            data=edge_arc_df,
            ax=ax,
            palette=arc_palette[palette_idxs],
        )
        ax.set_ylabel("Disease Progression (Hyperarc)", fontsize=18)
        ax.set_yticklabels(edge_arc_df.progression, fontsize=15)
        ax.set_title("Superimposed Hyperedge/Hyperarc Weights", fontsize=18)
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
        ax1.set_ylabel("Disease Set (Hyperedge)", fontsize=18)
        ax1.set_yticklabels(edge_arc_df.disease_set, fontsize=15)
        ax.grid("on")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=15)

        # Node weights - if including mortality node per disease, then weight of each node is split
        # by it's tail, head and mortality counterparts, rather than just the first two.
        if end_type // 2 == 1:
            thresh = 1 / 3
        else:
            thresh = 1 / 2
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.barplot(x="weight", y="node", data=node_weights_df, ax=ax, palette=palette)
        ax.axvline(x=thresh, ymin=0, ymax=10, c="r", linestyle="--")
        ax.set_title("Node Weights", fontsize=18)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=15)
        ax.set_ylabel("Tail/Head Nodes", fontsize=15)
        ax.set_xlabel("Weight", fontsize=15)

    # If returning incidence matrix
    if ret_inc_mat:
        output = (
            inc_mat,
            (hyperedge_weights_df, hyperarc_weights_df, node_weights_df),
            colarr,
        )
    else:
        output = hyperedge_weights_df, hyperarc_weights_df, node_weights_df
        # hyperedge_indexes, hyperedge_weights, hyperedge_prev, hyperarc_prev, hyperarc_worklist, hyperedge_worklist

    return output


##########################################################################################################
# 5. ORGANISE VARIABLES FOR DOWNSTREAM ANALYSIS
##########################################################################################################


def setup_vars(
    inc_mat, N_diseases, hyperarc_weights, hyperarc_titles, node_weights, mort_type=None
):
    """
    This function organises the directed incidence matrix into it's tail and head components,
    taking into account the inclusion of mortality and self-edges. Node and edge, tail and head
    degrees are calculated and then all data is ordered according to hyperarc degree.

    Args:
        inc_mat (np.array, dtype=np.int8) : Directed incidence matrix of hyperarcs with -1s to represent
        tail nodes and 1s to represent head nodes.

        N_diseases (int) : Number of nodes in the directed hypergraph.

        hyperarc_weights (np.array, dtype=np.float64) : Hyperarc weights.

        hyperarc_titles (np.array, dtype="<U512")  : Disease progression titles.

        node_weights (np.array, dype=np.float64) : Node head- and tail-counterpart weights

        mort_type (int) : If None, ignore mortality, if 0 then include one mortality node for death.
        If mort_type=0 we have 1 mortality node to represent progression to death. If mort_type=1
        then we have a single node for death and a single node to represent the individual still alive by the
        end of the cohort PoA.
    """

    # Number of edges and nodes in directed hypergraph and convert node and edge weights to arrays
    N_edges, N_nodes = inc_mat.shape
    edge_weights = np.asarray(hyperarc_weights, dtype=np.float64)
    hyperarc_titles = np.asarray(hyperarc_titles, dtype="<U512")
    node_weights = np.asarray(node_weights, dtype=np.float64)

    # if mort_type is not None:
    #     n_morts = [1, 2][mort_type]
    # else:
    #     n_morts = 0

    # Tail incidence matrix are where entries in inc_mat are negative
    inc_mat_tail = inc_mat.T.copy()
    inc_mat_tail[inc_mat_tail > 0] = 0
    inc_mat_tail = np.abs(inc_mat_tail)

    # Head incidence matrix are where entries in inc_mat are positive
    inc_mat_head = inc_mat.T.copy()

    # If there's only negatives in a row make them 1 too (these are self-edges)
    for col in range(inc_mat_head.shape[1]):
        col_vals = inc_mat_head[:, col]
        # if the column contains only minus one
        if (col_vals <= 0).all(axis=0):
            # replace numbers less than -1 with 1
            inc_mat_head[inc_mat_head[:, col] < 0, col] = 1
    inc_mat_head[inc_mat_head < 0] = 0

    # Create self-edges to add to incidence matrix, depending on mort_type
    if mort_type is None or mort_type == 0 or mort_type == 2 or mort_type == 4:
        selfedge = np.eye(N_nodes)[:N_diseases]
        tail_component = selfedge

    if mort_type is None:
        # this gives a selfedge
        head_component = np.eye(N_nodes)[:N_diseases]
    elif mort_type == 0:
        mortedge = np.concatenate(
            [np.zeros((N_diseases, N_diseases)), np.ones(N_diseases).reshape(-1, 1)],
            axis=1,
        )
        head_component = np.concatenate([selfedge, mortedge], axis=0)
    elif mort_type == 1:
        # one disease to alive transition (all pairs present)
        survedge = np.concatenate(
            [
                np.zeros((N_diseases, N_diseases)),
                np.ones((N_diseases, 1)),
                np.zeros((N_diseases, 1)),
            ],
            axis=1,
        )
        # one disease to mort transition
        mortedge = np.concatenate(
            [np.zeros((N_diseases, N_diseases + 1)), np.ones((N_diseases, 1))], axis=1
        )
        # adds selfedges for the mort and alive nodes
        head_component = np.concatenate([survedge, mortedge], axis=0)

    # Concatenate self-edge/mortality hyperarcs to tail and head incidence matrices
    if mort_type is None:
        # self loop for all diseases
        inc_mat_tail = np.concatenate([tail_component.T, inc_mat_tail], axis=1)
        inc_mat_head = np.concatenate([head_component.T, inc_mat_head], axis=1)

    # Calculate tail and head, node and edge degrees
    node_degs, edge_degs = centrality_utils.degree_centrality(
        inc_mat_tail, inc_mat_head, edge_weights, None
    )
    node_degree_tail, node_degree_head = node_degs
    edge_degree_tail, edge_degree_head = edge_degs

    # Compute edge degree valencies (number of nodes connecting to edge) to use to sort data
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
    disease = hyperarc_titles[sort_edges]

    # Format output
    edge_degs = (edge_degree_tail, edge_degree_head)
    output = (
        (inc_mat_tail, inc_mat_head),
        (edge_weights, disease),
        node_weights,
        node_degs,
        edge_degs,
    )

    return output
