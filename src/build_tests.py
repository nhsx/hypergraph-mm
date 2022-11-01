from individual_reps import _generate_powerset

import numpy as np
import random
import string

# Build test case


def _build_test_case(
    N_obs,
    N_cols,
    test_type=None,
    random_duplicates=False,
    incl_end=False,
    seed=0,
):
    """
    Manually build test cases to run edge weight construction on. There are a
    selection of test cases to select from. If None, a random test is built
    which is seeded.

    INPUTS:
    ----------------------
        N_obs, N_cols (int) : Number of observations and disease columns.

        incl_end (bool) : Flag to include end of progression variable.

        incl_mort (bool) : Flag to include random end of progression
        assignment, seeded by seed.

        test_type (int) : Type of test to build. If None, a random test is
        build, seeded by seed.

        seed (int) : Fix random state for reproducibility.
    """
    # Initialise disease columns, ordered condition worklist and duplicate
    # index worklist
    colarr = np.asarray(list(string.ascii_uppercase[:N_cols]), dtype="<U24")
    conds_worklist = -1 * np.ones((N_obs, N_cols), dtype=np.int8)
    condidx_worklist = -1 * np.ones((N_obs, 3), dtype=np.int8)
    columns_idxs = np.arange(N_cols)

    # Build random flag array and randomise everyone's progression
    if test_type is None:
        np.random.seed(seed)
        binmat = np.random.randint(
            low=0, high=2, size=(N_obs, N_cols), dtype=np.uint8
        )

        # Build condition and duplicate worklists at random
        for i, obs in enumerate(binmat):
            obs_idx = np.where(obs)[0]
            n_conds = obs_idx.shape[0]
            if n_conds == 1:
                conds_worklist[i, 0] = obs_idx[0]
                condidx_worklist[i, 0] = -2
            else:
                np.random.seed(1)
                np.random.shuffle(obs_idx)
                conds_worklist[i, :n_conds] = obs_idx

    # Test type 0: everyone has all conditions, except for N_cols individuals
    # who have 1 of each. For those individuals with all diseases, their
    # progression is equally split across all possible progressions
    elif test_type == 0:
        binmat = np.ones((N_obs, N_cols), dtype=np.uint8)
        binmat[:N_cols] = np.eye(N_cols)

        # Build condition and duplicate worklists
        for i, obs in enumerate(binmat):
            obs_idx = np.where(obs)[0]
            n_conds = obs_idx.shape[0]
            if n_conds == 1:
                conds_worklist[i, 0] = obs_idx[0]
                condidx_worklist[i, 0] = -2
            elif i >= N_cols and i % 2 == 0:
                conds_worklist[i, :n_conds] = obs_idx[::-1]
            else:
                conds_worklist[i, :n_conds] = obs_idx

    # Test type 1: everyone has all conditions, except for N_cols individuals
    # who have 1 of each. For those individuals with all diseases, their
    # progression is ordered mainly from 1,...,N_cols
    elif test_type == 1:
        binmat = np.ones((N_obs, N_cols), dtype=np.uint8)
        binmat[:N_cols] = np.eye(N_cols)

        # Build condition and duplicate worklists
        for i, obs in enumerate(binmat):
            obs_idx = np.where(obs)[0]
            n_conds = obs_idx.shape[0]
            if n_conds == 1:
                conds_worklist[i, 0] = obs_idx[0]
                condidx_worklist[i, 0] = -2
            elif i >= N_cols and i % 3 == 0:
                conds_worklist[i, :n_conds] = obs_idx[::-1]
            else:
                conds_worklist[i, :n_conds] = obs_idx

    # Test type 2: Equal number of individuals with each kind of disease set
    # with sequential ordering
    elif test_type == 2:
        binmat = np.zeros((N_obs, N_cols), dtype=np.uint8)
        pwset = _generate_powerset(columns_idxs, full=True)
        n_pwset = len(pwset)
        for i in range(N_obs):
            elem = pwset[i % n_pwset]
            binmat[i, elem] = 1
            n_conds = elem.shape[0]
            if n_conds == 1:
                conds_worklist[i, 0] = elem[0]
                condidx_worklist[i, 0] = -2
            else:
                conds_worklist[i, :n_conds] = elem

    # Test type 3: Random entries for all but 2 disease columns which have a
    # minor dependency. Make some random data
    elif test_type == 3:
        binmat = np.ones((N_obs, N_cols), dtype=np.uint8)
        np.random.seed(seed + 1)
        binmat = np.random.randint(
            low=0, high=2, size=(N_obs, N_cols), dtype=np.uint8
        )
        for j in range(N_obs):
            binmat[j, N_cols - 1] = (
                binmat[j, N_cols - 2]
                if np.random.random() < 0.75
                else binmat[j, N_cols - 1]
            )

        # Build condition and duplicate worklists at random
        for i, obs in enumerate(binmat):
            obs_idx = np.where(obs)[0]
            n_conds = obs_idx.shape[0]
            if n_conds == 1:
                conds_worklist[i, 0] = obs_idx[0]
                condidx_worklist[i, 0] = -2
            else:
                np.random.seed(1)
                np.random.shuffle(obs_idx)
                conds_worklist[i, :n_conds] = obs_idx

    # If including duplicates choose a single 1-duplicate location
    n_ind_conds = binmat.sum(axis=1)
    where_1cond = np.where(n_ind_conds == 1)[0]
    where_multcond = np.where(n_ind_conds > 1)[0]
    condidx_worklist[where_1cond, 0] = -2
    if random_duplicates:
        # Choose a random selection of individuals with multiple conditions
        # to have a 1-duplicate
        random.seed(seed - 1)
        select_dupl_idxs = np.array(
            random.sample(
                list(np.arange(where_multcond.shape[0])),
                k=where_multcond.shape[0] // 2,
            )
        )

        # Select number of duplicates, default to 1 now
        where_dupl = np.arange(N_cols - 1)
        np.random.seed(seed)
        n_dupl = np.random.randint(low=1, high=2)

        # Loop over dupl indexes
        for i in range(select_dupl_idxs.shape[0]):
            ind_idx = select_dupl_idxs[i]
            np.random.seed(seed + i)
            choose_dupl = where_dupl[
                np.random.randint(low=0, high=where_dupl.shape[0])
            ]
            condidx_worklist[ind_idx, :n_dupl] = choose_dupl

    # If including end of progression
    if incl_end:
        np.random.seed(seed)
        end_prog = np.random.randint(
            low=0, high=2, size=(N_obs), dtype=np.uint8
        )
    else:
        end_prog = -1 * np.ones((N_obs), dtype=np.int8)
    n_died = end_prog.sum()

    return binmat, conds_worklist, condidx_worklist, colarr, end_prog, n_died
