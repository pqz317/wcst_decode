import numpy as np
import pandas as pd

def generate_pseudo_population(frs, split, num_train_samples=1000, num_test_samples=100, rng=None):
    """
    Generates a psuedo population of spikes using a train/test split of conditions
    Uses method detailed in Bernardi., 2020. 
    Args:
        frs: a Dataframe of firing rate data with columns UnitID, TrialNumber, TimeBins, and some data column
            (eg. SpikeCounts or FiringRates)
        split: a Dataframe of trials split by train/test per condition. Columns: 
            - Condition: specific conditions train/test trials are selected for (eg. Rule)
            - TrainTrials: list of train trials for condition. 
            - TestTrials: list of test trials for condition. 
        num_train_samples: number of training samples to generate per condtion
        num_test_samples: number of testing samples to generate per condition
        rng: random number generator to be used. If None, will initialize one with no seed. 
    Returns: 
        a Dataframe containing pseudo population data, with columns:
            - UnitID: ID of unit (neuron) selected
            - TrialNumber: ID of original trial selected from
            - PseudoTrialNumber: ID of the pseudo trial generated
            - Type: Either Train or Test 
            - Condition
            - TimeBins
            - some data column (eg. SpikeCounts or FiringRates)
    """
    if rng is None:
        rng = np.random.default_rng()

    unit_ids = frs.UnitID.unique()
    num_units = len(unit_ids)
    num_conditions = len(split)
    samples_for_conditions = []
    conditions = []
    for _, row in split.iterrows():
        # per condition, sample from train and test trials
        condition = row["Condition"]
        train_trials = row["TrainTrials"]
        test_trials = row["TestTrials"]
        train_samples = rng.choice(train_trials, num_train_samples * num_units)
        test_samples = rng.choice(test_trials, num_test_samples * num_units)
        samples_for_conditions.append(train_samples)
        samples_for_conditions.append(test_samples)
        conditions.append(condition)
    trial_samples = np.concatenate(samples_for_conditions)
    num_pseudo_trials =  (num_train_samples + num_test_samples) * num_conditions
    tiled_unit_ids = np.tile(unit_ids, num_pseudo_trials)
    pseudo_trial_nums = np.repeat(np.arange(num_pseudo_trials), num_units)
    pseudo_trial_types = np.tile(
        np.concatenate((
            np.repeat(["Train"], num_train_samples * num_units), 
            np.repeat(["Test"], num_test_samples * num_units)
        )),
        num_conditions
    )
    repeat_conditions = np.repeat(conditions, (num_train_samples + num_test_samples) * num_units)
    pseudo_df = pd.DataFrame({
        "UnitID": tiled_unit_ids,
        "TrialNumber": trial_samples,
        "PseudoTrialNumber": pseudo_trial_nums,
        "Type": pseudo_trial_types,
        "Condition": repeat_conditions
    })
    num_time_bins = len(frs.TimeBins.unique())
    pseudo_pop = pd.merge(pseudo_df, frs, "inner", on=["UnitID", "TrialNumber"])
    if not len(pseudo_pop) == num_pseudo_trials * num_units * num_time_bins: 
        raise ValueError("Did not get expected number of rows in pseudo population")
    return pseudo_pop

def generate_multi_split_pseudo_population(frs, splitter, num_splits, num_train_samples=1000, num_test_samples=100, rng=None): 
    """
    Generates a psuedo population of spikes using for a number of train/test splits
    Uses method detailed in Bernardi., 2020. 

    NOTE: This is probably very memory inefficient, though works with existing code well. 

    Args:
        frs: a Dataframe of firing rate data with columns UnitID, TrialNumber, TimeBins, and some data column
            (eg. SpikeCounts or FiringRates)
        splitter: a iterator to generate train/test splits
        num_splits: number of train/test splits to generate
        num_train_samples: number of training samples to generate per condtion
        num_test_samples: number of testing samples to generate per condition
        rng: random number generator to be used. If None, will initialize one with no seed. 
    Returns: 
        a Dataframe containing pseudo population data, with columns:
            - UnitID: ID of unit (neuron) selected
            - TrialNumber: ID of original trial selected from
            - PseudoTrialNumber: ID of the pseudo trial generated
            - Type: Either Train or Test 
            - Condition
            - TimeBins
            - some data column (eg. SpikeCounts or FiringRates) 
            - SplitNumber: which train/test split this row is referring to. 
    """
    pseudo_pops = []
    for split_num in np.arange(num_splits):
        split = next(splitter)
        pseudo_pop = generate_pseudo_population(frs, split, num_train_samples, num_test_samples, rng)
        # make sure PseudoTrialNumber unique across all train/test splits
        pseudo_pop["PseudoTrialNumber"] = pseudo_pop["PseudoTrialNumber"] + split_num * len(pseudo_pop.PseudoTrialNumber.unique())
        pseudo_pop["SplitNum"] = split_num
        pseudo_pops.append(pseudo_pop)
    return pd.concat(pseudo_pops)
    