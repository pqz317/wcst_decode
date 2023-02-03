import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
)
from scipy.ndimage import gaussian_filter1d


def get_spikes_by_trial_interval_DEPRECATED(spike_times, intervals):
    """Finds all the spikes within a series of time intervals

    Args:
        spike_times: Dataframe with columns: SpikeTime, UnitID
        intervals: Dataframe with columns: TrialNumber,
            IntervalStartTime, IntervalEndTime

    Returns:
        DataFrame with columns: TrialNumber, UnitID, SpikeTime,
        SpikeTimeFromStart
    """
    # columns: TrialNumber, UnitID, SpikeTime, SpikeTimeFromStart
    spikes_by_trial = []
    # make sure both dfs are sorted by time
    spike_times = spike_times.sort_values(by=["SpikeTime"])
    intervals = intervals.sort_values(by=["IntervalStartTime"])

    # enter looping
    interval_idx = 0
    spike_times_idx = 0
    while interval_idx < len(intervals) and spike_times_idx < len(spike_times):
        spike_row = spike_times.iloc[spike_times_idx]
        spike_time = spike_row.SpikeTime
        interval_row = intervals.iloc[interval_idx]
        interval_start = interval_row.IntervalStartTime
        interval_end = interval_row.IntervalEndTime
        if spike_time < interval_start:
            # spike before current interval, move to next spike
            spike_times_idx += 1
        elif spike_time >= interval_end:
            # spike later than interval, move on to next interval
            # don't move on to next spike
            interval_idx +=1
        else:
            # spike time within interval
            # add spike, move on to next spike
            spikes_by_trial.append([
                interval_row.TrialNumber, 
                spike_row.UnitID, 
                spike_time, 
                spike_time - interval_start
            ])
            spike_times_idx += 1

    return pd.DataFrame(spikes_by_trial, columns=["TrialNumber", "UnitID", "SpikeTime", "SpikeTimeFromStart"])    


def get_spikes_by_interval(spike_times, intervals):
    """Finds all the spikes within a series of time intervals

    Args:
        spike_times: Dataframe with columns: SpikeTime, UnitID
        intervals: Dataframe with columns: IntervalID,
            IntervalStartTime, IntervalEndTime

    Returns:
        DataFrame with columns: IntervalID, UnitID, SpikeTime,
        SpikeTimeFromStart
    """

    def find_spikes_for_interval(interval):
        left_idx = spike_times.SpikeTime.searchsorted(interval.IntervalStartTime, side="left")
        right_idx = spike_times.SpikeTime.searchsorted(interval.IntervalEndTime, side="right")
        num_intervals = right_idx - left_idx
        spike_times_in_interval = spike_times[left_idx:right_idx]
        return spike_times_in_interval.assign(
            IntervalID = [interval.IntervalID] * num_intervals
        ).assign(
            SpikeTimeFromStart = spike_times_in_interval.SpikeTime - interval.IntervalStartTime
        )

    spike_times = spike_times.sort_values('SpikeTime')

    interval_spikes = intervals.apply(find_spikes_for_interval, axis=1)
    return pd.concat(interval_spikes.tolist())


def get_spikes_by_trial_interval(spike_times, trial_intervals):
    """Hacky workaround. Uses the get_spikes_by_interval implementation which is faster, 
    but preserves the 'TrialNumber' column naming which is helpful downstream. 

    Args:
        spike_times: Dataframe with columns: SpikeTime, UnitID
        trial_intervals: Dataframe with columns: TrialNumber,
            IntervalStartTime, IntervalEndTime

    Returns:
        DataFrame with columns: TrialNumber, UnitID, SpikeTime,
        SpikeTimeFromStart
    """
    intervals = trial_intervals.rename(columns={"TrialNumber": "IntervalID"})
    spikes_by_interval = get_spikes_by_interval(spike_times, intervals)
    spikes_by_interval.rename(columns={"IntervalID": "TrialNumber"}, inplace=True)
    return spikes_by_interval



def get_firing_rates_by_interval(spData, bins, smoothing):
    # spData is pandas dataframe with at least IntervalID, UnitId, and SpikeTimeFromStart columns
    trial_unit_index = pd.MultiIndex.from_product([np.unique(spData.IntervalID), np.unique(spData.UnitID), bins[:-1]], names=["IntervalID", "UnitID", "TimeBins"]).to_frame()
    trial_unit_index = trial_unit_index.droplevel(2).drop(columns=["IntervalID", "UnitID"]).reset_index()
    
    groupedData = spData.groupby(["IntervalID", "UnitID"])

    fr_DF = groupedData.apply(lambda x: pd.DataFrame(\
                            {"SpikeCounts": np.histogram(x.SpikeTimeFromStart/1000, bins)[0],\
                            "FiringRate": gaussian_filter1d(np.histogram(x.SpikeTimeFromStart/1000, bins)[0].astype(float), smoothing),\
                            "TimeBins": bins[:-1]}))
    all_units_df = trial_unit_index.merge(fr_DF.droplevel(2).reset_index(), how='outer', on=["IntervalID", "UnitID", "TimeBins"])
    all_units_df.FiringRate = all_units_df.FiringRate.fillna(0.0)
    all_units_df.SpikeCounts = all_units_df.SpikeCounts.fillna(0)
    return all_units_df

def get_temporal_drive_unit_ids(fs, subject, session):
    unit_info = spike_general.list_session_units(fs, subject, session)
    temp_units = unit_info[~unit_info["Channel"].str.contains("a")].UnitID.unique().astype(int)
    return temp_units

    
def get_anterior_drive_unit_ids(fs, subject, session):
    unit_info = spike_general.list_session_units(fs, subject, session)
    ant_units = unit_info[unit_info["Channel"].str.contains("a")].UnitID.unique().astype(int)
    return ant_units

def get_stats_for_units(firing_rates):
    """Per unit, calculate mean, variance for SpikeCounts, FiringRate
    Across all trials and time intervals

    Args:
        firing_rates: df with UnitID, SpikeCounts, FiringRate columns
    
    Returns:
        stats: df with UnitID, SpikeCountMean, FiringRateMean, SpikeCountVar, FiringRateVar columns
    """
    def calc_var_for_unit(unit):
        firing_rate_mean = np.mean(unit.FiringRate)
        spike_count_mean = np.mean(unit.SpikeCounts)
        firing_rate_var = np.var(unit.FiringRate)
        spike_count_var = np.var(unit.SpikeCounts)
        return pd.Series(
            [firing_rate_mean, spike_count_mean, firing_rate_var, spike_count_var], 
            index=["FiringRateMean", "SpikeCountMean", "FiringRateVar", "SpikeCountVar"]
        )
    return firing_rates.groupby(["UnitID"]).apply(calc_var_for_unit).reset_index()

def get_unit_fr_array(frs, column_name):
    """ Given a dataframe with columns UnitID, column_name specified, 
    Returns a num_units x num_time_bins numpy array
    Args:
        frs: dataframe with columns UnitID, column_name, 
    Returns:
        np array of dims: num_units x num_time_bins
    """
    grouped = frs[["UnitID", column_name]].groupby(by="UnitID").agg(list).to_numpy()
    return np.stack(grouped.squeeze(), axis=0)