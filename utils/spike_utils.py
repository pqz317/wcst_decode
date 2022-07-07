import numpy as np
import pandas as pd


def get_spikes_by_trial_interval(spike_times, intervals):
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
            # spike not in current interval, move to next spike
            spike_times_idx += 1
            continue
        if spike_time >= interval_start and spike_time < interval_end:
            # add spike, move on to next spike
            spikes_by_trial.append([
                interval_row.TrialNumber, 
                spike_row.UnitID, 
                spike_time, 
                spike_time - interval_start
            ])
            spike_times_idx += 1
        if spike_time >= interval_end:
            # spike later than interval, move on to next interval
            # don't move on to next spike
            interval_idx +=1
    return pd.DataFrame(spikes_by_trial, columns=["TrialNumber", "UnitID", "SpikeTime", "SpikeTimeFromStart"])

