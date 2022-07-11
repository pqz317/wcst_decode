import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
)

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


def get_temporal_drive_unit_ids(fs, subject, session):
    unit_info = spike_general.list_session_units(fs, subject, session)
    return unit_info[~unit_info["Channel"].str.contains("a")].UnitID.unique().astype(int)

    
def get_anterior_drive_unit_ids(fs, subject, session):
    unit_info = spike_general.list_session_units(fs, subject, session)
    return unit_info[unit_info["Channel"].str.contains("a")].UnitID.unique().astype(int)