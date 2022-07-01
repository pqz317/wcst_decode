import numpy as np
import pandas as pd


def get_spikes_by_trial_interval(spike_times, intervals):
    def spikes_times_in_interval(interval):
        print(interval.IntervalStartTime)
        print(interval)
        start_time = interval["IntervalStartTime"]
        end_time = interval["IntervalEndTime"]
        spikes_in_interval = spike_times[ \
            (spike_times["SpikeTime"] >= start_time) & \
            (spike_times["SpikeTime"] < end_time) \
        ]
        spikes_in_interval["TrialNumber"] = interval["TrialNumber"]
        spikes_in_interval["SpikeTimeFromStart"] = spike_times["SpikeTime"] - start_time
    applied = intervals.apply(spikes_times_in_interval, axis=1)
    return applied


def get_spikes_by_trial_interval(spike_times, intervals):
    # columns: TrialNumber, UnitID, SpikeTime, SpikeTimeFromStart
    spikes_by_trial = []
    # make sure both dfs are sorted by time
    spike_times = spike_times.sort_values(by=["SpikeTime"])
    intervals = intervals.sort_values(by=["IntervalStartTime"])

    print("Finished sorting, entering loop")

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

