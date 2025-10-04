import numpy as np
import pandas as pd
from spike_tools import (
    general as spike_general,
)
from scipy.ndimage import gaussian_filter1d
import json
from sklearn.linear_model import LinearRegression
from constants.behavioral_constants import *
from constants.decoding_constants import *

DEFAULT_SESS_INFO_PATH = "/data/rawdata/sub-{subject}/sess-{session}/session_info/sub-{subject}_sess-{session}_sessioninfomodified.json"
DEFAULT_FR_PATH = "/data/patrick_res/firing_rates/SA/{session}_firing_rates_1300_FeedbackOnset_1500_100_bins_1_smooth.pickle"

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



def get_unit_positions_per_sess(session, subject="SA", fr_path=DEFAULT_FR_PATH, sess_info_path=DEFAULT_SESS_INFO_PATH):
    info_path = sess_info_path.format(
        subject=subject,
        session=session,
    )
    with open(info_path, 'r') as f:
        data = json.load(f)
    locs = data['electrode_info']
    locs_df = pd.DataFrame.from_dict(locs)
    # switch types because some entries can be a list
    # NOTE: hack to convert relevant fields to str first
    locs_df = locs_df.astype({
        "structure_level1": "str",
        "structure_level2": "str",
        "structure_level3": "str",
        "structure_level4": "str",
        "structure_level5": "str"
    })
    locs_df = locs_df.replace("[]", None)

    # ensure a position exists
    electrode_pos_not_nan = locs_df[~locs_df['x'].isna() & ~locs_df['y'].isna() & ~locs_df['z'].isna()]
    # grab unit to electrode mapping
    units = spike_general.list_session_units(None, subject, session, species_dir="/data")
    unit_pos = pd.merge(units, electrode_pos_not_nan, left_on="Channel", right_on="electrode_id", how="left")
    unit_pos = unit_pos.astype({"UnitID": int})
    unit_pos["session"] = session
    if fr_path is not None: 
        session_fr_path = fr_path.format(
            session=session,
        )
        frs = pd.read_pickle(session_fr_path)
        unit_pos = unit_pos[unit_pos.UnitID.isin(frs.UnitID.unique())]
    unit_pos["PseudoUnitID"] = int(session) * 100 + unit_pos["UnitID"]
    return unit_pos


USE_STRUCTURE_2 = ["diencephalon (di)", "metencephalon (met)", "telencephalon (tel)", ""]

LEVEL_2_TO_MANUALS = {
    "posterior_medial_cortex (PMC)": "Parietal Cortex",
    "motor_cortex (motor)": "Premotor Cortex",
    "orbital_frontal_cortex (OFC)": "Prefrontal Cortex",
    "lateral_prefrontal_cortex (lat_PFC)": "Prefrontal Cortex",
    "anterior_cingulate_gyrus (ACgG)": "Anterior Cingulate Gyrus",
    "lateral_and_ventral_pallium (LVPal)": "Claustrum",
    "amygdala (Amy)": "Amygdala",
    "inferior_temporal_cortex (ITC)": "Hippocampus/MTL",
    "preoptic_complex (POC)": "Hippocampus/MTL",
    "basal_ganglia (BG)": "Basal Ganglia",
    "primary_visual_cortex (V1)": "Visual Cortex",
    "inferior_parietal_lobule (IPL)": "Parietal Cortex",
    "medial_pallium (MPal)": "Hippocampus/MTL",
    "superior_parietal_lobule (SPL)": "Parietal Cortex",
    "extrastriate_visual_areas_2-4 (V2-V4)": "Visual Cortex",
    "thalamus (Thal)": "unknown",
    # "floor_of_the_lateral_sulcus (floor_of_ls)": "Hippocampus/MTL",
    "floor_of_the_lateral_sulcus (floor_of_ls)": "unknown",
    "diagonal_subpallium (DSP)": "unknown",
    "cerebellum (Cb)": "unknown",
    "medial_temporal_lobe (MTL)": "Hippocampus/MTL",
    "somatosensory_cortex (SI/SII)": "Parietal Cortex",
    "core_and_belt_areas_of_auditory_cortex (core/belt)": "Hippocampus/MTL",
    "unknown": "unknown",
}



def get_manual_structure(positions):
    """
    Curates a structure level that combines levels 1 and 2, replacing any *phalon 
    structures with their level 2 substructures
    """
    positions["manual_structure"] = positions.apply(lambda x: LEVEL_2_TO_MANUALS[x.structure_level2], axis=1)
    return positions

def get_unit_positions(
    sessions, subject="SA", 
    get_manual_regions=True, 
    fr_path=DEFAULT_FR_PATH, 
    sess_info_path=DEFAULT_SESS_INFO_PATH
):
    """
    For each session, finds unit positions, concatenates
    """
    positions = pd.concat(sessions.apply(
        lambda x: get_unit_positions_per_sess(x.session_name, subject, fr_path, sess_info_path), 
        axis=1
    ).values)
    # still want to plot the None units
    positions = positions.fillna("unknown")
    # clean structure_level2 to not inlcude spaces or parentheses:
    positions["structure_level2_cleaned"] = (positions["structure_level2"]
        .str.replace(r'[()]', '', regex=True)
        .str.replace(' ', '_')
    )
    if get_manual_regions:
        positions = get_manual_structure(positions)
        positions["manual_structure_cleaned"] = positions.manual_structure.apply(lambda x: x.replace(" ", "_").replace("/", "_"))

    positions["drive"] = positions.Channel.apply(lambda x: "Anterior" if "a" in x else "Temporal")
    return positions

def get_subpop_ratios_by_region(subpop, valid_sess):
    all_pop = get_unit_positions(valid_sess)
    subpop_counts = subpop.groupby("manual_structure").count()["UnitID"].rename("SubpopCount")
    all_pop_counts = all_pop.groupby("manual_structure").count()["UnitID"].rename("PopCount")
    merged = pd.merge(subpop_counts, all_pop_counts, on="manual_structure", how="outer")
    merged = merged.fillna(0.0)
    merged["Ratio"] = merged["SubpopCount"] / merged["PopCount"]
    merged = merged.sort_values("Ratio", ascending=False)
    return merged

def zscore_frs(frs, group_cols=["UnitID"], mode="FiringRate"):
    def zscore_unit(group):
        mean = group[mode].mean()
        std = group[mode].std()
        # if std is 0, just set to 0. 
        group[mode] = np.nan_to_num((group[mode] - mean) / std)
        return group
    return frs.groupby(group_cols).apply(zscore_unit).reset_index(drop=True)

def mean_sub_frs(frs, group_cols=["UnitID"], mode="SpikeCounts"):
    def mean_sub_unit(group):
        mean = group[mode].mean()
        group[f"MeanSub{mode}"] = group[mode] - mean
        return group
    return frs.groupby(group_cols).apply(mean_sub_unit).reset_index(drop=True)

def block_lowest_val_sub_frs(frs, mode="SpikeCounts"):
    """
    Finds the trial with the lowest max value in the block, 
    uses firing rate at that trial as reference
    """
    def block_sub_unit(group):
        lowest_idx = group.MaxValue.idxmin()
        lowest_fr = group.loc[lowest_idx][mode]
        group[f"LowestSub{mode}"] = group[mode] - lowest_fr
        return group
    return frs.groupby(["UnitID", "BlockNumber"]).apply(block_sub_unit).reset_index(drop=True)

def get_avg_fr_per_interval(frs):
    return frs.groupby(["UnitID", "TrialNumber"]).mean().reset_index()

def filter_drift(units, subject):
    drift_units = pd.read_pickle(DRIFT_PATH.format(sub=subject))
    return units[~units.PseudoUnitID.isin(drift_units.PseudoUnitID)]

def filter_bad_regions(units):
    return units[~units.structure_level2_cleaned.isin(BAD_REGIONS)]

def get_subject_units(subject):
    if subject == "BL":
        units_path = BL_CORRECTED_UNITS_PATH.format(sub=subject)
    else: 
        units_path = UNITS_PATH.format(sub=subject)
    return pd.read_pickle(units_path)

def get_region_units(region_level, regions, units):
    if region_level is None or regions is None: 
        return units
    regions_arr = regions.split(",")
    return units[units[region_level].isin(regions_arr)].copy()

def get_all_region_units(region_level, regions, subjects=["SA", "BL"]):
    """
    Gets all pseudounitids associated with a region, from both subjects
    optionally filters for drifting units
    """
    all_units = []
    for sub in subjects: 
        units = get_subject_units(sub)
        units = get_region_units(region_level, regions, units)
        units = filter_drift(units, sub)
        all_units.append(units)
    return pd.concat(all_units).PseudoUnitID.unique()

def get_sig_units(args, units=None):
    """
    Grabs significant units per features, filters units to match. 
    Returns back list of units
    """
    if not args.sig_unit_level: 
        return units
    sig_path = SIG_UNITS_PATH.format(
        sub=args.subject,
        event=args.trial_event,
        level=args.sig_unit_level,
    )
    sig_units = pd.read_pickle(sig_path)

    if "feat_pair" in args:
        feat1, feat2 = args.feat_pair
        feat_sig_units = sig_units[(sig_units.feat == feat1) | (sig_units.feat == feat2)]
    elif "feat" in args:
        feat_sig_units = sig_units[sig_units.feat == args.feat]
    else: 
        raise ValueError("args has neither feat or feat_pair")
    if units is not None: 
        return feat_sig_units[feat_sig_units.PseudoUnitID.isin(units)].copy()
    else:
        return feat_sig_units.copy()


def regress_out_trial_number(frs):
    def regress_per_unit_timebin(group):
        y = group.FiringRate
        x = group.TrialNumber.values.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x)
        group["FiringRate"] = group["FiringRate"] - y_pred
        return group
    return frs.groupby(["TimeBins", "UnitID"], group_keys=False).apply(regress_per_unit_timebin)

def white_noise_frs(frs):
    rng = np.random.default_rng()
    frs["FiringRate"] = rng.normal(size=len(frs))
    return frs

def trial_num_as_frs(frs):
    frs["FiringRate"] = frs["TrialNumber"]
    return frs

def pref_belief_as_frs(frs, beh):
    def pref_belief_per_row(row):
        target_feat = FEATURES[row.UnitID % len(FEATURES)]
        return 1 if row.PreferredBelief == target_feat else 0
    merged = pd.merge(frs, beh[["TrialNumber", "PreferredBelief"]], on="TrialNumber")
    merged["FiringRate"] = merged.apply(lambda x: pref_belief_per_row(x), axis=1)
    return merged[["TrialNumber", "UnitID", "TimeBins", "FiringRate"]]





def get_frs_from_args(args, sess_name):
    """
    """
    # need to account for corrected unit positions for BL...
    units = get_subject_units(args.subject)
    units = get_region_units(args.region_level, args.regions, units)
    units = get_sig_units(args, units)

    # should always filter drift, and bad regions

    trial_interval = args.trial_interval
    spikes_path = SESS_SPIKES_PATH.format(
        sub=args.subject,
        sess_name=sess_name, 
        fr_type=args.fr_type,
        pre_interval=trial_interval.pre_interval, 
        event=trial_interval.event, 
        post_interval=trial_interval.post_interval, 
        interval_size=trial_interval.interval_size
    )
    frs = pd.read_pickle(spikes_path)
    frs["PseudoUnitID"] = int(sess_name) * 100 + frs.UnitID.astype(int)
    if units is not None: 
        frs = frs[frs.PseudoUnitID.isin(units.PseudoUnitID)]
    # create a time field as well that's relative to the trial event
    frs["Time"] = frs["TimeBins"] - args.trial_interval.pre_interval / 1000
    if hasattr(args, "time_range") and args.time_range is not None: 
        if len(args.time_range) !=2: 
            raise ValueError("must have two ranges")
        # time_range specified in milliseconds, relative to trial event, convert to 
        # be in seconds, relative to pre_interval
        start, end = [(x + args.trial_interval.pre_interval) / 1000 for x in args.time_range]
        frs = frs[(frs.TimeBins >= start) & (frs.TimeBins < end)]
    return frs

def find_peaks(df, value_col, region_level="whole_pop", time_col="Time"):
    """
    Given a df with PseudoUnitID, some value column, and some time column
    Finds peak times, the time which each unit has the highest value
    Returns a df with PseudoUnitID, peak_times, and a sorted list of PseudoUnitIDs
    """
    region_peaks = {}
    region_orders = {}
    def find_peak(group):
        row_idx = group[value_col].idxmax()
        return group.loc[row_idx][time_col]
    
    def process_per_region(region_df):
        peaks = region_df.groupby("PseudoUnitID").apply(find_peak).reset_index(name="peak_time")
        orders = peaks.sort_values(by="peak_time").PseudoUnitID
        region_peaks[region_df.name] = peaks
        region_orders[region_df.name] = orders

    df.groupby(region_level).apply(process_per_region)
    return region_peaks, region_orders