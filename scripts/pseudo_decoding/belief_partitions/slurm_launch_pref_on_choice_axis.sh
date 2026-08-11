#!/bin/bash

# Projects preference activity (High X vs. High Not X, in Correct/Chose trials) onto the choice
# axis, to compare how much preference decodability the choice axis captures.
#
# Reads:
#   preference splits/units from /data/patrick_res/belief_partitions/both_{event}[_{region}]_...
#   choice axes from /data/patrick_res/choice_reward/both_{event}[_{region}]_Response_Correct/
#     (all units, correct trials only -- see axis_beh_filters below)
# Writes:
#   /data/patrick_res/choice_axis_projection_accs/, mirroring the preference run's dir name, under
#   mode pref_on_choice_Response_Correct so it sits beside the earlier all-trials-axis results
#
# Runs the whole population plus each of the 6 regions of interest, both trial events, matching
# slurm_launch_choice_all_units.sh so each region's projection uses that region's choice axis.

# Default values
partition="ckpt-all"
mem="16G"
# no SGD fits here, just projections and a threshold, so lighter than the decoding runs
time_limit="180"

trial_events="StimOnset FeedbackOnsetLong"
sig_unit_level="pref_99th_window_filter_drift"
beh_filters='{"Response": "Correct", "Choice": "Chose"}'

# Trials the choice axis was fit on, picking which choice run to read the axis from.
# '{}' reproduces the original all-trials axis runs, stored as mode pref_on_choice.
# '{"Response": "Correct"}' uses the correct-only choice re-runs, stored as mode
# pref_on_choice_Response_Correct -- the better matched axis, since the projected preference
# trials are themselves Correct + Chose.
# Set here rather than passed in: extra_args="$@" drops the inner quoting and the json would be
# word split.
axis_beh_filters='{"Response": "Correct"}'

# "whole_pop" is a sentinel for the no-region-filter run
regions="whole_pop amygdala_Amy basal_ganglia_BG inferior_temporal_cortex_ITC medial_pallium_MPal lateral_prefrontal_cortex_lat_PFC anterior_cingulate_gyrus_ACgG"

# Optional args passed to the projection script
extra_args="$@"

# Function to submit a job array
submit_job_array () {
    local array_range=$1
    local job_name=$2
    local python_args=$3
    sbatch --array="$array_range" <<EOT;
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH -p $partition
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=$mem
#SBATCH --time=$time_limit

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/decode_pref_on_choice_axis.py $python_args $extra_args
EOT
}

for region in $regions; do
    if [ "$region" == "whole_pop" ]; then
        region_args=""
        region_tag="all"
    else
        region_args="--region_level structure_level2_cleaned --regions $region"
        # short tag for the slurm job name, e.g. inferior_temporal_cortex_ITC -> ITC
        region_tag="${region##*_}"
    fi

    for trial_event in $trial_events; do
        common_args="--trial_event $trial_event \
            --subject both \
            --sig_unit_level $sig_unit_level \
            --beh_filters '$beh_filters' \
            --axis_beh_filters '$axis_beh_filters' \
            $region_args"

        # 12 jobs: one per feature
        submit_job_array "0-11" "p${region_tag}${trial_event:0:4}" \
            "$common_args --feat_idx \$SLURM_ARRAY_TASK_ID"

        # 120 jobs: 12 features x 10 shuffle indices
        submit_job_array "0-119" "shp${region_tag}${trial_event:0:4}" \
            "$common_args --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12))"
    done
done
