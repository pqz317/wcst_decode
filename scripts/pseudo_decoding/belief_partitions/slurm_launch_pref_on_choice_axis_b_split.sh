#!/bin/bash

# Step 5 of claude_notes/stim_belief_alignment_updated.md: B2 vs. C projected onto the A-vs-B1
# stimulus axis, with a threshold refit and nothing else.
#
# The leak-free version of the pref_on_choice_Response_Correct_BeliefPartition_High_Not_X runs.
# Group B is split into disjoint halves: --axis_b_split_half 1 reads the axis fit on B1 only, and
# --b_split_half 2 scores B2 only, so no trial both trains the axis and is scored along it. Without
# the split a B trial's own noise helps build the axis, so it projects further toward the Chose end
# than a fresh B trial would -- and since the ordering along this axis is A < B < C with C the
# positive class, that closes the gap being measured rather than opening it (Issue 2).
#
# Reads:
#   axis from /data/patrick_res/choice_reward/both_{event}[_{region}]_Response_Correct_BeliefPartition_High Not X_b_split_half_1/
#     -- slurm_launch_stim_belief_axis_b_split.sh
#   splits/units from /data/patrick_res/belief_partitions/both_{event}[_{region}]_..._units_b_split_half_2/
#     -- slurm_launch_stim_belief_pref_b_split.sh
# Both must have finished. Writes /data/patrick_res/choice_axis_projection_accs/, mirroring the
# preference run's dir name, under mode
# pref_on_choice_Response_Correct_BeliefPartition_High_Not_X_axis_b_split_half_1 -- so it sits
# beside the earlier no-split projections rather than over them.
#
# Runs the whole population plus each of the 6 regions of interest, feedback aligned, matching
# slurm_launch_stim_belief_axis_b_split.sh so each region's projection uses that region's axis.
# 7 populations x 1 event x (12 true + 120 shuffle) = 924 jobs.

# Default values
partition="ckpt-all"
mem="16G"
# no SGD fits here, just projections and a threshold, so lighter than the decoding runs
time_limit="180"

# feedback aligned only, matching the two runs this reads from
trial_events="FeedbackOnsetLong"
sig_unit_level="pref_99th_window_filter_drift"
beh_filters='{"Response": "Correct", "Choice": "Chose"}'

# Trials the stimulus axis was fit on, picking which choice run to read the axis from. Set here
# rather than passed in: extra_args="$@" drops the inner quoting and the json would be word split.
axis_beh_filters='{"Response": "Correct", "BeliefPartition": "High Not X"}'

# halves of group B: score B2, read the axis fit on B1. These must be complementary
b_split_half=2
axis_b_split_half=1

# "whole_pop" is a sentinel for the no-region-filter run
regions="whole_pop amygdala_Amy basal_ganglia_BG inferior_temporal_cortex_ITC medial_pallium_MPal lateral_prefrontal_cortex_lat_PFC anterior_cingulate_gyrus_ACgG"

# The projection's shuffles are the null for the whole analysis -- true axis, permuted preference
# labels -- so they are not optional. The toggles exist only to submit the 924 jobs as 84 + 840
# if the queue is not empty.
submit_true=true
submit_shuffles=true

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
            --b_split_half $b_split_half \
            --axis_b_split_half $axis_b_split_half \
            $region_args"

        # 12 jobs: one per feature
        if [ "$submit_true" = true ]; then
            submit_job_array "0-11" "psb${region_tag}${trial_event:0:4}" \
                "$common_args --feat_idx \$SLURM_ARRAY_TASK_ID"
        fi

        # 120 jobs: 12 features x 10 shuffle indices
        if [ "$submit_shuffles" = true ]; then
            submit_job_array "0-119" "shpsb${region_tag}${trial_event:0:4}" \
                "$common_args --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12))"
        fi
    done
done
