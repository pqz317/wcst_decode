#!/bin/bash

# Choice decoding (Chose X vs. Not Chose X) on the FULL population -- no selectivity subpop,
# drift-filtered only (sig_unit_level=all_filter_drift).
#
# Motivation: the choice_99th_window_filter_drift subpop covers only 52% of the units in the
# pref_99th_window_filter_drift subpop, so a choice axis fit on it has no weight for ~half the
# preference population. Fitting on all (drift-filtered) units means the choice axis restricted
# to preference units is a genuine sub-vector, with nothing zero-filled.
#
# Runs the whole population plus each of the 6 regions of interest, both trial events.
#
# WARNING: writes into the same canonical directories as the existing June-2025 runs, i.e.
#   /data/patrick_res/choice_reward/both_{event}[_{region}]_all_filter_drift_units/
# For 10 of the 14 configs those directories already contain {feat}_choice_* files, which WILL be
# overwritten. Files for other modes (reward, chose_and_correct) in those directories are untouched.
# Only ACgG and the whole-population configs are new.

# Default values
partition="ckpt-all"
mem="32G"
# bumped from the usual 180: the full population is ~1100 units vs ~341 for the choice subpop,
# and both the SGD fits and generate_pseudo_population_v2 scale with unit count
time_limit="240"

trial_events="StimOnset FeedbackOnsetLong"
mode="choice"
sig_unit_level="all_filter_drift"

# "whole_pop" is a sentinel for the no-region-filter run
regions="whole_pop amygdala_Amy basal_ganglia_BG inferior_temporal_cortex_ITC medial_pallium_MPal lateral_prefrontal_cortex_lat_PFC anterior_cingulate_gyrus_ACgG"

# The choice-axis projection analysis uses TRUE axes with session-permuted preference labels, so it
# does not need choice shuffles. Set to false to skip the 1680 shuffle jobs and submit only 168.
submit_shuffles=true

# Optional args passed to decoding script
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
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/decode_belief_partitions.py $python_args $extra_args
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
        common_args="--mode $mode --trial_event $trial_event \
            --subject both \
            --sig_unit_level $sig_unit_level \
            $region_args \
            --base_output_path /data/patrick_res/choice_reward"

        # 12 jobs: one per feature
        submit_job_array "0-11" "c${region_tag}${trial_event:0:4}" \
            "$common_args --feat_idx \$SLURM_ARRAY_TASK_ID"

        # 120 jobs: 12 features x 10 shuffle indices
        if [ "$submit_shuffles" = true ]; then
            submit_job_array "0-119" "shc${region_tag}${trial_event:0:4}" \
                "$common_args --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12))"
        fi
    done
done
