#!/bin/bash

# Choice decoding (Chose X vs. Not Chose X) on the FULL population -- no selectivity subpop.
#
# Motivation: the choice_99th_window_filter_drift subpop covers only 52% of the units in the
# pref_99th_window_filter_drift subpop, so a choice axis fit on it has no weight for ~half the
# preference population. Fitting on all units means the choice axis restricted to preference
# units is a genuine sub-vector, with nothing zero-filled.
#
# Runs the whole population plus each of the 6 regions of interest, both trial events.
#
# Deliberately passes NO --sig_unit_level. get_sig_units() returns the units table untouched when
# the level is unset, and get_frs_from_args() still applies filter_bad_regions + filter_drift
# afterwards -- so this is "all units, bad regions and drifting units removed", which is what
# all_filter_drift was meant to mean.
#
# Do NOT use --sig_unit_level all_filter_drift: those pickles are stale artifacts whose
# sig_type="all" branch in 20250606_generate_significant_units.py selects only
# ["feat", "structure_level2", "session", "PseudoUnitID"], dropping structure_level2_cleaned, so
# filter_bad_regions() raises AttributeError. They are also built from all_units.pickle for BOTH
# subjects, while get_subject_units() reads all_units_corrected.pickle for BL -- so BL region
# labels in those files disagree with the canonical table.
#
# Output dirs are therefore /data/patrick_res/choice_reward/both_{event}[_{region}]/ -- no
# collision with the existing *_units runs, so nothing is overwritten.

# Default values
partition="ckpt-all"
mem="32G"
# bumped from the usual 180: the full population is ~1100 units vs ~341 for the choice subpop,
# and both the SGD fits and generate_pseudo_population_v2 scale with unit count
time_limit="240"

trial_events="StimOnset FeedbackOnsetLong"
mode="choice"

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
