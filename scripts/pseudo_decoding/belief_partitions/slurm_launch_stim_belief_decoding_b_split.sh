#!/bin/bash

# Steps 3 and 4 of claude_notes/stim_belief_alignment_updated.md: the two decoder runs the
# projection analysis reads, each fit on its own half of group B.
#
# Relative to a feature X, over correct trials:
#   A: High Not X, Not Chose    B: High Not X, Chose    C: High X, Chose
#
#   S: choice decoding restricted to BeliefPartition == "High Not X"  ->  A vs. B, the stimulus
#      axis. --b_split_half 1 trains on B1 only. Full population, no --sig_unit_level, for the
#      reasons spelled out in slurm_launch_choice_all_units.sh.
#   P: preference decoding over correct, chosen trials                ->  C vs. B, and
#      --b_split_half 2 keeps B2 only. This run does double duty: it is the B2-vs-C preference
#      decoder in its own right, and its {feat}_pref_splits.pickle is what the projection reuses,
#      so the projection scores the same test trials it was scored on.
#
# The two halves are why both runs are in one launcher: B is in both contrasts, so without the
# split a B trial trains the axis and is then scored along it, which pushes B toward C and closes
# the gap the projection measures (Issue 2 of the note). Keeping the runs together is what keeps
# --b_split_half 1 and 2 from drifting apart. The assignment comes from
# stim_belief_groups.draw_b_split, seeded with --train_test_seed -- leave that at its default 42,
# or the halves stop matching each other, the cosine analysis and the single unit anova.
#
# Note this costs trials on both sides. B goes from ~65 to ~33, and since the balance step takes
# the other class down to match, the axis run drops from ~65 to ~33 trials per class and the
# preference decoder from ~47 to ~33. That is the price of removing the leak, not a bug.
#
# Writes, both beside the existing full-B runs rather than over them:
#   /data/patrick_res/choice_reward/both_FeedbackOnsetLong[_{region}]_Response_Correct_BeliefPartition_High Not X_b_split_half_1/
#   /data/patrick_res/belief_partitions/both_FeedbackOnsetLong[_{region}]_Response_Correct_Choice_Chose_pref_99th_window_filter_drift_units_b_split_half_2/
#
# 2 runs x 7 populations x 1 event x (12 true + 120 shuffle) = 1848 jobs, under the 2000 job
# cluster cap. Run this, then slurm_launch_pref_on_choice_axis_b_split.sh, which reads both.

# Default values
partition="ckpt-all"
mem="32G"

# feedback aligned only, matching what the figures in
# notebooks/20260810_visualize_pref_on_choice_axis.ipynb draw
trial_events="FeedbackOnsetLong"

# S is the stimulus axis, P the preference run. Set to just one to submit half the grid
runs="S P"

# "whole_pop" is a sentinel for the no-region-filter run
regions="whole_pop amygdala_Amy basal_ganglia_BG inferior_temporal_cortex_ITC medial_pallium_MPal lateral_prefrontal_cortex_lat_PFC anterior_cingulate_gyrus_ACgG"

# Shuffles are wanted for both runs, unlike the no-split choice runs: the stacked figure in
# notebooks/20260810_visualize_pref_on_choice_axis.ipynb draws each as its own row, with a null
# band read through belief_partitions_io.read_results.
# The two toggles let the 1848 jobs be submitted as 168 + 1680 if the queue is not empty, without
# re-running anything.
submit_true=true
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

for run in $runs; do
    # beh_filters is single quoted where it is used, because "High Not X" contains spaces --
    # extra_args="$@" drops the caller's quoting and the generated sbatch script re-parses the
    # line, so passing it in would word split it into three arguments
    if [ "$run" == "S" ]; then
        mode="choice"
        beh_filters='{"Response":"Correct","BeliefPartition":"High Not X"}'
        b_split_half=1
        sig_unit_args=""
        base_output_path="/data/patrick_res/choice_reward"
        # the full population is ~1100 units, and both the SGD fits and
        # generate_pseudo_population_v2 scale with unit count. Matches slurm_launch_choice_all_units.sh
        time_limit="240"
    else
        mode="pref"
        beh_filters='{"Response":"Correct","Choice":"Chose"}'
        b_split_half=2
        # a full-population preference decoder gives uninterpretable results, so the projection
        # ratio is accepted as being across different unit sets -- see the note's Decisions
        sig_unit_args="--sig_unit_level pref_99th_window_filter_drift"
        base_output_path="/data/patrick_res/belief_partitions"
        time_limit="180"
    fi

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
            --beh_filters '$beh_filters' \
            --b_split_half $b_split_half \
            $sig_unit_args \
            $region_args \
            --base_output_path $base_output_path"

        # 12 jobs: one per feature
        if [ "$submit_true" = true ]; then
            submit_job_array "0-11" "sb${run}${region_tag}${trial_event:0:4}" \
                "$common_args --feat_idx \$SLURM_ARRAY_TASK_ID"
        fi

        # 120 jobs: 12 features x 10 shuffle indices
        if [ "$submit_shuffles" = true ]; then
            submit_job_array "0-119" "shsb${run}${region_tag}${trial_event:0:4}" \
                "$common_args --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12))"
        fi
    done
done
done
