#!/bin/bash

# Per-unit ANOVA for the single-unit test of H1 vs H3, claude_notes/stim_belief_single_unit_anova_lite.md.
#
# The population analyses of stim_belief_alignment_updated.md come out null everywhere, but the
# cosine they report uses only the SIGN of each unit's contribution, so it is equally null under H1
# (disjoint stimulus and belief codes) and H3 (mixed selectivity with random sign). This grid
# measures the sign-free half: a per-unit fraction of variance for each contrast, whose joint
# distribution across units separates the two. See section 1.2 of that note.
#
# Two runs, one binary factor each, over the A/B/C design's groups:
#
#   Run S -- stimulus:  A (X not chosen) vs B1 (X chosen), both X-not-preferred, factor Choice
#   Run P -- belief:    B2 (X not preferred) vs C (X preferred), both X-chosen, factor BeliefPartition
#
# Group B is SPLIT: Run S sees B1 and Run P sees B2, so the two runs share no trials at all. This is
# not a refinement. Both runs' per-unit contrasts contain B, so sharing it would put B's sampling
# noise in both, and since the ANOVA statistic is a function of the SQUARED contrast, a negative
# correlation of the signed effects becomes a POSITIVE correlation of the magnitudes -- section 3.1
# of the note puts the resulting null overlap ratio at 5.16 (alpha=0.05) to 15.8 (alpha=0.01)
# instead of 1, well above anything the data can produce. Every unit would look co-selective and H1
# would be rejected by arithmetic. The split is drawn by
# stim_belief_vector_alignment.draw_b_split, so on the true run it is the identical partition the
# population analysis used.
#
# --shuffle_method circular_shift is what pairs the two runs' shuffles. The shift is drawn from
# default_rng(shuffle_idx) and applied to the full session behavior BEFORE any filtering, which is
# identical across the two runs, so Run S's shuffle j and Run P's shuffle j are the same shift of
# the same session. That pairing is what makes the downstream null a JOINT one, able to absorb the
# drift x belief-autocorrelation coupling the B split does not touch (section 3.3).
#
# --use_x True is required, not cosmetic: without it BeliefPartition holds "High Not CIRCLE" rather
# than "High Not X" and Run S's filter silently matches zero trials.
#
# Reads out as x_Choice_comb_time_fracvar (Run S) and x_BeliefPartition_comb_time_fracvar (Run P).
# Crossed by scripts/anova_analysis/stim_belief_unit_overlap.py; 20250329_aggregate_shuffles.py is
# NOT needed, since that analysis uses exact permutation p-values rather than percentile columns.
#
# Writes /data/patrick_res/anova/{sub}_{event}_{conds}_{filters}_window_500_b_split_half_{n}/,
# shuffles in .../circular_shift_shuffles/. ~9 GB total.
#
# Check first:  python3 claude_notes/stim_belief_anova_split_check.py
# which verifies over all 300 (session, feat) pairs that the two runs are disjoint, that the halves
# partition B, and that the true-run split matches the population analysis's.

# ---------------------------------------------------------------------------------------------
# JOB COUNT. The shipped 20250329_run_anova.sh shape is one job per (feature, shuffle): 12 + 1200
# per launch, and this grid is 8 launches = 9696 jobs, ~5x a 2000-job cap. So shuffles are BATCHED
# --num_shuffles_per_job at a time in one process, each still writing its own pickle under the name
# it would have had as its own job. The on-disk layout is identical either way; only the scheduling
# changes.
#
#   1 shuffle/job    12 + 1200 per launch   9696 total   over the cap
#   10 shuffles/job  12 +  120 per launch   1056 total   fits in one submission
#
# Two more levers if the cap still bites, neither needing code:
#   - append %N to the array range (e.g. --array=0-119%40) to throttle CONCURRENT tasks without
#     changing the submitted count -- use this if the cap is on running rather than queued jobs
#   - the 8 launches are independent, so run this script once per event (edit trial_events) as waves
#
# --skip_existing True means a preempted batch resumes at the shuffle it died on rather than redoing
# the ones already written, which matters on ckpt-all. Re-submitting the same array is therefore the
# repair for anything preemption ate.
# ---------------------------------------------------------------------------------------------

partition="ckpt-all"
mem="32G"

# Measured on one session of SA/CIRCLE and scaled to all 22: ~1.8 min per feature-shuffle for
# StimOnset (16 windows) and ~3.4 min for FeedbackOnsetLong (29 windows), so a 10-shuffle batch is
# ~0.3 h and ~0.6 h. These requests leave ~5x headroom for the larger sessions.
time_limit_StimOnset="3:00:00"
time_limit_FeedbackOnsetLong="4:00:00"

subjects="SA BL"
trial_events="StimOnset FeedbackOnsetLong"
runs="S P"

num_shuffles=100
shuffles_per_job=10
# 12 features x (num_shuffles / shuffles_per_job) batches per launch
shuffle_array_max=$(( 12 * num_shuffles / shuffles_per_job - 1 ))

# Optional args passed through to run_anova.py
extra_args="$@"

submit_job_array () {
    local array_range="$1"
    local job_name="$2"
    local time_limit="$3"
    local python_args="$4"
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
    /src/wcst_decode/scripts/anova_analysis/run_anova.py $python_args $extra_args
EOT
}

for subject in $subjects; do
for trial_event in $trial_events; do
for run in $runs; do
    # the two runs of section 2.1. beh_filters is single-quoted where it is used, because
    # "High Not X" contains spaces -- passing it unquoted would word-split into three arguments
    if [ "$run" == "S" ]; then
        conditions="Choice"
        beh_filters='{"Response":"Correct","BeliefPartition":"High Not X"}'
        b_split_half=1
    else
        conditions="BeliefPartition"
        beh_filters='{"Response":"Correct","Choice":"Chose","BeliefConf":"High"}'
        b_split_half=2
    fi

    eval time_limit=\$time_limit_${trial_event}

    common_args="--subject $subject \
        --trial_event $trial_event \
        --conditions $conditions \
        --beh_filters '$beh_filters' \
        --b_split_half $b_split_half \
        --window_size 500 \
        --use_x True \
        --shuffle_method circular_shift"

    # 12 jobs: the true run, one per feature
    submit_job_array "0-11" "sb${run}${subject:0:1}${trial_event:0:4}" "$time_limit" \
        "$common_args --feat_idx \$SLURM_ARRAY_TASK_ID"

    # 120 jobs: one per (feature, batch of $shuffles_per_job shuffles).
    # feat = idx % 12, shuffle start = (idx / 12) * shuffles_per_job, so the starts are
    # {0, 10, ..., 90} and the batches tile 0-99 with no gaps and no overlap
    submit_job_array "0-$shuffle_array_max" "sh${run}${subject:0:1}${trial_event:0:4}" "$time_limit" \
        "$common_args \
         --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) \
         --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12 * $shuffles_per_job)) \
         --num_shuffles_per_job $shuffles_per_job \
         --skip_existing True"
done
done
done
