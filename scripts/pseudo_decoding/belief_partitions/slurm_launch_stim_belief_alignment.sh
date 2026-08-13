#!/bin/bash

# Alignment between the stimulus population vector (r_B1 - r_A) and the belief population vector
# (r_C - r_B2), on the FULL population, over the three-group A/B/C design of
# claude_notes/stim_belief_alignment_updated.md.
#
# Replaces slurm_launch_choice_pref_alignment.sh: there the two contrasts are independently
# filtered and z-scored, here a single pool of correct trials is split three ways and both vectors
# share one metric. Group B is halved so it isn't shared between the two vectors -- see the script
# docstring, and Issue 1 of that note, for why that is mandatory rather than a refinement.
#
# Writes /data/patrick_res/stim_belief_alignment/both_{event}_Response_Correct/, shuffles in
# .../shuffles/. The pool filter is part of the design, not a flag, so unlike the choice_pref
# launcher there is no --choice_beh_filters and only one variant per event.
#
# Two structural differences from every other launcher here, both inherited:
#
#   1. NO --feat_idx loop. With no subpopulation selection the unit set doesn't depend on the
#      feature, so one job loads each session's firing rates once and computes all 12 features
#      from them. Splitting by feature would re-read the same pickles 12 times.
#
#   2. NO region loop, and deliberately no --sig_unit_level / --region_level / --regions -- the
#      script raises if any of them is set. Each unit's coordinate is computed independently of
#      every other unit, so a region breakdown is a downstream groupby over the saved per-unit
#      vectors rather than a separate run. So is the deattenuated cosine, since the per-group
#      variances and trial counts are saved alongside the means.
#
# Parallelism is therefore over (event x shuffle) only: 2 x 11 = 22 jobs.
#
# Off the cluster, the same 22 cases run in one process with
#   python3 stim_belief_vector_alignment.py --subject both --run_all True
# which writes the identical files, and reads each session's firing rates once per event instead
# of once per (event, shuffle) -- the shuffles only permute behavior. --trial_events /
# --num_shuffles shrink that grid.

# Default values
partition="ckpt-all"
mem="8G"
# no SGD fits, no pseudo-population construction; the per-session array is ~13MB, and the cost is
# almost entirely reading one firing rate pickle per session
time_limit="60"

trial_events="StimOnset FeedbackOnsetLong"

# Optional args passed to the alignment script
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
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/stim_belief_vector_alignment.py $python_args $extra_args
EOT
}

for trial_event in $trial_events; do
    common_args="--trial_event $trial_event \
        --subject both"

    # 1 job: the true run
    submit_job_array "0-0" "sb${trial_event:0:4}" \
        "$common_args"

    # 10 jobs: one per shuffle index
    submit_job_array "0-9" "shsb${trial_event:0:4}" \
        "$common_args --shuffle_idx \$SLURM_ARRAY_TASK_ID"
done
