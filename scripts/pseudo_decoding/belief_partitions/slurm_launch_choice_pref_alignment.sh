#!/bin/bash

# Alignment between the choice population vector (Chose X vs. Not Chose X) and the preference
# population vector (High X vs. High Not X), on the FULL population.
#
# The encoding counterpart to slurm_launch_pref_on_choice_axis.sh: mean differences over z-scored
# activity rather than decoder weights, so no models are fit and no decoder runs are read. See
# claude_notes/weight_vs_mean_difference_axes.md for why the two are different quantities.
#
# Writes /data/patrick_res/choice_pref_alignment/both_{event}/, shuffles in .../shuffles/, with the
# choice contrast's filters in the file name (see choice_beh_filters below), so runs against
# differently filtered choice vectors sit side by side in the same dir.
#
# Two structural differences from every other launcher here:
#
#   1. NO --feat_idx loop. With no subpopulation selection the unit set doesn't depend on the
#      feature, so one job loads each session's firing rates once and computes all 12 features
#      from them. Splitting by feature would re-read the same pickles 12 times.
#
#   2. NO region loop, and deliberately no --sig_unit_level / --region_level / --regions -- the
#      script raises if any of them is set. Each unit's coordinate is computed independently of
#      every other unit, so a region breakdown is a downstream groupby over the saved per-unit
#      vectors rather than a separate run.
#
# Parallelism is therefore over (event x shuffle) only: 2 x 11 = 22 jobs.
#
# Off the cluster, the same 22 cases run in one process with
#   python3 choice_pref_vector_alignment.py --subject both --run_all True \
#       --choice_beh_filters '{"Response": "Correct"}'
# which writes the identical files, and reads each session's firing rates once per event instead
# of once per (event, shuffle) -- the shuffles only permute behavior. --trial_events /
# --num_shuffles shrink that grid. The choice filters are NOT part of that grid, so a second
# filter setting is a second launch either way.

# Default values
partition="ckpt-all"
mem="8G"
# no SGD fits, no pseudo-population construction; the per-session array is ~13MB, and the cost is
# almost entirely reading one firing rate pickle per session
time_limit="60"

trial_events="StimOnset FeedbackOnsetLong"

# Trials the choice vector is built from, the encoding counterpart to the projection runs'
# --axis_beh_filters.
# '{}' reproduces the original all-trials vector, stored under mode choice_pref_align.
# '{"Response": "Correct"}' restricts it to correct trials -- the pool v_pref is itself drawn from
# -- and is stored under mode choice_pref_align_Response_Correct.
# Set here rather than passed in: extra_args="$@" drops the inner quoting and the json would be
# word split.
choice_beh_filters='{"Response": "Correct"}'

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
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/choice_pref_vector_alignment.py $python_args $extra_args
EOT
}

for trial_event in $trial_events; do
    common_args="--trial_event $trial_event \
        --subject both \
        --choice_beh_filters '$choice_beh_filters'"

    # 1 job: the true run
    submit_job_array "0-0" "a${trial_event:0:4}" \
        "$common_args"

    # 10 jobs: one per shuffle index
    submit_job_array "0-9" "sha${trial_event:0:4}" \
        "$common_args --shuffle_idx \$SLURM_ARRAY_TASK_ID"
done
