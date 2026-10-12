#!/bin/bash
# Held-out feature dimension CCGP over the whole StimOnset and FeedbackOnsetLong intervals,
# dim_ccgp_heldout_feat_beliefs.py --full_interval, on High trials where the preferred feature was
# chosen and rewarded, with per-fold units from
#   dim_pair_anova.py --per_fold --full_interval --trial_event {event} --beh_filters {filters}
# Per event: 3 true runs (one per pair of dimensions) and 30 shuffle runs (3 pairs x 10
# session-permute shuffles), 66 jobs in all.

# Default values
partition="ckpt-all"

trial_events="StimOnset FeedbackOnsetLong"

# Optional args passed to decoding script
extra_args="$@"

common="--subject both --full_interval true --sig_unit_level dim_pair_fold_99th_Response_Correct_PreferredChosen_True_window_filter_drift --beh_filters '{\"Response\":\"Correct\",\"PreferredChosen\":true}' --base_output_path /data/patrick_res/dim_ccgp_heldout"

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
#SBATCH --mem=16G
#SBATCH --time=240

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/dim_ccgp_heldout_feat_beliefs.py $python_args $extra_args
EOT
}

for trial_event in $trial_events; do
    # First job array: 3 true runs, one per pair of dimensions
    submit_job_array "0-2" "dimho${trial_event}" \
        "$common --trial_event $trial_event --dim_pair_idx \$SLURM_ARRAY_TASK_ID"

    # Second job array: 30 shuffle runs, 3 pairs x 10 shuffles
    submit_job_array "0-29" "shdimho${trial_event}" \
        "$common --trial_event $trial_event --dim_pair_idx \$((\$SLURM_ARRAY_TASK_ID % 3)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 3))"
done
