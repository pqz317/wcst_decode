#!/bin/bash
# Fixed-tuple dimension CCGP over the whole StimOnset and FeedbackOnsetLong intervals,
# dim_ccgp_tuple_feat_beliefs.py, on High trials where the preferred feature was chosen and rewarded,
# with the units preference-selective for any of the tuple's features, from run_anova2.py on the
# tuples' sessions (generate_dim_tuples.py) and 20250606_generate_significant_units.py --level_tag dim_tuples.
# Per event: 8 true runs (one per tuple) and 80 shuffle runs (8 tuples x 10 session-permute
# shuffles), 176 jobs in all. Each job runs both the dimension and the random folds. One split of the
# largest tuple on FeedbackOnsetLong took 16 min locally (multithreaded), ~2 h for 8 splits; the limit
# leaves room for 1 CPU.

# Default values
partition="ckpt-all"

trial_events="StimOnset FeedbackOnsetLong"

# Optional args passed to decoding script
extra_args="$@"

common="--subject both --sig_unit_level pref_99th_dim_tuples_window_filter_drift --beh_filters '{\"Response\":\"Correct\",\"PreferredChosen\":true}' --base_output_path /data/patrick_res/dim_ccgp_tuples"

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
#SBATCH --time=600

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/dim_ccgp_tuple_feat_beliefs.py $python_args $extra_args
EOT
}

for trial_event in $trial_events; do
    # First job array: 8 true runs, one per tuple
    submit_job_array "0-7" "dimtup${trial_event}" \
        "$common --trial_event $trial_event --tuple_idx \$SLURM_ARRAY_TASK_ID"

    # Second job array: 80 shuffle runs, 8 tuples x 10 shuffles
    submit_job_array "0-79" "shdimtup${trial_event}" \
        "$common --trial_event $trial_event --tuple_idx \$((\$SLURM_ARRAY_TASK_ID % 8)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 8))"
done
