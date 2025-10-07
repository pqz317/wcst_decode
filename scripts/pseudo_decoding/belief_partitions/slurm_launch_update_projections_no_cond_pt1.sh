#!/bin/bash

# Default values
# partition="ckpt-all"
partition="gpu-a100"

trial_event="StimOnset"
modes="pref conf"
conditions=(\
    '{\"Response\":\"Correct\"\,\"Choice\":\"Chose\"}' \
    '{\"Response\":\"Incorrect\"\,\"Choice\":\"Chose\"}' \
    '{\"Response\":\"Correct\"}' \
    '{\"Response\":\"Incorrect\"}' \
    '{\"Response\":\"Correct\"\,\"Choice\":\"Chose\"\,\"BeliefPartition\":\"Low\"}' \
    # '{\"Response\":\"Correct\"\,\"Choice\":\"Chose\"\,\"BeliefPartition\":\"High\ X\"}' \
    # '{\"Response\":\"Correct\"\,\"Choice\":\"Chose\"\,\"BeliefPartition\":\"High\ Not\ X\"}' \
    # '{\"Response\":\"Incorrect\"\,\"Choice\":\"Chose\"\,\"BeliefPartition\":\"Low\"}' \
    # '{\"Response\":\"Incorrect\"\,\"Choice\":\"Chose\"\,\"BeliefPartition\":\"High\ X\"}' \
    # '{\"Response\":\"Incorrect\"\,\"Choice\":\"Chose\"\,\"BeliefPartition\":\"High\ Not\ X\"}' \
)


declare -A mode_to_subpop
mode_to_subpop["pref"]="pref_99th_no_cond_window_filter_drift"
mode_to_subpop["conf"]="conf_99th_no_cond_window_filter_drift"

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
#SBATCH --mem=16G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/pref_conf_projection_updates.py $python_args $extra_args
EOT
}

# Loop over trial events and modes
for mode in $modes; do
    for condition in "${conditions[@]}"; do
        # First job array: 12 jobs
        submit_job_array "0-11" "${trial_event}${mode}" \
            "--mode $mode --trial_event $trial_event --sig_unit_level ${mode_to_subpop[$mode]} --conditions $condition --feat_idx \$SLURM_ARRAY_TASK_ID"

        # Second job array: 120 jobs with shuffle indices
        submit_job_array "0-119" "sh${trial_event}${mode}" \
            "--mode $mode --trial_event $trial_event --sig_unit_level ${mode_to_subpop[$mode]} --conditions $condition --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12))"
    done
done
