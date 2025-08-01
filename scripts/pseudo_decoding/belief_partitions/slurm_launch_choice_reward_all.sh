#!/bin/bash

# Default values
partition="ckpt-all"

trial_events="StimOnset FeedbackOnsetLong"
modes="reward choice"

declare -A mode_to_subpop
mode_to_subpop["reward"]="response_99th_window_filter_drift"
mode_to_subpop["choice"]="choice_99th_window_filter_drift"

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
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/decode_belief_partitions.py $python_args $extra_args
EOT
}

# Loop over trial events and modes
for trial_event in $trial_events; do
    for mode in $modes; do
        # First job array: 12 jobs
        submit_job_array "0-11" "${trial_event}${mode}" \
            "--mode $mode --trial_event $trial_event --sig_unit_level ${mode_to_subpop[$mode]} --feat_idx \$SLURM_ARRAY_TASK_ID --balance_cols Choice,Response --base_output_path /data/patrick_res/choice_reward"

        # Second job array: 120 jobs with shuffle indices
        submit_job_array "0-119" "sh${trial_event}${mode}" \
            "--mode $mode --trial_event $trial_event --sig_unit_level ${mode_to_subpop[$mode]} --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12)) --balance_cols Choice,Response --base_output_path /data/patrick_res/choice_reward"
    done
done