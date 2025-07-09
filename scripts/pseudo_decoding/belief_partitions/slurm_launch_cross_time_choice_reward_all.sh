#!/bin/bash
trial_events="StimOnset FeedbackOnsetLong"
modes="reward choice"

# Optional args passed to decoding script
extra_args="$@"

# Function to submit a job array
submit_job_array () {
    local array_range=$1
    local job_name=$2
    local python_args=$3
    sbatch --array="$array_range" <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
        /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/decode_belief_partitions_cross_time.py $python_args $extra_args
EOT
}

# Loop over trial events and modes
for trial_event in $trial_events; do
    for mode in $modes; do

        # Get the other trial event:
        if [[ "$trial_event" == "StimOnset" ]]; then
            other_trial_event="FeedbackOnsetLong"
        else
            other_trial_event="StimOnset"
        fi

        # First job array: 12 jobs
        submit_job_array "0-11" "cross_${trial_event}${mode}" \
            "--mode $mode --trial_event $trial_event --feat_idx \$SLURM_ARRAY_TASK_ID --balance_cols Choice,Response --base_output_path /data/patrick_res/choice_reward"

        # Second job array: 12 jobs with other trial event models
        submit_job_array "0-11" "cross_${trial_event}${mode}_${other_trial_event}model" \
            "--mode $mode --trial_event $trial_event --model_trial_event $other_trial_event --feat_idx \$SLURM_ARRAY_TASK_ID --balance_cols Choice,Response --base_output_path /data/patrick_res/choice_reward"
    done
done