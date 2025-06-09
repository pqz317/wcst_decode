#!/bin/bash

# Default values
partition="ckpt-all"

while getopts "p:" opt; do
  case $opt in
    p) partition="$OPTARG" ;; # if -p, set to partition
    --) break ;; ## stop parsing after encountering --
  esac
done
shift $((OPTIND-1))


trial_events="StimOnset FeedbackOnsetLong"
modes="pref conf"

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
            "--mode $mode --trial_event $trial_event --feat_idx \$SLURM_ARRAY_TASK_ID"

        # Second job array: 120 jobs with shuffle indices
        submit_job_array "0-119" "sh${trial_event}${mode}" \
            "--mode $mode --trial_event $trial_event --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12))"
    done
done