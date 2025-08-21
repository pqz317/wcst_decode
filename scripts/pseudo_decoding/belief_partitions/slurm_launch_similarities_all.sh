#!/bin/bash

# Default values
partition="ckpt-all"

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
#SBATCH --mem=2G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/similarity_of_feat_beliefs.py $python_args $extra_args
EOT
}

# First job array: 28 jobs
submit_job_array "0-27" "sim" \
    "--trial_event StimOnset --pair_idx \$SLURM_ARRAY_TASK_ID"

# Second job array: 280 jobs with shuffle indices
submit_job_array "0-279" "shsim" \
    "--trial_event StimOnset --pair_idx \$((\$SLURM_ARRAY_TASK_ID % 28)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 28))"
