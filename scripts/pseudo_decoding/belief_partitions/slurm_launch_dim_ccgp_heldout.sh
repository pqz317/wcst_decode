#!/bin/bash
# Held-out feature dimension CCGP, dim_ccgp_heldout_feat_beliefs.py: 3 true runs (one per pair of
# dimensions of generate_dim_heldout_sessions.py) and 30 shuffle runs (3 pairs x 10
# session-permute shuffles), 33 jobs in all. All units, so no --sig_unit_level.

# Default values
partition="ckpt-all"

# Optional args passed to decoding script
extra_args="$@"

common="--subject both --trial_event StimOnset --base_output_path /data/patrick_res/dim_ccgp_heldout"

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
#SBATCH --mem=32G
#SBATCH --time=360

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/dim_ccgp_heldout_feat_beliefs.py $python_args $extra_args
EOT
}

# First job array: 3 true runs, one per pair of dimensions
submit_job_array "0-2" "dimho" \
    "$common --dim_pair_idx \$SLURM_ARRAY_TASK_ID"

# Second job array: 30 shuffle runs, 3 pairs x 10 shuffles
submit_job_array "0-29" "shdimho" \
    "$common --dim_pair_idx \$((\$SLURM_ARRAY_TASK_ID % 3)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 3))"
