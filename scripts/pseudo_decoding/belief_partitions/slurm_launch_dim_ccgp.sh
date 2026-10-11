#!/bin/bash
# Dimension CCGP, dim_ccgp_feat_beliefs.py: 92 true runs (one per quadruple of
# generate_dim_quads.py) and 920 shuffle runs (92 quadruples x 10 session-permute shuffles),
# 1012 jobs in all.

# Default values
partition="ckpt-all"

# Optional args passed to decoding script
extra_args="$@"

common="--subject both --trial_event StimOnset --sig_unit_level pref_99th_no_cond_window_filter_drift --base_output_path /data/patrick_res/dim_ccgp"

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
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/dim_ccgp_feat_beliefs.py $python_args $extra_args
EOT
}

# First job array: 92 true runs, one per quadruple
submit_job_array "0-91" "dimccgp" \
    "$common --quad_idx \$SLURM_ARRAY_TASK_ID"

# Second job array: 920 shuffle runs, 92 quadruples x 10 shuffles
submit_job_array "0-919" "shdimccgp" \
    "$common --quad_idx \$((\$SLURM_ARRAY_TASK_ID % 92)) --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 92))"
