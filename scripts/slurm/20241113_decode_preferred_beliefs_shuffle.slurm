#!/bin/bash
#SBATCH --job-name=pref_shuf

#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/20241113_decode_preferred_beliefs.py \
    --pair_idx $(($SLURM_ARRAY_TASK_ID % 18)) \
    --shuffle_idx $(($SLURM_ARRAY_TASK_ID / 18)) \
    $not_pref_flag

    