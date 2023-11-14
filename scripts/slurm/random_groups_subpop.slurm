#!/bin/bash
#SBATCH --job-name=subpop_random_groups

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=360

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231113_decode_features_random_splits.py \
    --shuffle_idx $SLURM_ARRAY_TASK_ID \ 
    --subpop_path /data/state_update_subpop.pickle --subpop_name state_update