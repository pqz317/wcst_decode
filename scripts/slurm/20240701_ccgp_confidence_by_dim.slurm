#!/bin/bash
#SBATCH --job-name=ccgp_conf

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=180

condition_arr=("" "--use_next_trial_conf")

module load singularity
singularity exec --writable-tmpfs \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/20240701_ccgp_confidence_by_dim.py \
    --seed_idx $(($SLURM_ARRAY_TASK_ID / 2)) $(($SLURM_ARRAY_TASK_ID % 2))