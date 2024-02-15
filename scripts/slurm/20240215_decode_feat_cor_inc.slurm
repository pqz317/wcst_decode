#!/bin/bash
#SBATCH --job-name=high_val_sub

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180

cond_arr=("Correct" "Incorrect")
res_arr=("--use_residual_fr" "")
shuffle_arr=("--should_shuffle" "")


module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20240214_decode_feat_by_max.py \
    --condition ${cond_arr[$(($SLURM_ARRAY_TASK_ID % 2))]} \
    ${res_arr[$((($SLURM_ARRAY_TASK_ID / 2) % 2))]} ${shuffle_arr[$((($SLURM_ARRAY_TASK_ID / 4) % 2))]}