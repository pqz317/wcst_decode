#!/bin/bash
#SBATCH --job-name=value_glm

#SBATCH -p ckpt
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=180

fb_arr=("Response" "RPEGroup")
res_arr=("--use_residual_fr" "")


module load singularity
singularity exec --writable-tmpfs \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/glm_analysis/20240402_fit_value_glm.py $(($SLURM_ARRAY_TASK_ID / 4)) \
    --feedback_type ${fb_arr[$(($SLURM_ARRAY_TASK_ID % 2))]} \
    ${res_arr[$((($SLURM_ARRAY_TASK_ID / 2) % 2))]}
