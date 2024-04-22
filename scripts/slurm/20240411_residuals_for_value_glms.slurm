#!/bin/bash
#SBATCH --job-name=res_fr

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=180

fb_arr=("Response" "RPEGroup")


module load singularity
singularity exec --writable-tmpfs \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/glm_analysis/20240212_generate_residual_frs.py $(($SLURM_ARRAY_TASK_ID / 2)) \
    --feedback_type ${fb_arr[$(($SLURM_ARRAY_TASK_ID % 2))]} \
    --include_interactions \
    --use_prev_trial_params