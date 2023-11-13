#!/bin/bash
#SBATCH --job-name=card_interaction_glm

#SBATCH -p ckpt
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --bind /gscratch/walkerlab/patrick:/data /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 /src/wcst_decode/scripts/glm_analysis/hyak_fit_card_shuffle_glm.py $SLURM_ARRAY_TASK_ID
