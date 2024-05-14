#!/bin/bash
#SBATCH --job-name=maxfeat_glm

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=180

model_arr=("Linear" "Poisson")


module load singularity
singularity exec --writable-tmpfs \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/glm_analysis/20240510_fit_maxfeat_glm.py $SLURM_ARRAY_TASK_ID  \
    --model Linear --norm_mode mean_sub