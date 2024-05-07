#!/bin/bash
#SBATCH --job-name=value_glm

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
    /src/wcst_decode/scripts/glm_analysis/20240501_fit_value_glm_agg.py $(($SLURM_ARRAY_TASK_ID / 2)) \
    --model ${model_arr[$(($SLURM_ARRAY_TASK_ID % 2))]} --block_zscore_fr