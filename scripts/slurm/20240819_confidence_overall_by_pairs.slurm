#!/bin/bash
#SBATCH --job-name=conf_overall

#SBATCH -p gpu-a100
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
    /src/wcst_decode/scripts/pseudo_decoding/20240819_confidence_overall_by_pairs.py \
    --pair_idx $SLURM_ARRAY_TASK_ID 