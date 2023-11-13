#!/bin/bash
#SBATCH --job-name=patrick_test

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1M
#SBATCH --time=60

module load singularity
singularity exec --writable-tmpfs --bind /gscratch/walkerlab/patrick:/data /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 /src/wcst_decode/scripts/glm_analysis/hyak_test.py
