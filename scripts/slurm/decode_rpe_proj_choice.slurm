#!/bin/bash
#SBATCH --job-name=rpe_proj_choice

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/code/wcst_decode /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 /code/wcst_decode/scripts/pseudo_decoding/20231005_decode_rpe_groups.py --proj_path /data/orth_axes_choice.npy --proj_name orth_choice
