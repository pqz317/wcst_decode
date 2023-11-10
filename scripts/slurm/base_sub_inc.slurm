#!/bin/bash
#SBATCH --job-name=base_sub_cor

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231103_decode_features_subpop_abstract.py \
    --subtrials_path /data/inc_trials.pickle --subtrials_name inc \
    --subpop_path /data/state_update_subpop.pickle --subpop_name state_update