#!/bin/bash
#SBATCH --job-name=max_feat_by_pairs

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=180


module load singularity
singularity exec --writable-tmpfs \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/20240725_high_conf_max_feat_by_pairs.py \
    --pair_idx $(($SLURM_ARRAY_TASK_ID % 18)) --shuffle_idx $(($SLURM_ARRAY_TASK_ID / 18))
