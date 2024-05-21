#!/bin/bash
#SBATCH --job-name=max_feat_per

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180

norm_arr=("--norm_type block_zscore_fr" "--norm_type block_mean_sub" "")


module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20240521_decode_max_feat_pair.py GREEN MAGENTA \
    --should_shuffle --shuffle_seed $SLURM_ARRAY_TASK_ID