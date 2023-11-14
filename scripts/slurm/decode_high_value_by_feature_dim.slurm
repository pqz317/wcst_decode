#!/bin/bash
#SBATCH --job-name=high_val_feat_dim

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180

array=("MaxFeat" "RandomMaxFeat")

cat ${array[$SLURM_ARRAY_TASK_ID]}

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231113_decode_feature_dim_by_high_value_feature.py \
    --condition ${array[$SLURM_ARRAY_TASK_ID]}