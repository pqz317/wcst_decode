#!/bin/bash
#SBATCH --job-name=stim_on

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=180

dim_arr=("Color" "Shape" "Pattern")
condition_arr=("attended" "not_attended")

module load singularity
singularity exec --writable-tmpfs \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/20240612_decode_feature_by_attention_stim_onset.py \
    --feature_dim ${dim_arr[$(($SLURM_ARRAY_TASK_ID % 3))]} \
    --condition ${condition_arr[$(($SLURM_ARRAY_TASK_ID / 3))]}