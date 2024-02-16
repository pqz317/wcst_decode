#!/bin/bash
#SBATCH --job-name=others

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180

cond_arr=("Correct" "Incorrect" "all")
res_arr=("--use_residual_fr" "")
feat_arr=("CYAN,GREEN,MAGENTA,YELLOW" "ESCHER,POLKADOT,RIPPLE,SWIRL")
feat_dim_arr=("Color" "Pattern")

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20240214_decode_feat_by_max.py \
    --feature_list ${feat_arr[$((($SLURM_ARRAY_TASK_ID / 6) % 2))]} \
    --feature_dim ${feat_dim_arr[$((($SLURM_ARRAY_TASK_ID / 6) % 2))]} \
    --condition ${cond_arr[$(($SLURM_ARRAY_TASK_ID % 3))]} \
    ${res_arr[$((($SLURM_ARRAY_TASK_ID / 3) % 2))]}