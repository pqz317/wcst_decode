#!/bin/bash
#SBATCH --job-name=feat_sub_cor

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=180

array=("pfc" "hip")

module load singularity
singularity exec --writable-tmpfs --nv --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231107_cross_decode_pseudo.py \
    --subpop_path /data/${array[$SLURM_ARRAY_TASK_ID]}_subpop.pickle \
    --subpop_name ${array[$SLURM_ARRAY_TASK_ID]} \
    --subtrials_path /data/cor_bal_trials.pickle --subtrials_name cor_bal