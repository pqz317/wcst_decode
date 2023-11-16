#!/bin/bash
#SBATCH --job-name=abs_sub

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=180

array=("hip" "pfc")

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231103_decode_features_subpop_abstract.py \
    --abstract --abs_cond RPEGroup \
    --subpop_path /data/${array[$SLURM_ARRAY_TASK_ID]}_subpop.pickle --subpop_name ${array[$SLURM_ARRAY_TASK_ID]}

