#!/bin/bash
#SBATCH --job-name=high_val_sub

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180

cond_arr=("MaxFeat" "RandomMaxFeat" "MaxFeat" "RandomMaxFeat")
trial_arr=("high_conf" "high_conf" "low_conf" "low_conf")


module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231113_decode_high_value_feature.py \
    --condition ${cond_arr[$SLURM_ARRAY_TASK_ID]} \
    --subtrials_path /data/${trial_arr[$SLURM_ARRAY_TASK_ID]}idence_trials.pickle \
    --subtrials_name ${trial_arr[$SLURM_ARRAY_TASK_ID]}