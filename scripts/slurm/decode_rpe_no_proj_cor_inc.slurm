#!/bin/bash
#SBATCH --job-name=rpe_proj_fb

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=180

array=("inc_bal" "cor_bal")

cat ${array[$SLURM_ARRAY_TASK_ID]}

module load singularity
singularity exec --writable-tmpfs --nv --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231005_decode_rpe_groups.py \
    --subtrials_path /data/${array[$SLURM_ARRAY_TASK_ID]}_trials.pickle --subtrials_name ${array[$SLURM_ARRAY_TASK_ID]}