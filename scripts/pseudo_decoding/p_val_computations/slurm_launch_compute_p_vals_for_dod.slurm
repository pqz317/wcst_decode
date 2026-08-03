#!/bin/bash
# launch with sbatch --array=0-13 wcst_decode/scripts/pseudo_decoding/p_val_computations/slurm_launch_compute_p_vals_for_dod.slurm

#SBATCH --job-name=dod_p_vals
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=180


module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/p_val_computations/compute_p_vals_for_dod.py --combo_id $SLURM_ARRAY_TASK_ID
