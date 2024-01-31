#!/bin/bash
#SBATCH --job-name=feat_max

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=180

feat_arr=("YELLOW" "POLKADOT" "SQUARE" "CYAN" "TRIANGLE" "ESCHER")


module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif \
    /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20240130_decode_feature_max_val_by_conf.py \
    --feature ${feat_arr[$SLURM_ARRAY_TASK_ID]}