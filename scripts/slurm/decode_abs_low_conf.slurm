#!/bin/bash
#SBATCH --job-name=abs_low_conf

#SBATCH -p gpu-a100
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv --bind /gscratch/walkerlab/patrick:/data /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 /src/wcst_decode/scripts/pseudo_decoding/20231103_decode_features_subpop_abstract.py --abstract --subtrials_path /data/low_confidence_trials.pickle --subtrials_name low_conf
