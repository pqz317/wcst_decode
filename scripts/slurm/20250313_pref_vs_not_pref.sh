#!/bin/bash
sbatch --array=0-11 <<EOT
#!/bin/bash
#SBATCH --job-name=pvsnp
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/20250311_decode_single_selected_features.py \
    --condition pref_vs_not_pref \
    --feat_idx \$SLURM_ARRAY_TASK_ID $1 $2 $3 $4 $5 $6 $7 $8 $9
EOT

sbatch --array=0-119 <<EOT
#!/bin/bash
#SBATCH --job-name=pvsnp_shuf

#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/20250311_decode_single_selected_features.py \
    --condition pref_vs_not_pref \
    --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) \
    --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12)) \
    $1 $2 $3 $4 $5 $6 $7 $8 $9
EOT