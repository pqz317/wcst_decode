#!/bin/bash
sbatch --array=0-16 <<EOT
#!/bin/bash
#SBATCH --job-name=pref
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
    /src/wcst_decode/scripts/pseudo_decoding/20241113_decode_preferred_beliefs_by_pairs.py \
    --pair_idx \$SLURM_ARRAY_TASK_ID $1 $2 $3 $4 $5 $6 $7 $8 $9
EOT

sbatch --array=0-16 <<EOT
#!/bin/bash
#SBATCH --job-name=npref
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
    /src/wcst_decode/scripts/pseudo_decoding/20241113_decode_preferred_beliefs_by_pairs.py \
    --chosen_not_preferred True \
    --pair_idx \$SLURM_ARRAY_TASK_ID $1 $2 $3 $4 $5 $6 $7 $8 $9
EOT

sbatch --array=0-169<<EOT
#!/bin/bash
#SBATCH --job-name=prefshuf

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
    /src/wcst_decode/scripts/pseudo_decoding/20241113_decode_preferred_beliefs_by_pairs.py \
    --pair_idx \$((\$SLURM_ARRAY_TASK_ID % 17)) \
    --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 17)) \
    $1 $2 $3 $4 $5 $6 $7 $8 $9
EOT

sbatch --array=0-169<<EOT
#!/bin/bash
#SBATCH --job-name=nprefshuf

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
    /src/wcst_decode/scripts/pseudo_decoding/20241113_decode_preferred_beliefs_by_pairs.py \
    --chosen_not_preferred True \
    --pair_idx \$((\$SLURM_ARRAY_TASK_ID % 17)) \
    --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 17)) \
    $1 $2 $3 $4 $5 $6 $7 $8 $9
EOT