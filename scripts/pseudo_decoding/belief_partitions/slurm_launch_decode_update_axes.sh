#!/bin/bash

# Pref and conf decoders for decoder_update_projections.py, trained on the axis trials of the pair
# split only (decode_update_axes.py). Same units and resources as slurm_launch_decode_no_cond.sh.
# No shuffles: decoder_update_projections.py tests by sign flips instead.
#
# 2 modes x 12 features = 24 jobs. When all have finished, run locally:
#   python -m scripts.pseudo_decoding.belief_partitions.decoder_update_projections

partition="ckpt-all"
modes="pref conf"

declare -A mode_to_subpop
mode_to_subpop["pref"]="pref_99th_no_cond_window_filter_drift"
mode_to_subpop["conf"]="conf_99th_no_cond_window_filter_drift"

# Optional args passed to decoding script
extra_args="$@"

submit_job_array () {
    local array_range=$1
    local job_name=$2
    local python_args=$3
    sbatch --array="$array_range" <<EOT;
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH -p $partition
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=180

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/decode_update_axes.py $python_args $extra_args
EOT
}

for mode in $modes; do
    submit_job_array "0-11" "ua${mode}" \
        "--mode $mode --trial_event StimOnset --sig_unit_level ${mode_to_subpop[$mode]} --feat_idx \$SLURM_ARRAY_TASK_ID"
done
