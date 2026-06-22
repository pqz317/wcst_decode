#!/bin/bash

# Default values
partition="ckpt-all"

trial_events="StimOnset FeedbackOnsetLong"
mode="choice"
subpop="choice_99th_window_filter_drift"

# (train_partition, test_partition) pairs to evaluate
declare -a train_partitions=("In X Dim"    "Not in X Dim" "Low"       "In X Dim" "Low"          "Not in X Dim")
declare -a test_partitions=( "Not in X Dim" "In X Dim"    "In X Dim"  "Low"      "Not in X Dim" "Low")

# Optional args passed through to the decoding script
extra_args="$@"

submit_job_array () {
    local array_range=$1
    local job_name=$2
    local python_args=$3
    sbatch --array="$array_range" <<EOT;
#!/bin/bash
#SBATCH --job-name="$job_name"
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
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/decode_choice_belief_dim_cross_partition.py $python_args $extra_args
EOT
}

for i in "${!train_partitions[@]}"; do
    train_part="${train_partitions[$i]}"
    test_part="${test_partitions[$i]}"
    for trial_event in $trial_events; do
        pair_tag="$(echo "${train_part}_to_${test_part}" | tr ' ' '_')"

        # Real runs: 12 jobs (one per feature)
        submit_job_array "0-11" "${trial_event}_${pair_tag}" \
            "--mode $mode --trial_event $trial_event --sig_unit_level $subpop \
            --feat_idx \$SLURM_ARRAY_TASK_ID --balance_by_filters True \
            --train_partition '$train_part' --test_partition '$test_part' \
            --base_model_path /data/patrick_res/choice_belief_dim \
            --base_output_path /data/patrick_res/choice_belief_dim_cross_partition"

        # Shuffle runs: 120 jobs (10 shuffles × 12 features)
        # Loads shuffle_N models from train partition; evaluates on real test data.
        submit_job_array "0-119" "sh_${trial_event}_${pair_tag}" \
            "--mode $mode --trial_event $trial_event --sig_unit_level $subpop \
            --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) \
            --shuffle_idx \$((\$SLURM_ARRAY_TASK_ID / 12)) \
            --balance_by_filters True \
            --train_partition '$train_part' --test_partition '$test_part' \
            --base_model_path /data/patrick_res/choice_belief_dim \
            --base_output_path /data/patrick_res/choice_belief_dim_cross_partition"
    done
done
