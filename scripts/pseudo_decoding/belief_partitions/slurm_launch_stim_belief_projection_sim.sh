#!/bin/bash

# Stim/belief projection simulation, stim_belief_projection_sim.py. See
# claude_notes/Stim Belief projection simulation proposal.md.
#
# Needs, in /data/patrick_res/stim_belief_projection_sim/, both run locally in under a minute:
#   layout.pickle     python3 scripts/pseudo_decoding/belief_partitions/stim_belief_sim_layout.py
#   calibration.json  python3 scripts/pseudo_decoding/belief_partitions/stim_belief_projection_sim.py --mode calibrate
#
# Every cos(theta), both split variants, one draw per job: 7 populations x 2 splits x 12 features x
# num_sim_repeats = 840 jobs at 5 repeats, under the 2000 job cap. Shuffles are batched inside each
# job rather than given their own, so the grid never has to shrink to fit.
#
# Whole population jobs are the slow ones: ~8s per CPU fit of the real decoder, most of it torch's
# dropout mask, so ~20 min per job. Regions are a few minutes.

partition="ckpt-all"
mem="8G"
time_limit="120"

num_sim_repeats=5
# the null the projection is tested against, as in the paper
sim_shuffles=10

splits="b_split no_split"
populations="whole_pop amygdala_Amy basal_ganglia_BG inferior_temporal_cortex_ITC medial_pallium_MPal lateral_prefrontal_cortex_lat_PFC anterior_cingulate_gyrus_ACgG"

# Optional args passed to the simulation script
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
#SBATCH --mem=$mem
#SBATCH --time=$time_limit

module load singularity
singularity exec --writable-tmpfs --nv \
    --bind /gscratch/walkerlab/patrick:/data,/mmfs1/home/pqz317/wcst_decode:/src/wcst_decode \
    /gscratch/walkerlab/patrick/singularity/wcst_decode_image.sif /usr/bin/python3 \
    /src/wcst_decode/scripts/pseudo_decoding/belief_partitions/stim_belief_projection_sim.py $python_args $extra_args
EOT
}

for population in $populations; do
    # short tag for the slurm job name, e.g. inferior_temporal_cortex_ITC -> ITC
    if [ "$population" == "whole_pop" ]; then
        tag="all"
    else
        tag="${population##*_}"
    fi

    # per split, 12 features x num_sim_repeats
    for split in $splits; do
        submit_job_array "0-$((12 * num_sim_repeats - 1))" "sbsim${tag}${split:0:1}" \
            "--mode simulate --population $population --split $split --num_shuffles $sim_shuffles \
             --feat_idx \$((\$SLURM_ARRAY_TASK_ID % 12)) --repeat \$((\$SLURM_ARRAY_TASK_ID / 12))"
    done
done
