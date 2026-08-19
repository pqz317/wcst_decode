#!/bin/bash

# Crosses the two per-unit ANOVA runs that 20260819_run_stim_belief_unit_anova.sh produced, and
# tests H1 (disjoint stimulus and belief codes) against H3 (mixed selectivity with random sign).
# See the script docstring and claude_notes/stim_belief_single_unit_anova_lite.md part 4.
#
# Run this only after the grid is complete: 12 true + 1200 shuffle pickles in each of the 8 run
# directories. The script raises on a missing file rather than quietly building the null on fewer
# shuffles, so an incomplete grid fails fast.
#
# One job per trial event, and that is the whole parallelism available:
#
#   - NO --feat_idx loop. p-values are accumulated one feature at a time into a single (unit, feat,
#     window) item table, and the statistics are computed over all 12 features together, so a
#     per-feature job would have nothing to combine at the end.
#   - NO region loop. Every region is a mask over the same arrays -- one pass reports the whole
#     population and all 6 regions of interest.
#   - Both subjects run inside one job, since the headline statistics pool them.
#
# The cost is almost entirely reading pickles: ~2400 files / ~2.3 GB per (subject, event). Memory is
# flat in the number of features, because only the uint16 exceedance counts survive a feature.
#
# Off the cluster the same two cases run in one process with
#   python3 stim_belief_unit_overlap.py --run_all True
# which writes the identical files.

partition="ckpt-all"
mem="16G"
time_limit="120"

trial_events="StimOnset FeedbackOnsetLong"

# Optional args passed through (e.g. --alpha, --num_shuffles, --loo_matched)
extra_args="$@"

for trial_event in $trial_events; do
sbatch <<EOT;
#!/bin/bash
#SBATCH --job-name=ovl${trial_event:0:4}
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
    /src/wcst_decode/scripts/anova_analysis/stim_belief_unit_overlap.py \
    --trial_event $trial_event $extra_args
EOT
done
