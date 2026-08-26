"""
The A/B/C groups of claude_notes/stim_belief_alignment_updated.md, and group B's half-split.

Relative to a feature X, over correct trials:

    A: X not chosen, X not preferred     (BeliefPartition == "High Not X", Choice == "Not Chose")
    B: X chosen,     X not preferred     (BeliefPartition == "High Not X", Choice == "Chose")
    C: X chosen,     X preferred         (BeliefPartition == "High X",     Choice == "Chose")

Group B sits in both the stimulus contrast (A vs. B) and the belief contrast (B vs. C), so any pair
of analyses that spans the two shares B's sampling noise unless B is split. draw_b_split is the one
canonical assignment: it is used by the population cosine analysis
(stim_belief_vector_alignment.py, Issue 1(a)), the single-unit anova (scripts/anova_analysis/
run_anova.py), and the decoding runs of Steps 3-5 (decode_belief_partitions.py with
--b_split_half, Issue 2).

This module exists so all three can share those definitions: stim_belief_vector_alignment.py
imports decode_belief_partitions, so decode_belief_partitions cannot import back from it without a
cycle. Nothing here imports anything from this package -- keep it that way.
"""

import numpy as np

from constants.behavioral_constants import *

# the single pool all three groups are drawn from. Not configurable: "control for external
# confounds by examining only correct trials" is part of the design, not a variant of it. It is
# still written into the output directory name, so a future variant would sit beside this one
POOL_FILTERS = {"Response": "Correct"}

# the three cells before any halving. cos_raw halves only B; cos_cv halves all three, so this is
# what the half-split and the min-trial guard are expressed over
BASE_GROUPS = ["A", "B", "C"]

# a seed field, so the three groups' half-assignments are independent draws rather than the same
# permutation applied three times
GROUP_CODE = {"A": 1, "B": 2, "C": 3}


def group_masks(beh):
    """
    The three (Choice x BeliefPartition) cells, as boolean masks over an already-filtered beh.

    Same definitions as claude_notes/stim_belief_group_counts.py, which is where the trial counts
    in the note's Step 1 table come from -- kept in sync by hand, since claude_notes isn't a package.
    """
    return {
        "A": (beh.BeliefPartition == "High Not X") & (beh.Choice == "Not Chose"),
        "B": (beh.BeliefPartition == "High Not X") & (beh.Choice == "Chose"),
        "C": (beh.BeliefPartition == "High X")     & (beh.Choice == "Chose"),
    }


def draw_half_split(session, feat, trials, seed, shuffle_idx, group="B", repeat=0):
    """
    Splits one base group's trial numbers into two disjoint halves.

    Deterministic in (session, feat, seed, shuffle_idx, group, repeat) and nothing else, so it is
    reproducible without being persisted: Steps 3 and 5 of the note need the choice decoder to
    train on B1 only and the projection to score B2 only, and they get the identical assignment by
    importing draw_b_split below and calling it with the same arguments.

    Halves are drawn within the group rather than as a boolean over every trial in the session, so
    the two are exactly balanced (differing by at most one trial when |G| is odd) rather than
    binomially scattered around |G|/2.

    cos_raw needs one call, on B, and that is what draw_b_split names. cos_cv needs all three of
    A, B and C halved, once per repeat, which is what `group` and `repeat` index. The two agree on
    B at repeat 0 by construction, so cos_cv's first repeat reuses cos_raw's B1/B2 exactly.
    """
    # a list seed is hashed as SeedSequence entropy, so the six fields can't collide the way an
    # arithmetic combination can. shuffle_idx is offset by 1 to keep the true run's 0 distinct
    rng = np.random.default_rng([
        int(session), FEATURES.index(feat), seed,
        0 if shuffle_idx is None else shuffle_idx + 1,
        GROUP_CODE[group], repeat,
    ])
    perm = rng.permutation(np.sort(np.asarray(trials)))
    h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
    assert len(np.intersect1d(h1, h2)) == 0, f"{group} halves overlap"
    assert len(h1) + len(h2) == len(trials), f"{group} halves don't partition {group}"
    return h1, h2


def draw_b_split(session, feat, b_trials, seed, shuffle_idx):
    """
    Group B's canonical halves, B1 and B2 (Issue 1 fix (a)) -- the assignment v_stim = r_B1 - r_A
    and v_pref = r_C - r_B2 are built from, and the one Steps 3 and 5 reproduce by importing this.
    """
    return draw_half_split(session, feat, b_trials, seed, shuffle_idx, group="B", repeat=0)
