"""
Simulation of the stim/belief projection analysis: if the stim axis (A vs. B) and the belief axis
(B vs. C) really were aligned, would decode_pref_on_choice_axis.py show it, with the real runs'
sessions, units and trial counts? See claude_notes/Stim Belief projection simulation proposal.md.

Generative model, per (population, feature), isotropic unit variance noise:

    r_A = 0
    r_B = s u
    r_C = s u + v_pref,   v_pref = p (cos(theta) u_P + sin(theta) u_perp)   on the pref units P, 0 elsewhere

u is a unit vector over the axis run's units. P is, per session, as many units as the real pref run
has, taken at random: units are iid, so which ones doesn't matter, and by construction they are the
units that carry the pref signal -- which is what the real ones are selected for. u_P is u restricted
to P and renormalized, u_perp a unit vector on P orthogonal to it, so cos(theta) is exactly the
alignment of v_stim and v_pref on P, the only units the projection sees. s is the norm of v_stim over
all units, p the norm of v_pref.

Everything else follows the real runs, with the layout (sessions, units per session, pref units per
session, post-balance trial counts per condition) read by stim_belief_sim_layout.py:

- per session, per condition train/test splits, test_ratio of the trials held out (rounded up)
- pseudo trials resampled per session and condition, one draw shared by a session's units, the numpy
  equivalent of pseudo_utils.generate_pseudo_population_v2 -- which is what keeps a single time bin
  down to minutes
- the real decoder, NormedDropoutMultinomialLogisticRegressor through ModelWrapper and Trainer
- the stim axis from decode_pref_on_choice_axis.axis_from_models, restricted to the pref units,
  sign fixed so that C sits on the Chose end, and only a threshold refit on B vs. C
- shuffles permute the condition labels within a session and regenerate the splits. Trials are iid
  here, so that is equivalent to the real runs' session permutation

s and p are set analytically rather than by grid search. A decoder's weights are the true mean
difference plus the noise of estimating it from the training trials, which adds
T = sum over units of (1/n_0,train + 1/n_1,train) to their squared norm, so with unit noise its
expected test accuracy is

    acc = Phi( d^2 / (2 sqrt(d^2 + T)) ),    d = norm of the mean difference

s is solved from the stim gap over its peak window (T over all the axis run's units), p from the pref
gap over all bins (T over the pref units), each so the mean over features hits 0.5 + gap.

The trained decoder is less efficient than a plain mean difference, which the formula takes as
kappa T in place of T, kappa fit once to the real decoder on the ITC, ACC and whole population
layouts (12 features, a grid of 4 signal sizes each): 1.2 for stim decoding, 1.65 for pref, whose
per population fits scatter from 0.6 to 2.4, mostly the noise of a single draw at low signal. With
kappa = 1 the simulated pref gap came out at a third of its ITC target. Both are flags, and are
recorded with the calibration.

The noise scale is not a free parameter: the batch norm and the threshold refit are scale invariant,
so only s / sigma and p / sigma matter, and sigma = 1.

Modes:

- calibrate: writes s, p per population to CALIBRATION_PATH, from the b_split layouts and the
  targets stim_belief_sim_layout.py read. No decoders, runs in seconds. no_split uses the same s, p:
  the signal is the same, only the analysis differs.
- simulate, one (population, feature, split, repeat) per job: fits the stim axis once, then for each
  cos(theta) draws C and runs pref decoding and the projection, each with shuffles. C's noise, the
  splits and the pseudo trial draws are shared across cos(theta), so the curves over it are smooth.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from scipy.optimize import brentq

import utils.pseudo_classifier_utils as pseudo_classifier_utils
from constants.behavioral_constants import *
from constants.decoding_constants import *
from models.trainer import Trainer
from models.model_wrapper import ModelWrapper
from models.multinomial_logistic_regressor import NormedDropoutMultinomialLogisticRegressor
from scripts.pseudo_decoding.belief_partitions.belief_partition_configs import BeliefPartitionConfigs
from scripts.pseudo_decoding.belief_partitions.decode_pref_on_choice_axis import axis_from_models
from scripts.pseudo_decoding.belief_partitions.stim_belief_sim_layout import (
    LAYOUT_PATH, OUTPUT_PATH, POPULATIONS, SPLIT_VARIANTS, TARGETS_SPLIT
)

CALIBRATION_PATH = os.path.join(OUTPUT_PATH, "calibration.json")

# how much more estimation noise the trained decoder behaves as having than a mean difference, see
# the module docstring
STIM_NOISE_FACTOR = 1.2
PREF_NOISE_FACTOR = 1.65

# conditions of each decoder, in the order the real split dataframes list them. A: Not Chose,
# B: Chose for the stim axis run; B: High Not X, C: High X for the pref run
STIM_CONDS = {"A": "Not Chose", "B": "Chose"}
PREF_CONDS = {"B": "High Not X", "C": "High X"}

DEFAULT_COS_THETAS = [-0.5, 0, 0.25, 0.5, 0.75, 1]

# stage codes for the seeds, so every random draw has its own stream
SEED_SIGNAL, SEED_TRIALS, SEED_SPLITS, SEED_PSEUDO, SEED_SHUFFLE, SEED_TORCH, SEED_C, SEED_UNITS = range(8)


def rng_for(args, stage, *extra):
    """
    A list seed is hashed as SeedSequence entropy, same idiom as stim_belief_groups.draw_half_split
    """
    return np.random.default_rng([
        POPULATIONS.index(args.population), args.feat_idx,
        list(SPLIT_VARIANTS).index(args.split), args.repeat, stage, *extra
    ])


def unit_vector(rng, n):
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def draw_pref_units(layout, rng):
    """
    Per session, the real run's number of pref units, at random, as columns within the session
    """
    return [np.sort(rng.choice(row.n_units, row.n_pref_units, replace=False)) for row in layout.itertuples()]


def draw_signal(n_units, pref_cols, rng):
    """
    u over all units, and on the pref units u restricted and renormalized, u_P, with a unit vector
    orthogonal to it, u_perp
    """
    u = unit_vector(rng, n_units)
    u_p = u[pref_cols] / np.linalg.norm(u[pref_cols])
    u_perp = rng.standard_normal(len(pref_cols))
    u_perp -= (u_perp @ u_p) * u_p
    return u, u_p, u_perp / np.linalg.norm(u_perp)


def pref_vector(n_units, pref_cols, p, cos_theta, u_p, u_perp):
    """
    v_pref over all units, p (cos(theta) u_P + sin(theta) u_perp) on the pref units, 0 elsewhere
    """
    v_pref = np.zeros(n_units)
    v_pref[pref_cols] = p * (cos_theta * u_p + np.sqrt(1 - cos_theta**2) * u_perp)
    return v_pref


def unit_slices(layout):
    """
    Each session's columns in the concatenated unit vector
    """
    ends = np.cumsum(layout.n_units.to_numpy())
    return [slice(end - n, end) for n, end in zip(layout.n_units, ends)]


def simulate_noise(layout, split, rng):
    """
    Per session unit variance noise for A, the axis run's B, and the pref run's B, as trials x units.
    Under b_split the two B's are disjoint draws. Under no_split the pref run's B is a subsample of
    the axis run's, so a pref B trial also trains the axis, as in the earlier leaky runs.
    Means are added separately, so the same noise serves every value of s.
    """
    noise = []
    for row in layout.itertuples():
        a = rng.standard_normal((row.n_A, row.n_units))
        b_axis = rng.standard_normal((row.n_B_axis, row.n_units))
        if split == "b_split":
            b_pref = rng.standard_normal((row.n_B_pref, row.n_units))
        else:
            if row.n_B_pref > row.n_B_axis:
                raise ValueError(f"session {row.session}: pref run has more B trials than the axis run")
            b_pref = b_axis[rng.choice(row.n_B_axis, row.n_B_pref, replace=False)]
        noise.append({"A": a, "B_axis": b_axis, "B_pref": b_pref})
    return noise


def simulate_c_noise(layout, rng):
    return [rng.standard_normal((row.n_C, row.n_units)) for row in layout.itertuples()]


def make_splits(counts, num_splits, test_ratio, rng):
    """
    Per condition, num_splits random train/test splits of trials 0..n-1, as ConditionTrialSplitter
    does: shuffle, hold out ceil(n * test_ratio) for test.
    Returns {cond: [(train, test), ...]}
    """
    splits = {}
    for cond, n in counts.items():
        splits[cond] = []
        for _ in range(num_splits):
            trials = rng.permutation(n)
            split_at = int(np.ceil(n * test_ratio))
            splits[cond].append((trials[split_at:], trials[:split_at]))
    return splits


def pseudo_trials(sess_data, sess_splits, conds, split_idx, num_train, num_test, rngs):
    """
    Pseudo trials across sessions, the numpy equivalent of pseudo_utils.generate_pseudo_population_v2:
    per session, per condition in order, draws train then test trials with replacement, one draw
    shared by all of the session's units, and concatenates sessions along units.

    sess_data: per session {cond: trials x units}, sess_splits: per session make_splits output,
    rngs: one generator per session. Returns x_train, y_train, x_test, y_test, labels being conds.
    """
    x_train, x_test = [], []
    for data, splits, rng in zip(sess_data, sess_splits, rngs):
        train_blocks, test_blocks = [], []
        for cond in conds:
            train, test = splits[cond][split_idx]
            train_blocks.append(data[cond][rng.choice(train, num_train)])
            test_blocks.append(data[cond][rng.choice(test, num_test)])
        x_train.append(np.vstack(train_blocks))
        x_test.append(np.vstack(test_blocks))
    y_train = np.repeat(conds, num_train)
    y_test = np.repeat(conds, num_test)
    return np.hstack(x_train), y_train, np.hstack(x_test), y_test


def shuffle_labels(data, conds, rng):
    """
    Permutes trials between two conditions within a session, keeping each condition's count
    """
    pooled = np.vstack([data[cond] for cond in conds])
    perm = rng.permutation(len(pooled))
    n_first = len(data[conds[0]])
    return {conds[0]: pooled[perm[:n_first]], conds[1]: pooled[perm[n_first:]]}


def fit_decoder(x_train, y_train, classes, args, torch_seed):
    torch.manual_seed(torch_seed)
    init_params = {"n_inputs": x_train.shape[1], "p_dropout": args.p_dropout, "n_classes": len(classes)}
    trainer = Trainer(learning_rate=args.learning_rate, max_iter=args.max_iter)
    return ModelWrapper(NormedDropoutMultinomialLogisticRegressor, init_params, trainer, classes).fit(x_train, y_train)


def decode(sess_data, conds, mode, args, shuffle_idx, keep_models=False, project_axis=None):
    """
    The real decoding run on simulated trials: splits, pseudo trials, one decoder per split.
    If project_axis is given, also scores the same pseudo trials along it with a refit threshold,
    positive class MODE_TO_DIRECTION_LABELS[mode]["high"].
    shuffle_idx None is the true run, otherwise labels are permuted within session first.
    Returns rows of accuracies, and the models if keep_models.
    """
    # the decoder's mode is in the seed too, so stim and pref runs don't share resampling streams
    mode_code = list(MODE_TO_CLASSES).index(mode)
    stage_extra = [mode_code, 0] if shuffle_idx is None else [mode_code, 1, shuffle_idx]
    if shuffle_idx is not None:
        sess_data = [
            shuffle_labels(data, conds, rng_for(args, SEED_SHUFFLE, *stage_extra, sess_idx))
            for sess_idx, data in enumerate(sess_data)
        ]
    sess_splits = [
        make_splits({cond: len(data[cond]) for cond in conds}, args.num_splits, args.test_ratio,
                    rng_for(args, SEED_SPLITS, *stage_extra, sess_idx))
        for sess_idx, data in enumerate(sess_data)
    ]
    classes = MODE_TO_CLASSES[mode]
    high = MODE_TO_DIRECTION_LABELS[mode]["high"]
    rows, models = [], []
    for split_idx in range(args.num_splits):
        rngs = [rng_for(args, SEED_PSEUDO, *stage_extra, split_idx, sess_idx) for sess_idx in range(len(sess_data))]
        x_train, y_train, x_test, y_test = pseudo_trials(
            sess_data, sess_splits, conds, split_idx, args.num_train_per_cond, args.num_test_per_cond, rngs
        )
        model = fit_decoder(x_train, y_train, classes, args, int(rng_for(args, SEED_TORCH, *stage_extra, split_idx).integers(2**31)))
        rows.append({"analysis": mode, "shuffle_idx": shuffle_idx, "split_idx": split_idx, "Accuracy": model.score(x_test, y_test)})
        if keep_models:
            models.append(model)
        if project_axis is not None:
            proj_train, proj_test = x_train @ project_axis, x_test @ project_axis
            threshold = pseudo_classifier_utils.fit_threshold(proj_train, y_train == high)
            acc = pseudo_classifier_utils.score_threshold(proj_test, y_test == high, threshold)
            rows.append({"analysis": "proj", "shuffle_idx": shuffle_idx, "split_idx": split_idx, "Accuracy": acc})
    return rows, models


def stim_data(noise, slices, s, u):
    """
    Per session A and B trials of the axis run: A = noise, B = s u + noise
    """
    return [
        {STIM_CONDS["A"]: n["A"], STIM_CONDS["B"]: n["B_axis"] + s * u[sl]}
        for n, sl in zip(noise, slices)
    ]


def pref_data(noise, c_noise, slices, mean_b, mean_c, unit_idxs):
    """
    Per session B and C trials of the pref run, restricted to that session's pref units, for the
    sessions the pref run has
    """
    data = []
    for n, cn, sl, idxs in zip(noise, c_noise, slices, unit_idxs):
        if len(idxs) == 0 or len(cn) == 0:
            continue
        b = (n["B_pref"] + mean_b[sl])[:, idxs]
        c = (cn + mean_c[sl])[:, idxs]
        data.append({PREF_CONDS["B"]: b, PREF_CONDS["C"]: c})
    return data


def global_idxs(slices, unit_idxs):
    """
    Pref units as columns of the concatenated unit vector, in the order pref_data lays them out, for
    the sessions pref_data keeps
    """
    return np.concatenate([np.arange(sl.start, sl.stop)[idxs] for sl, idxs in zip(slices, unit_idxs)])


def cosines(layout, noise, c_noise, slices, s, u, mean_c, split):
    """
    cos_raw and cos_unb of the population vector analysis, over the units of sessions that have all
    three groups, and cos_true, the cosine of the true vectors over those same units: v_pref is 0 off
    the pref units, so over the full population it is cos(theta) shrunk by the share of u on them.
    The numerator correction is zero under b_split, whose two B's are disjoint, and
    sum_u s2_B / n_B_axis under no_split, whose pref B is a subsample of the axis B
    """
    v_stim, v_pref, num_corr, sq_stim_corr, sq_pref_corr = [], [], 0.0, 0.0, 0.0
    true_stim, true_pref = [], []
    for row, n, cn, sl in zip(layout.itertuples(), noise, c_noise, slices):
        if row.n_C < 2 or row.n_B_pref < 2:
            continue
        a, b_axis = n["A"], n["B_axis"] + s * u[sl]
        b_pref, c = n["B_pref"] + s * u[sl], cn + mean_c[sl]
        v_stim.append(b_axis.mean(0) - a.mean(0))
        v_pref.append(c.mean(0) - b_pref.mean(0))
        true_stim.append(s * u[sl])
        true_pref.append(mean_c[sl] - s * u[sl])
        s2 = {k: x.var(0, ddof=1) for k, x in [("A", a), ("Ba", b_axis), ("Bp", b_pref), ("C", c)]}
        if split == "no_split":
            num_corr += np.sum(s2["Ba"] / len(b_axis))
        sq_stim_corr += np.sum(s2["A"] / len(a) + s2["Ba"] / len(b_axis))
        sq_pref_corr += np.sum(s2["Bp"] / len(b_pref) + s2["C"] / len(c))
    v_stim, v_pref = np.concatenate(v_stim), np.concatenate(v_pref)
    num = v_stim @ v_pref
    sq_stim, sq_pref = v_stim @ v_stim, v_pref @ v_pref
    cos_raw = num / np.sqrt(sq_stim * sq_pref)
    sq_stim_unb, sq_pref_unb = sq_stim - sq_stim_corr, sq_pref - sq_pref_corr
    cos_unb = (num + num_corr) / np.sqrt(sq_stim_unb * sq_pref_unb) if min(sq_stim_unb, sq_pref_unb) > 0 else np.nan
    return {
        "cos_raw": cos_raw, "cos_unb": cos_unb, "cos_true": cos(np.concatenate(true_stim), np.concatenate(true_pref)),
        "att_stim": np.sqrt(max(sq_stim_unb, 0) / sq_stim), "att_pref": np.sqrt(max(sq_pref_unb, 0) / sq_pref),
    }


def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def load_layout(args):
    layout = pd.read_pickle(LAYOUT_PATH)["layouts"][(args.population, args.feat, args.split)]
    return layout.reset_index(drop=True)


def train_count(n, test_ratio):
    return n - np.ceil(n * test_ratio)


def expected_acc(d2, T):
    """
    Expected test accuracy of a decoder whose weights are the mean difference, squared norm d2, plus
    estimation noise of total power T, under unit isotropic noise
    """
    return norm.cdf(d2 / (2 * np.sqrt(d2 + T)))


def noise_power(layout, units_col, n0_col, n1_col, test_ratio):
    """
    T = sum over units of (1/n_0,train + 1/n_1,train), each unit taking its session's counts
    """
    rows = layout[layout[units_col] > 0]
    n0, n1 = train_count(rows[n0_col], test_ratio), train_count(rows[n1_col], test_ratio)
    return float(np.sum(rows[units_col] * (1 / n0 + 1 / n1)))


def solve_signal(gap, Ts):
    """
    The mean difference norm at which expected accuracy, averaged over features, is 0.5 + gap
    """
    if gap <= 0:
        return 0.0
    Ts = np.asarray(Ts)
    return brentq(lambda d: expected_acc(d**2, Ts).mean() - (0.5 + gap), 0, 100)


def analytic_calibration(test_ratio, stim_noise_factor, pref_noise_factor):
    """
    s and p per population, with the numbers they come from: the target gaps, the features' mean
    noise power, and the stim axis quality and projection ceiling they imply
    """
    store = pd.read_pickle(LAYOUT_PATH)
    layouts, targets = store["layouts"], store["targets"].set_index("population")
    calibration = {}
    for population in POPULATIONS:
        feat_layouts = [layouts[(population, feat, TARGETS_SPLIT)] for feat in FEATURES]
        T_stim = [stim_noise_factor * noise_power(l, "n_units", "n_A", "n_B_axis", test_ratio) for l in feat_layouts]
        T_pref = [pref_noise_factor * noise_power(l, "n_pref_units", "n_B_pref", "n_C", test_ratio) for l in feat_layouts]
        stim_gap, pref_gap = targets.loc[population, "stim_gap_peak"], targets.loc[population, "pref_gap_all"]
        s, p = solve_signal(stim_gap, T_stim), solve_signal(pref_gap, T_pref)
        # the axis averages 8 splits of one pool, so its noise is roughly that of the full counts
        T_axis = np.mean([np.sum(l.n_units * (1 / l.n_A + 1 / l.n_B_axis)) for l in feat_layouts])
        rho = s / np.sqrt(s**2 + T_axis)
        calibration[population] = {
            "s": s, "p": p,
            "stim_noise_factor": stim_noise_factor, "pref_noise_factor": pref_noise_factor,
            "stim_gap_target": stim_gap, "pref_gap_target": pref_gap,
            "T_stim": float(np.mean(T_stim)), "T_pref": float(np.mean(T_pref)),
            # expected cos(estimated stim axis, true stim axis), and the projection's accuracy along it
            # at cos(theta) = 1 with the true pref means, a rough ceiling
            "rho_expected": rho, "proj_acc_ceiling": norm.cdf(p * rho / 2),
        }
    return calibration


def simulate(args):
    """
    Fits the stim axis once, then pref decoding and the projection at each cos(theta)
    """
    calib = json.load(open(args.calibration_path))[args.population]
    s, p = calib["s"], calib["p"]
    layout = load_layout(args)
    slices = unit_slices(layout)
    n_units = layout.n_units.sum()
    unit_idxs = draw_pref_units(layout, rng_for(args, SEED_UNITS))
    pref_cols = global_idxs(slices, unit_idxs)
    u, u_p, u_perp = draw_signal(n_units, pref_cols, rng_for(args, SEED_SIGNAL))
    noise = simulate_noise(layout, args.split, rng_for(args, SEED_TRIALS))
    c_noise = simulate_c_noise(layout, rng_for(args, SEED_C))

    rows = []
    stim_rows, models = decode(stim_data(noise, slices, s, u), list(STIM_CONDS.values()), "choice", args, None, keep_models=True)
    rows += [{**r, "cos_theta": np.nan} for r in stim_rows]
    classes = MODE_TO_CLASSES["choice"]
    axis = axis_from_models(
        models, classes.index(MODE_TO_DIRECTION_LABELS["choice"]["high"]), classes.index(MODE_TO_DIRECTION_LABELS["choice"]["low"])
    )
    # true mean difference expected from the axis run's trial counts, for rho's formula
    per_unit_noise = np.concatenate([np.full(row.n_units, 1 / row.n_A + 1 / row.n_B_axis) for row in layout.itertuples()])
    rho_formula = s / np.sqrt(s**2 + per_unit_noise.sum())

    extras = []
    shuffles = [None] + list(range(args.num_shuffles))
    mean_b = s * u
    proj_axis = axis[pref_cols]
    for cos_theta in args.cos_thetas:
        v_pref = pref_vector(n_units, pref_cols, p, cos_theta, u_p, u_perp)
        mean_c = mean_b + v_pref
        data = pref_data(noise, c_noise, slices, mean_b, mean_c, unit_idxs)
        for shuffle_idx in shuffles:
            res, _ = decode(data, list(PREF_CONDS.values()), "pref", args, shuffle_idx, project_axis=proj_axis)
            rows += [{**r, "cos_theta": cos_theta} for r in res]
        extras.append({
            "cos_theta": cos_theta,
            "s": s, "p": p,
            "n_units": n_units, "n_pref_units": len(pref_cols),
            # how well the estimated axis recovers the true stim direction, all units and pref units
            "rho": cos(axis, u), "rho_pref": cos(proj_axis, u[pref_cols]),
            "rho_formula": rho_formula,
            # expected accuracy along the estimated axis for the true means, isotropic unit noise.
            # The sign is fixed, so when C sits on the Not Chose end the best threshold is past the
            # last trial and every trial gets one label: the refit floors the accuracy at 0.5, and
            # anti-alignment reads as chance rather than below it
            "pred_proj_acc_signed": norm.cdf(proj_axis @ v_pref[pref_cols] / (2 * np.linalg.norm(proj_axis))),
            # pref along the true direction, and what the calibration expects of the trained decoder
            "pred_pref_acc": norm.cdf(p / 2),
            "calib_pref_acc": expected_acc(p**2, calib["pref_noise_factor"] * noise_power(layout, "n_pref_units", "n_B_pref", "n_C", args.test_ratio)),
            "calib_stim_acc": expected_acc(s**2, calib["stim_noise_factor"] * noise_power(layout, "n_units", "n_A", "n_B_axis", args.test_ratio)),
            **cosines(layout, noise, c_noise, slices, s, u, mean_c, args.split),
        })
        extras[-1]["pred_proj_acc"] = max(extras[-1]["pred_proj_acc_signed"], 0.5)
        print(f"simulated cos(theta) = {cos_theta}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(extras)


def output_file(args):
    name = f"simulate/{args.split}/{args.population}_{args.feat}_repeat_{args.repeat}{args.output_suffix}.pickle"
    path = os.path.join(args.output_path, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def main(args):
    if args.mode == "calibrate":
        calibration = analytic_calibration(args.test_ratio, args.stim_noise_factor, args.pref_noise_factor)
        print(pd.DataFrame(calibration).T.round(3).to_string())
        os.makedirs(os.path.dirname(args.calibration_path), exist_ok=True)
        with open(args.calibration_path, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"wrote {args.calibration_path}", flush=True)
        return
    if args.population is None or args.feat_idx is None:
        raise ValueError("simulate needs --population and --feat_idx")
    args.feat = FEATURES[args.feat_idx]
    print(f"{args.mode}: {args.population}, {args.feat}, {args.split}, repeat {args.repeat}", flush=True)
    torch.set_num_threads(args.num_threads)
    meta = {"population": args.population, "feat": args.feat, "split": args.split, "repeat": args.repeat}
    accs, extras = simulate(args)
    out = {"accs": accs.assign(**meta), "extras": extras.assign(**meta)}
    path = output_file(args)
    pd.to_pickle(out, path)
    print(f"wrote {path}", flush=True)


def get_parser():
    configs = BeliefPartitionConfigs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "simulate"], required=True)
    # simulate only: calibrate covers every population at once
    parser.add_argument("--population", choices=POPULATIONS, default=None)
    parser.add_argument("--feat_idx", type=int, default=None)
    parser.add_argument("--split", choices=list(SPLIT_VARIANTS), default="b_split")
    # which independent draw of signal and trials this job is
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--num_shuffles", type=int, default=10)
    parser.add_argument("--cos_thetas", type=float, nargs="+", default=DEFAULT_COS_THETAS)
    # the real runs' decoding settings
    parser.add_argument("--num_splits", type=int, default=configs.num_splits)
    parser.add_argument("--test_ratio", type=float, default=configs.test_ratio)
    parser.add_argument("--num_train_per_cond", type=int, default=configs.num_train_per_cond)
    parser.add_argument("--num_test_per_cond", type=int, default=configs.num_test_per_cond)
    parser.add_argument("--learning_rate", type=float, default=configs.learning_rate)
    parser.add_argument("--max_iter", type=int, default=configs.max_iter)
    parser.add_argument("--p_dropout", type=float, default=configs.p_dropout)
    parser.add_argument("--calibration_path", default=CALIBRATION_PATH)
    parser.add_argument("--stim_noise_factor", type=float, default=STIM_NOISE_FACTOR)
    parser.add_argument("--pref_noise_factor", type=float, default=PREF_NOISE_FACTOR)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    # for smoke runs, so they don't overwrite the real outputs
    parser.add_argument("--output_suffix", default="")
    parser.add_argument("--num_threads", type=int, default=1)
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
