"""Count-normalized LLM selection with fixed training and no covariate shift.

    python applications/llm/wsr.py
    Rscript applications/llm/plot_wsr.R

Keep the first 200 saved observations for training and fit random forests once.
Repeat 100 random calibration/test splits of the remaining 300 (100/200).
The loss is min(pos_err + 0.5*npos_err, K)/K, with K=5 by default.
SCoRE receives the exact positive support: at K=5, {0.1,0.2,...,1}.
Target levels and reported WSR use this normalized [0,1] scale.

The existing reward is accuracy + 3*not_ub. In the saved data, not_ub is binary;
we preserve this implemented definition, not an inferred count of matching labels.
Average reward is total selected reward divided by the number of test units.
Compare loss prediction and the adjusted reward score under deterministic,
heterogeneous, and homogeneous boosting, with gamma=alpha and paired draws.
Only raw replication CSVs are written; the R plotter exports PDF and TeX.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SCoRE import SCoRE_WSR, eBH


HERE = Path(__file__).resolve().parent
LABEL_COLUMNS = ("accuracy", "pos_err", "npos_err", "not_ub")
FEATURE_COLUMNS = (
    "generations|numsets", "lexical_sim", "semanticEntropy|unnorm",
    "generations|spectral_eigv_clip|disagreement_w", "generations|eccentricity|disagreement_w",
    "generations|degree|disagreement_w", "generations|spectral_eigv_clip|agreement_w",
    "generations|eccentricity|agreement_w", "generations|degree|agreement_w",
    "generations|spectral_eigv_clip|jaccard", "generations|eccentricity|jaccard",
    "generations|degree|jaccard",
)
SCORES = ("loss", "cost_reward")
METHODS = ("score_dtm", "score_hete", "score_homo")
REWARD_FLOOR = 1e-8
LABEL_COUNT = 14
DEFAULT_CLIP = 5.0


def positive_loss_support(clip=DEFAULT_CLIP):
    """All attainable positive losses, independent of observed outcomes."""
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be positive and finite.")
    raw_support = np.arange(1, 2 * LABEL_COUNT + 1, dtype=float) / 2
    return np.unique(np.minimum(raw_support, clip) / clip)


def prepare_data(frame, reward="confidence", clip=DEFAULT_CLIP):
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be positive and finite.")
    required = set(FEATURE_COLUMNS + LABEL_COLUMNS)
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing feature/label columns: {sorted(required - set(frame.columns))}")
    if not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("Features and labels must be finite.")
    if (frame[list(LABEL_COLUMNS)].to_numpy() < 0).any():
        raise ValueError("Error counts, accuracy, and not_ub must be nonnegative.")
    errors = frame[["pos_err", "npos_err"]].to_numpy(dtype=float)
    if (errors != np.floor(errors)).any() or (errors.sum(axis=1) > LABEL_COUNT).any():
        raise ValueError("pos_err and npos_err must be integer counts totaling at most 14 labels.")
    loss = np.minimum(errors[:, 0] + 0.5 * errors[:, 1], clip) / clip
    if reward == "confidence":
        observed_reward = frame.accuracy.to_numpy(dtype=float) + 3 * frame.not_ub.to_numpy(dtype=float)
    elif reward == "constant":
        observed_reward = np.ones(len(frame))
    else:
        raise ValueError(f"Unknown reward: {reward}")
    return dict(features=frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
                loss=loss, reward=observed_reward)


def fit_fixed_training(data, ntrain=200, trees=100, model_seed=0):
    """Train on the fixed prefix only, then predict all held-out observations."""
    if not 0 < ntrain < len(data["loss"]):
        raise ValueError("ntrain must leave a nonempty held-out pool.")
    seeds = np.random.SeedSequence([model_seed, 81]).spawn(2)
    predictions = {}
    for name, stream in zip(("loss", "reward"), seeds):
        model = RandomForestRegressor(n_estimators=trees, n_jobs=1,
                                      random_state=int(stream.generate_state(1)[0]))
        model.fit(data["features"][:ntrain], data[name][:ntrain])
        predictions[name + "_prediction"] = model.predict(data["features"][ntrain:])
    return dict(loss=data["loss"][ntrain:], reward=data["reward"][ntrain:], **predictions)


def score_functions(loss_prediction, reward_prediction, alpha_normalized):
    reward = np.maximum(reward_prediction, REWARD_FLOOR)
    excess = loss_prediction - alpha_normalized
    return dict(loss=loss_prediction, cost_reward=excess / reward)


def compare_scores(Lc, lc, lt, rc, rt, alpha_normalized, seed, clip=DEFAULT_CLIP):
    """No realized test outcomes enter selection. Cost is identically one."""
    if alpha_normalized == 0:
        # At zero risk budget, the common support permits positive loss for
        # every test unit. Use the empty selection without dividing by alpha.
        return {(score, method): np.empty(0, dtype=int)
                for score in SCORES for method in METHODS}
    sc = score_functions(lc, rc, alpha_normalized)
    st = score_functions(lt, rt, alpha_normalized)
    values = positive_loss_support(clip)
    support = np.broadcast_to(values, (len(lt), len(values)))
    level_key = list(np.float64(alpha_normalized).tobytes())
    # Preserve the original homogeneous stream, and share each method's draws
    # across scores. No randomness depends on the order of the comparison.
    uniforms = {
        "score_homo": np.random.default_rng(np.random.SeedSequence([seed, 82, *level_key])).uniform(),
        "score_hete": np.random.default_rng(np.random.SeedSequence([seed, 83, *level_key])).uniform(size=len(lt)),
    }
    selections = {}
    for score in SCORES:
        selected, evalues = SCoRE_WSR((Lc, sc[score]), (None, st[score]), alpha_normalized,
                                     alpha_normalized, finite_t=support, return_evals=True)
        selections[score, "score_dtm"] = selected
        for method in METHODS[1:]:
            with np.errstate(divide="ignore"):
                boosted = np.divide(evalues, uniforms[method], out=np.zeros_like(evalues), where=evalues > 0)
            selections[score, method] = eBH(boosted, alpha_normalized)
    return selections


def run_replication(pool, seed, alphas, ncalib=100, ntest=200, clip=DEFAULT_CLIP):
    if min(ncalib, ntest) <= 0 or ncalib + ntest > len(pool["loss"]):
        raise ValueError("Calibration/test sizes must be positive and fit in the held-out pool.")
    order = np.random.default_rng(np.random.SeedSequence([seed, 80])).permutation(len(pool["loss"]))
    calib, test = order[:ncalib], order[ncalib:ncalib + ntest]
    rows = []
    for alpha in alphas:
        level = float(alpha)
        selections = compare_scores(pool["loss"][calib], pool["loss_prediction"][calib],
                                     pool["loss_prediction"][test], pool["reward_prediction"][calib],
                                     pool["reward_prediction"][test], level, seed, clip)
        for (score, method), selected in selections.items():
            normalized_risk = float(pool["loss"][test][selected].mean()) if len(selected) else 0.0
            rows.append(dict(seed=seed, alpha=float(alpha), gamma=float(alpha),
                             alpha_normalized=level, gamma_normalized=level,
                             score=score, method=method, wsr=normalized_risk,
                             wsr_normalized=normalized_risk, n_selected=len(selected),
                             fraction_selected=len(selected)/ntest,
                             avg_reward=float(pool["reward"][test][selected].sum()/ntest)))
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE/"data"/"saved_feature_and_label_new.csv")
    parser.add_argument("--ntrain", type=int, default=200)
    parser.add_argument("--ncalib", type=int, default=100)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--nrep", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    parser.add_argument("--reward", choices=("confidence", "constant"), default="confidence")
    parser.add_argument("--alphas", nargs="+", type=float,
                        help="Targets on the normalized loss scale [0,1]; default 0,0.1,...,1")
    parser.add_argument("--output", type=Path, help="Default: results_wsr/<reward>")
    args = parser.parse_args(argv)
    if any(getattr(args, key) <= 0 for key in ("ntrain", "ncalib", "ntest", "nrep", "trees")):
        parser.error("Sample sizes, nrep, and trees must be positive.")
    if min(args.seed_start, args.model_seed) < 0:
        parser.error("Seeds must be nonnegative.")
    if not np.isfinite(args.clip) or args.clip <= 0:
        parser.error("clip must be positive and finite.")
    if args.alphas is None:
        args.alphas = (np.arange(11) / 10).tolist()
    if any(not np.isfinite(a) or not 0 <= a <= 1 for a in args.alphas):
        parser.error("Target levels must lie in [0, 1].")
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("Target levels must not contain duplicates.")
    if args.output is None:
        args.output = HERE/"results_wsr"/args.reward
    return args


def main(argv=None):
    args = parse_args(argv)
    data = prepare_data(pd.read_csv(args.data), args.reward, args.clip)
    if args.ntrain + args.ncalib + args.ntest > len(data["loss"]):
        raise ValueError("Requested train/calibration/test sizes exceed the available observations.")
    pool = fit_fixed_training(data, args.ntrain, args.trees, args.model_seed)
    rows = []
    for seed in tqdm(range(args.seed_start, args.seed_start + args.nrep), desc="LLM WSR"):
        rows.extend(run_replication(pool, seed, args.alphas, args.ncalib, args.ntest, args.clip))
    results = pd.DataFrame(rows)
    metadata = dict(ntrain=args.ntrain, ncalib=args.ncalib, ntest=args.ntest,
                    n_available=len(data["loss"]), n_features=len(FEATURE_COLUMNS),
                    reward=args.reward, reward_floor=REWARD_FLOOR,
                    reward_definition="accuracy_plus_3_not_ub" if args.reward == "confidence" else "one",
                    loss_definition="clipped_pos_err_plus_half_npos_err", normalization="count",
                    loss_normalizer=args.clip, clip=args.clip,
                    loss_support="finite", label_count=LABEL_COUNT, raw_loss_step=0.5,
                    shift="exchangeable", training="fixed_prefix", model="rf",
                    model_seed=args.model_seed, trees=args.trees, report_scale="normalized_clipped",
                    input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest())
    for key, value in metadata.items():
        results[key] = value
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/f"llm_wsr_seeds_{args.seed_start}_{args.seed_start + args.nrep - 1}.csv"
    results.to_csv(path, index=False)
    print(f"Saved {len(results)} raw replication rows to {path.resolve()}")


if __name__ == "__main__":
    main()
