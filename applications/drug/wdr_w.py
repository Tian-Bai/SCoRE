"""Drug WSR comparison under the original molecular-weight covariate shift.

    python applications/drug/wdr_w.py
    Rscript applications/drug/plot_wdr_w.R

Uses w(X)=sigmoid(abs(MW(X)-400)/400), as in sdr_w.py. Calibration draws are
uniform and independent of target draws with probabilities proportional to w.
Both samples are drawn WITH replacement from the held-out empirical pool:
duplicates and overlap are allowed, giving iid samples from two explicit
empirical distributions (unlike the old batches sampled without replacement).

Default weights are estimated once from the independent training fold using a
random-forest domain classifier on Morgan fingerprints, as in the old script.
An equally sized shifted training sample is drawn proportional to w. The odds
p/(1-p) estimate the density ratio; probabilities are clipped to [1e-3,1-1e-3].
Use --weights oracle for known ratios (their global normalization cancels).
Estimated weights do not imply exact finite-sample WSR control.

SA costs, four scores, homogeneous boosting, gamma=alpha, rewards and plots
are shared with wsr.py. Raw replication CSVs only; no summary CSV output.
"""

import hashlib
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

if __package__:
    from . import wsr
else:
    import wsr


PROBABILITY_CLIP = 1e-3


def shift_weights(molecular_weight):
    mw = np.asarray(molecular_weight, dtype=float)
    if mw.ndim != 1 or not np.isfinite(mw).all() or (mw <= 0).any():
        raise ValueError("Molecular weights must be a positive finite vector.")
    return expit(np.abs(mw - 400) / 400)


def fingerprints(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    features = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {value}")
        features.append(generator.GetFingerprintAsNumPy(molecule))
    return np.stack(features)


def estimate_weights(frame, seed=0, trees=100):
    """Fit using training structures only; no outcomes or held-out fitting."""
    required = {"split", "SMILES", "mw"}
    if not required.issubset(frame.columns):
        raise ValueError("Weight estimation requires split, SMILES, and mw columns.")
    train = frame.loc[frame.split == "train"]
    pool = frame.loc[frame.split == "calibtest"]
    if len(train) < 2:
        raise ValueError("Weight estimation needs at least two training molecules.")
    weights = shift_weights(train.mw)
    sample_stream, model_stream = np.random.SeedSequence([seed, 91]).spawn(2)
    rng = np.random.default_rng(sample_stream)
    shifted = rng.choice(len(train), size=len(train), replace=True, p=weights / weights.sum())
    features = fingerprints(train.SMILES)
    classifier = RandomForestClassifier(n_estimators=trees, n_jobs=1,
                                        random_state=int(model_stream.generate_state(1)[0]))
    classifier.fit(np.concatenate([features, features[shifted]]),
                   np.r_[np.zeros(len(train)), np.ones(len(train))])
    probability = classifier.predict_proba(fingerprints(pool.SMILES))[:, 1]
    probability = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return probability / (1 - probability)


def draw_indices(true_weights, seed, ncalib=None, ntest=None):
    """Independent iid source and target draws, with fixed sample sizes."""
    weights = np.asarray(true_weights, dtype=float)
    if weights.ndim != 1 or len(weights) < 4 or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Need at least four positive finite shift weights.")
    ncalib = len(weights) // 2 if ncalib is None else ncalib
    ntest = len(weights) - len(weights) // 2 if ntest is None else ntest
    source_stream, target_stream = np.random.SeedSequence([seed, 90]).spawn(2)
    calib = np.random.default_rng(source_stream).choice(len(weights), size=ncalib, replace=True)
    test = np.random.default_rng(target_stream).choice(len(weights), size=ntest, replace=True,
                                                      p=weights / weights.sum())
    return calib, test


def run_replication(data, true_weights, fitted_weights, seed,
                    count_alphas=wsr.COUNT_ALPHAS, cost_alphas=wsr.COST_ALPHAS,
                    scores=wsr.SCORES, methods=("score_homo",)):
    fitted_weights = np.asarray(fitted_weights, dtype=float)
    if (fitted_weights.shape != data["sa"].shape or
            not np.isfinite(fitted_weights).all() or (fitted_weights <= 0).any()):
        raise ValueError("Fitted weights must be positive, finite, and match the held-out pool.")
    if len(true_weights) != len(fitted_weights):
        raise ValueError("True and fitted weights must have the same length.")
    calib, test = draw_indices(true_weights, seed)
    return wsr.evaluate_split(data, calib, test, seed, count_alphas, cost_alphas, scores, methods,
                              weights=(fitted_weights[calib], fitted_weights[test]))


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = wsr.argument_parser(__doc__, output_help="Default: results_wdr_w/<dataset>/<reward>/<weights>")
    parser.add_argument("--weights", choices=("estimated", "oracle"), default="estimated")
    parser.add_argument("--weight-seed", type=int, default=0)
    parser.add_argument("--weight-trees", type=int, default=100)
    args = wsr.parse_args(argv, parser=parser, output_root="results_wdr_w")
    if args.weight_seed < 0 or args.weight_trees < 1:
        parser.error("weight-seed must be nonnegative and weight-trees positive.")
    if not any(arg == "--output" or arg.startswith("--output=") for arg in argv):
        args.output /= args.weights
    return args


def main(argv=None):
    args = parse_args(argv)
    frame = pd.read_csv(args.predictions)
    data, offset = wsr.prepare_data(frame, args.reward)
    if "mw" not in frame:
        raise ValueError("The prediction CSV must contain molecular weights (mw).")
    true_weights = shift_weights(frame.loc[frame.split == "calibtest", "mw"])
    if args.weights == "oracle":
        fitted_weights = true_weights
    else:
        print("Fitting density-ratio weights on training molecules...", flush=True)
        fitted_weights = estimate_weights(frame, args.weight_seed, args.weight_trees)
    rows = []
    for seed in tqdm(range(args.seed_start, args.seed_start + args.nrep), desc=f"Shifted drug WSR: {args.data}"):
        rows.extend(run_replication(data, true_weights, fitted_weights, seed,
                                     args.count_alphas, args.cost_alphas, args.scores, args.methods))
    results = pd.DataFrame(rows)
    metadata = dict(dataset=args.data, model=args.model, reward=args.reward, activity_offset=offset,
                    reward_floor=wsr.REWARD_FLOOR, shift="sigmoid_abs_mw_minus_400_over_400",
                    sampling="independent_empirical_with_replacement", weights=args.weights,
                    weight_seed=args.weight_seed, weight_trees=args.weight_trees,
                    probability_clip=PROBABILITY_CLIP, loss="SA_times_false_lead",
                    cost_definition="observed_SA", support="finite_SA", prediction_seed=args.prediction_seed,
                    predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest())
    for name, value in metadata.items():
        results[name] = value
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"drug_wdr_w_seeds_{args.seed_start}_{args.seed_start + args.nrep - 1}.csv"
    results.to_csv(path, index=False)
    print(f"Saved {len(results)} raw replication rows to {path.resolve()}")


if __name__ == "__main__":
    main()
