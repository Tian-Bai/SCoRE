"""Exchangeable-data comparison: three SCoRE variants and two binarized baselines.

Run from any directory; see simulation/README.md for examples and definitions.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SCoRE import (
    CS, SCoRE_WSR, eBH, gen_data_1, gen_data_2, gen_data_Jin2023,
    loss_1, loss_2, loss_Jin2023,
)


METHODS = ("score_dtm", "score_hete", "score_homo", "binarized_half", "binarized_quarter")
# Panel ID -> (original manuscript setting, cost relationship).
SETTINGS = {1: (1, "unit"), 2: (4, "unit"), 3: (6, "unit"),
            4: (6, "unrelated"), 5: (6, "increasing"), 6: (6, "decreasing")}
COST_DESIGNS = {"unit": "unit", "unrelated": "interaction_sigmoid_v1",
                "increasing": "interaction_sigmoid_v1", "decreasing": "interaction_sigmoid_v1"}
CLIP_LOOKUP = {
    (1, 0.05): 0.4, (1, 0.1): 0.6, (1, 0.2): 1.4, (1, 0.3): 2.6,
    (2, 0.05): 0.3, (2, 0.1): 0.4, (2, 0.2): 0.7, (2, 0.3): 1.2,
}
CONFIG_COLUMNS = [
    "setting", "source_setting", "case", "dgp", "sigma", "dim", "tau", "clip",
    "ntrain", "ncalib", "ntest", "trees", "target", "cost_mode", "cost_design", "cost_features", "score",
]


def unit_costs(X, mode):
    """Observable nonlinear costs in (1, 4); no regression function is needed.

    The sigmoid DGP's response signal increases with X_1*X_2, so decreasing
    cost in this interaction gives a positive cost-risk association. X_9*X_10
    is independent of the response. All three costs have the same marginal law.
    """
    if mode == "unit":
        return np.ones(len(X))
    if mode == "unrelated":
        index = 3 * X[:, 8] * X[:, 9]
    elif mode in ("increasing", "decreasing"):
        index = 3 * X[:, 0] * X[:, 1]
        if mode == "increasing":
            index = -index
    else:
        raise ValueError(f"Unknown cost mode: {mode}")
    return 1 + 3 / (1 + np.exp(-index))


def prediction_features(X, mode, cost_features):
    """Optionally expose the known cost to every competing prediction model."""
    return np.column_stack((X, unit_costs(X, mode))) if cost_features else X


def with_cost_design(raw):
    """Normalize legacy cost and prediction metadata before combining results."""
    legacy = raw.cost_mode.map({"unit": "unit", "unrelated": "x10_linear",
                                "increasing": "mu_sigmoid", "decreasing": "mu_sigmoid"})
    design = raw.cost_design.fillna(legacy) if "cost_design" in raw else legacy
    features = raw.cost_features.fillna(False) if "cost_features" in raw else False
    return raw.assign(cost_design=design, cost_features=features)


def make_data(setting, seed, ntrain=1000, ncalib=1000, ntest=100,
              sigma=0.1, dim=20, tau=10.0, trees=100, clip=None, cost_mode="unit",
              cost_features=False):
    """Fit loss scores on independent training data; test losses are for evaluation.

    Here setting is the original source setting, not the new panel ID.
    The predictor f for squared error is fitted on another independent sample.
    Separate seeded streams keep data/model randomness independent of boosting.
    """
    case, dgp = (setting - 1) // 2 + 1, (setting - 1) % 2 + 1
    generator = {1: gen_data_1, 2: gen_data_2, 3: gen_data_Jin2023}[case]
    streams = np.random.SeedSequence([seed, setting, 0]).spawn(6)

    def draw(size, stream):
        X, _, _, Y = generator(dgp, size, sigma, dim,
                               random_state=np.random.default_rng(stream))
        return X, Y

    def forest(stream):
        return RandomForestRegressor(n_estimators=trees, n_jobs=1,
                                     random_state=int(stream.generate_state(1)[0]))

    if case == 2:
        clip = CLIP_LOOKUP.get((dgp, sigma)) if clip is None else clip
        if clip is None or not np.isfinite(clip) or clip <= 0:
            raise ValueError("Specify --clip for squared-error settings at this sigma.")
        Xprep, Yprep = draw(ntrain, streams[0])
        predictor = forest(streams[1]).fit(Xprep, Yprep)

    def loss(X, Y):
        if case == 1:
            return loss_1(Y)
        if case == 2:
            return loss_2(Y, predictor, X, clip)
        return loss_Jin2023(Y, tau)

    Xtrain, Ytrain = draw(ntrain, streams[2])
    Xcalib, Ycalib = draw(ncalib, streams[3])
    Xtest, Ytest = draw(ntest, streams[4])
    features = lambda X: prediction_features(X, cost_mode, cost_features)
    model = forest(streams[5]).fit(features(Xtrain), loss(Xtrain, Ytrain))
    cc, ct = unit_costs(Xcalib, cost_mode), unit_costs(Xtest, cost_mode)
    # Predict the base loss, then use known costs to predict cost-scaled loss.
    return (cc * loss(Xcalib, Ycalib), cc * model.predict(features(Xcalib)),
            ct * loss(Xtest, Ytest), ct * model.predict(features(Xtest)), cc, ct)


def compare_methods(Lc, Sc, St, alpha, seed, setting, cc=None, ct=None):
    """Return selections without access to test losses, using common loss scores."""
    with_cost = cc is not None or ct is not None
    cc = np.ones(len(Lc)) if cc is None else np.asarray(cc)
    ct = np.ones(len(St)) if ct is None else np.asarray(ct)
    selected, evalues = SCoRE_WSR((Lc, Sc, cc), (None, St, ct), alpha, alpha,
                                 Lbar_t=ct, return_evals=True)
    selections = {"score_dtm": selected}
    # Use a level-derived seed so changing the alpha grid/order does not change
    # existing results. Independent method streams do not depend on call order.
    level_key = list(np.float64(alpha).tobytes())
    for method_id, method in enumerate(("score_hete", "score_homo"), start=1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, setting, method_id, *level_key]))
        uniform = rng.uniform(size=len(St) if method == "score_hete" else None)
        with np.errstate(divide="ignore"):
            boosted = np.divide(evalues, uniform, out=np.zeros_like(evalues), where=evalues > 0)
        selections[method] = eBH(boosted, alpha, costs=ct)
    for method, divisor in (("binarized_half", 2), ("binarized_quarter", 4)):
        u = alpha / divisor
        beta = (alpha - u) / (1 - u)
        if with_cost:
            # L/c <= u + (1-u) * 1{L/c > u}. Control the cost-normalized
            # binary loss at beta; this controls the original WSR at alpha.
            binary_loss = cc * (Lc / cc > u)
            selections[method] = SCoRE_WSR((binary_loss, Sc, cc), (None, St, ct),
                                            beta, beta, finite_t=ct)
        else:
            selections[method] = CS(((Lc > u).astype(int), Sc), (None, St), alpha=beta)
    return selections


def run_replication(setting, seed, alphas, **data_options):
    source_setting, cost_mode = SETTINGS[setting]
    Lc, Sc, Lt, St, cc, ct = make_data(source_setting, seed, cost_mode=cost_mode, **data_options)
    costs = {} if cost_mode == "unit" else dict(cc=cc, ct=ct)
    rows = []
    for alpha in alphas:
        for method, selected in compare_methods(Lc, Sc, St, alpha, seed, setting, **costs).items():
            n_selected = len(selected)
            wsr = float(np.sum(Lt[selected]) / np.sum(ct[selected])) if n_selected else 0.0
            u = alpha / 2 if method == "binarized_half" else alpha / 4 if method == "binarized_quarter" else np.nan
            testing_level = alpha if np.isnan(u) else (alpha-u)/(1-u)
            rows.append(dict(seed=seed, alpha=alpha, method=method,
                             gamma=testing_level if method.startswith("score_") or cost_mode != "unit" else np.nan,
                             u=u, testing_level=testing_level,
                             wsr=wsr, n_selected=n_selected,
                             fraction_selected=n_selected / len(Lt),
                             nonempty=int(n_selected > 0), violation=int(wsr > alpha)))
    return rows


def summarize(raw):
    """Unconditional Monte Carlo means and standard errors over replications."""
    raw = with_cost_design(raw)
    keys = CONFIG_COLUMNS + ["alpha", "method"]
    grouped = raw.groupby(keys, dropna=False, sort=True)
    summary = grouped.agg(
        nrep=("seed", "size"), wsr_mean=("wsr", "mean"), wsr_sd=("wsr", "std"),
        fraction_selected_mean=("fraction_selected", "mean"),
        fraction_selected_sd=("fraction_selected", "std"),
        nonempty_mean=("nonempty", "mean"), violation_mean=("violation", "mean"),
    ).reset_index()
    for metric in ("wsr", "fraction_selected"):
        summary[f"{metric}_se"] = summary[f"{metric}_sd"] / np.sqrt(summary.nrep)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument("--nrep", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ncalib", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--cost-features", action="store_true",
                        help="Include known c(X) alongside X in the loss/reward predictors for every score")
    parser.add_argument("--alphas", nargs="+", type=float, default=[i / 20 for i in range(1, 11)])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results_wsr")
    args = parser.parse_args(argv)
    if any(getattr(args, key) <= 0 for key in ("nrep", "ntrain", "ncalib", "ntest", "trees")):
        parser.error("Sample sizes, nrep, and trees must be positive.")
    if args.seed_start < 0 or args.dim < 4:
        parser.error("seed-start must be nonnegative and dim must be at least 4.")
    if 4 in args.settings and args.dim < 10:
        parser.error("The unrelated-cost setting requires dim >= 10 (uses X_9 and X_10).")
    if not np.isfinite(args.sigma) or args.sigma < 0 or not np.isfinite(args.tau) or args.tau <= 0:
        parser.error("sigma must be finite and nonnegative; tau must be finite and positive.")
    if any(not 0 < alpha < 1 for alpha in args.alphas):
        parser.error("alphas must lie strictly between 0 and 1.")
    if len(set(args.settings)) != len(args.settings) or len(set(args.alphas)) != len(args.alphas):
        parser.error("settings and alphas must not contain duplicates.")
    if args.clip is not None and (not np.isfinite(args.clip) or args.clip <= 0):
        parser.error("clip must be finite and positive.")
    for setting in args.settings:
        source, _ = SETTINGS[setting]
        if source in (3, 4) and args.clip is None and (source - 2, args.sigma) not in CLIP_LOOKUP:
            parser.error("Specify --clip for squared-error settings at this sigma.")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    options = {key: getattr(args, key) for key in
               ("ntrain", "ncalib", "ntest", "sigma", "dim", "tau", "trees", "clip", "cost_features")}
    for setting in args.settings:
        source_setting, cost_mode = SETTINGS[setting]
        case, dgp = (source_setting - 1) // 2 + 1, (source_setting - 1) % 2 + 1
        rows = []
        for seed in tqdm(range(args.seed_start, args.seed_start + args.nrep), desc=f"Setting {setting}"):
            rows.extend(run_replication(setting, seed, args.alphas, **options))
        frame = pd.DataFrame(rows)
        metadata = dict(options, setting=setting, source_setting=source_setting, case=case, dgp=dgp,
                        target="WSR", cost_mode=cost_mode,
                        cost_design=COST_DESIGNS[cost_mode], score="loss")
        metadata["clip"] = (args.clip if args.clip is not None else CLIP_LOOKUP[(dgp, args.sigma)]) if case == 2 else np.nan
        for key, value in metadata.items():
            frame[key] = value
        # Save each completed setting, so an interrupted later setting does
        # not discard earlier results. Seed ranges support separate run chunks.
        stem = f"setting_{setting}_seeds_{args.seed_start}_{args.seed_start + args.nrep - 1}"
        frame.to_csv(args.output / f"{stem}.csv", index=False)
    print(f"Saved raw replications to {args.output.resolve()}")


if __name__ == "__main__":
    main()
