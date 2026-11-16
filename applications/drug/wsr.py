"""Drug selection with count and SA-cost normalization (Section 8.1).

Run from the repository root:
    python applications/drug/wsr.py
    Rscript applications/drug/plot_wsr.R

Uses saved, independently trained predictions and repeats exchangeable 50/50
calibration/test splits of the held-out pool. Both normalizations use the same
loss L=SA*1{Y<=c0}; only the denominator changes (1 versus SA per selected unit).
SA is observed, not predicted. The positive loss support is exactly {SA}.
Gamma=alpha throughout; defaults compare four scores with homogeneous boosting.
Raw replication CSVs only; no summary CSVs.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SCoRE import SCoRE_WSR, SCoRE_WSR_w, eBH


HERE = Path(__file__).resolve().parent
METHODS = ("score_dtm", "score_hete", "score_homo")
SCORES = ("loss", "cost", "reward", "cost_reward")
REWARD_FLOOR = 1e-8
COUNT_ALPHAS = np.arange(1, 7) / 4
COST_ALPHAS = np.arange(1, 11) / 20


def prepare_data(frame, reward="diversity"):
    """Prepare predictions without using held-out outcomes to define scores.

    Diversity is observed 1-AvgTanimoto. For activity, use max(Y+offset,0),
    with offset=max(0,1-min(training Y)); predictions receive the same offset.
    This avoids the old test-outcome-dependent shift for negative activities.
    """
    required = {"split", "sa", "1{Y<=c}", "pred_1{Y<=c}"}
    if reward == "diversity":
        required.add("tanimoto")
    elif reward == "activity":
        required.update(("Y", "pred_Y"))
    elif reward != "constant":
        raise ValueError(f"Unknown reward: {reward}")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    pool = frame.loc[frame.split == "calibtest"].reset_index(drop=True)
    if len(pool) < 4:
        raise ValueError("Need at least four calibtest observations.")
    numeric = list(required - {"split"})
    if not np.isfinite(pool[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Held-out prediction data contain nonfinite values.")
    sa = pool.sa.to_numpy(dtype=float)
    indicator = pool["1{Y<=c}"].to_numpy(dtype=float)
    if (sa <= 0).any() or not np.isin(indicator, [0, 1]).all():
        raise ValueError("SA must be positive and the base loss must be binary.")
    probability = pool["pred_1{Y<=c}"].to_numpy(dtype=float)
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predicted false-lead probabilities must be in [0,1].")
    offset = 0.0
    if reward == "diversity":
        observed_reward = 1 - pool.tanimoto.to_numpy(dtype=float)
        if ((observed_reward < 0) | (observed_reward > 1)).any():
            raise ValueError("Tanimoto similarities must be in [0,1].")
        predicted_reward = observed_reward.copy()
    elif reward == "activity":
        train_y = frame.loc[frame.split == "train", "Y"].to_numpy(dtype=float)
        if not len(train_y) or not np.isfinite(train_y).all():
            raise ValueError("Activity reward requires finite training outcomes.")
        offset = float(max(0, 1 - train_y.min()))
        observed_reward = np.maximum(pool.Y.to_numpy(dtype=float) + offset, 0)
        predicted_reward = np.maximum(pool.pred_Y.to_numpy(dtype=float) + offset, 0)
    else:
        observed_reward = predicted_reward = np.ones(len(pool))
    return dict(loss=sa * indicator, loss_prediction=sa * probability, sa=sa,
                reward=observed_reward, reward_prediction=predicted_reward), offset


def score_functions(loss_prediction, denominator_cost, reward_prediction, alpha):
    reward = np.maximum(reward_prediction, REWARD_FLOOR)
    excess = loss_prediction - alpha * denominator_cost
    return dict(loss=loss_prediction, cost=excess,
                reward=loss_prediction / reward, cost_reward=excess / reward)


def select_methods(Lc, Sc, St, cc, ct, sa_test, alpha, uniforms, methods=METHODS,
                   weights=None):
    """Selection sees test scores, observed costs and support, never test labels."""
    if weights is None:
        selected, evalues = SCoRE_WSR((Lc, Sc, cc), (None, St, ct), alpha, alpha,
                                     finite_t=sa_test, return_evals=True)
    else:
        selected, evalues = SCoRE_WSR_w((Lc, Sc, cc), (None, St, ct), *weights,
                                       alpha, alpha, finite_t=sa_test, return_evals=True)
    selections = {}
    for method in methods:
        if method == "score_dtm":
            selections[method] = selected
        else:
            with np.errstate(divide="ignore"):
                boosted = np.divide(evalues, uniforms[method],
                                    out=np.zeros_like(evalues), where=evalues > 0)
            selections[method] = eBH(boosted, alpha, costs=ct)
    return selections


def evaluate(loss, cost, reward, selected):
    nsel = len(selected)
    return dict(wsr=float(loss[selected].sum() / cost[selected].sum()) if nsel else 0.0,
                fraction_selected=nsel / len(loss),
                avg_reward=float(reward[selected].sum() / len(loss)),
                n_selected=nsel, selected_cost=float(cost[selected].sum()))


def run_replication(data, seed, count_alphas=COUNT_ALPHAS, cost_alphas=COST_ALPHAS,
                    scores=SCORES, methods=("score_homo",)):
    """Pair samples and boosting draws across scores and normalizations."""
    order = np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(len(data["sa"]))
    calib, test = np.split(order, [len(order) // 2])
    return evaluate_split(data, calib, test, seed, count_alphas, cost_alphas, scores, methods)


def evaluate_split(data, calib, test, seed, count_alphas=COUNT_ALPHAS, cost_alphas=COST_ALPHAS,
                   scores=SCORES, methods=("score_homo",), weights=None):
    """Common comparison for exchangeable or shifted samples; costs differ from weights."""
    cdata = {name: values[calib] for name, values in data.items()}
    tdata = {name: values[test] for name, values in data.items()}
    rows = []
    for normalization, levels in (("count", count_alphas), ("cost", cost_alphas)):
        cc = cdata["sa"] if normalization == "cost" else np.ones(len(calib))
        ct = tdata["sa"] if normalization == "cost" else np.ones(len(test))
        for alpha in levels:
            # Independent streams keyed by alpha, shared across competing scores.
            key = list(np.float64(alpha).tobytes())
            uniforms = {
                "score_hete": np.random.default_rng(np.random.SeedSequence([seed, 1, *key])).uniform(size=len(test)),
                "score_homo": np.random.default_rng(np.random.SeedSequence([seed, 2, *key])).uniform(),
            }
            sc = score_functions(cdata["loss_prediction"], cc, cdata["reward_prediction"], alpha)
            st = score_functions(tdata["loss_prediction"], ct, tdata["reward_prediction"], alpha)
            for score in scores:
                selections = select_methods(cdata["loss"], sc[score], st[score], cc, ct,
                                              tdata["sa"], alpha, uniforms, methods, weights=weights)
                for method, selected in selections.items():
                    rows.append(dict(seed=seed, normalization=normalization, alpha=float(alpha),
                                     gamma=float(alpha), score=score, method=method,
                                     ncalib=len(calib), ntest=len(test),
                                     **evaluate(tdata["loss"], ct, tdata["reward"], selected)))
    return rows


def argument_parser(description=__doc__, output_help="Output directory (default: results_wsr/<dataset>/<reward>)"):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="caco2_wang", help="Dataset name in predicted_results")
    parser.add_argument("--model", default="DGL_AttentiveFP")
    parser.add_argument("--prediction-seed", type=int, default=1)
    parser.add_argument("--predictions", type=Path, help="Explicit prediction CSV path")
    parser.add_argument("--reward", choices=("diversity", "activity", "constant"), default="diversity")
    parser.add_argument("--scores", nargs="+", choices=SCORES, default=list(SCORES))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["score_homo"])
    parser.add_argument("--count-alphas", nargs="+", type=float, default=COUNT_ALPHAS.tolist())
    parser.add_argument("--cost-alphas", nargs="+", type=float, default=COST_ALPHAS.tolist())
    parser.add_argument("--nrep", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, help=output_help)
    return parser


def parse_args(argv=None, parser=None, output_root="results_wsr"):
    parser = argument_parser() if parser is None else parser
    args = parser.parse_args(argv)
    if args.nrep < 1 or args.seed_start < 0 or args.prediction_seed < 0:
        parser.error("nrep must be positive and seeds nonnegative.")
    for name in ("count_alphas", "cost_alphas"):
        levels = getattr(args, name)
        if any(not np.isfinite(a) or a <= 0 for a in levels):
            parser.error("Target levels must be finite and positive.")
    if any(a > 1 for a in args.cost_alphas):
        parser.error("Cost-normalized target levels must be at most 1.")
    for name in ("count_alphas", "cost_alphas", "scores", "methods"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"{name.replace('_', '-')} must not contain duplicates.")
    args.model = args.model.removeprefix("DGL_")
    if args.model == "rdkit_2d_normalized":
        args.model = "rdkit_2d"
    if args.predictions is None:
        args.predictions = HERE / "predicted_results" / args.data / args.model / f"{args.data}_random_seed{args.prediction_seed}_pred.csv"
    if args.output is None:
        args.output = HERE / output_root / args.data / args.reward
    return args


def main(argv=None):
    args = parse_args(argv)
    data, offset = prepare_data(pd.read_csv(args.predictions), args.reward)
    rows = []
    for seed in tqdm(range(args.seed_start, args.seed_start + args.nrep), desc=f"Drug WSR: {args.data}"):
        rows.extend(run_replication(data, seed, args.count_alphas, args.cost_alphas, args.scores, args.methods))
    frame = pd.DataFrame(rows)
    metadata = dict(dataset=args.data, model=args.model, reward=args.reward, activity_offset=offset,
                    reward_floor=REWARD_FLOOR, shift="exchangeable", loss="SA_times_false_lead",
                    cost_definition="observed_SA", support="finite_SA", prediction_seed=args.prediction_seed,
                    predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest())
    for name, value in metadata.items():
        frame[name] = value
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"drug_wsr_seeds_{args.seed_start}_{args.seed_start + args.nrep - 1}.csv"
    frame.to_csv(path, index=False)
    print(f"Saved {len(frame)} raw replication rows to {path.resolve()}")


if __name__ == "__main__":
    main()
