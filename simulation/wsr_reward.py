"""Compare four SCoRE scores with reward Y**2 across the six WSR settings.

Uses homogeneous boosting, gamma=alpha, and paired data/model/random draws.
Reuses read-only settings and CLI utilities from wsr.py without changing it.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SCoRE import SCoRE_WSR, gen_data_1, gen_data_2, gen_data_Jin2023, loss_1, loss_2, loss_Jin2023

if __package__:
    from .wsr import (SETTINGS, CLIP_LOOKUP, COST_DESIGNS, unit_costs, prediction_features,
                      with_cost_design, parse_args as parse_wsr_args)
else:
    from wsr import (SETTINGS, CLIP_LOOKUP, COST_DESIGNS, unit_costs, prediction_features,
                     with_cost_design, parse_args as parse_wsr_args)


SCORES = ("loss", "cost", "reward", "cost_reward")
REWARD_FLOOR = 1e-8
CONFIG_COLUMNS = ["setting", "source_setting", "case", "dgp", "cost_mode", "cost_design", "cost_features",
                  "sigma", "dim", "tau", "clip", "ntrain", "ncalib", "ntest",
                  "trees", "reward", "reward_floor", "boosting"]
METRICS = ("wsr", "percent_selected", "avg_reward")


def make_reward_data(setting, seed, ntrain=1000, ncalib=1000, ntest=100,
                     sigma=0.1, dim=20, tau=10.0, trees=100, clip=None, cost_features=False):
    """Return observed/predicted losses, costs, and rewards on independent splits.

    The first six random streams match wsr.py; only the reward model is new.
    No test labels are used in either fitted model or any score.
    """
    source, mode = SETTINGS[setting]
    case, dgp = (source - 1) // 2 + 1, (source - 1) % 2 + 1
    generator = {1: gen_data_1, 2: gen_data_2, 3: gen_data_Jin2023}[case]
    streams = np.random.SeedSequence([seed, source, 0]).spawn(7)

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
    Xc, Yc = draw(ncalib, streams[3])
    Xt, Yt = draw(ntest, streams[4])
    features = lambda X: prediction_features(X, mode, cost_features)
    Ftrain, Fc, Ft = features(Xtrain), features(Xc), features(Xt)
    loss_model = forest(streams[5]).fit(Ftrain, loss(Xtrain, Ytrain))
    reward_model = forest(streams[6]).fit(Ftrain, Ytrain ** 2)
    cc, ct = unit_costs(Xc, mode), unit_costs(Xt, mode)
    return dict(Lc=cc * loss(Xc, Yc), Lt=ct * loss(Xt, Yt), cc=cc, ct=ct,
                lc=cc * loss_model.predict(Fc), lt=ct * loss_model.predict(Ft),
                rc=reward_model.predict(Fc), rt=reward_model.predict(Ft), reward=Yt ** 2)


def score_functions(loss_prediction, cost, reward_prediction, alpha):
    """Cost means subtracting alpha*c; all variants use the same cost-scaled loss."""
    reward = np.maximum(reward_prediction, REWARD_FLOOR)
    excess = loss_prediction - alpha * cost
    return dict(loss=loss_prediction, cost=excess,
                reward=loss_prediction / reward, cost_reward=excess / reward)


def compare_scores(Lc, lc, lt, cc, ct, rc, rt, alpha, seed, setting):
    """Select with four scores; takes no realized test loss or reward."""
    calibration_scores = score_functions(lc, cc, rc, alpha)
    test_scores = score_functions(lt, ct, rt, alpha)
    # Reset the generator to the same seed for each score, so all four receive
    # exactly the same homogeneous uniform draw. Alpha order is irrelevant.
    key = [seed, setting, 92, *np.float64(alpha).tobytes()]
    return {
        name: SCoRE_WSR((Lc, calibration_scores[name], cc), (None, test_scores[name], ct),
                         alpha, alpha, prune="homo", Lbar_t=ct,
                         random_state=np.random.default_rng(np.random.SeedSequence(key)))
        for name in SCORES
    }


def evaluate(Lt, ct, reward, selected):
    """Average reward is sum of selected rewards / m, not / number selected."""
    return dict(wsr=float(Lt[selected].sum() / ct[selected].sum()) if len(selected) else 0.0,
                percent_selected=100 * len(selected) / len(Lt),
                avg_reward=float(reward[selected].sum() / len(Lt)), n_selected=len(selected))


def run_replication(setting, seed, alphas, **options):
    data = make_reward_data(setting, seed, **options)
    selection_data = {key: data[key] for key in ("Lc", "lc", "lt", "cc", "ct", "rc", "rt")}
    rows = []
    for alpha in alphas:
        for name, selected in compare_scores(**selection_data, alpha=alpha, seed=seed, setting=setting).items():
            rows.append(dict(seed=seed, alpha=alpha, gamma=alpha, score=name,
                             **evaluate(data["Lt"], data["ct"], data["reward"], selected)))
    return rows


def summarize(raw):
    raw = with_cost_design(raw)
    grouped = raw.groupby(CONFIG_COLUMNS + ["alpha", "score"], dropna=False, sort=True)
    specs = {"nrep": ("seed", "size")}
    for metric in METRICS:
        specs[f"{metric}_mean"] = (metric, "mean")
        specs[f"{metric}_sd"] = (metric, "std")
    summary = grouped.agg(**specs).reset_index()
    for metric in METRICS:
        summary[f"{metric}_se"] = summary[f"{metric}_sd"] / np.sqrt(summary.nrep)
    return summary


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print("Options below are shared with wsr.py; the reward output defaults to results_wsr_reward.")
    if not any(arg == "--output" or arg.startswith("--output=") for arg in argv):
        argv += ["--output", str(Path(__file__).resolve().parent / "results_wsr_reward")]
    return parse_wsr_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    options = {key: getattr(args, key) for key in
               ("ntrain", "ncalib", "ntest", "sigma", "dim", "tau", "trees", "clip", "cost_features")}
    for setting in args.settings:
        source, mode = SETTINGS[setting]
        case, dgp = (source - 1) // 2 + 1, (source - 1) % 2 + 1
        rows = []
        for seed in tqdm(range(args.seed_start, args.seed_start + args.nrep), desc=f"Reward setting {setting}"):
            rows.extend(run_replication(setting, seed, args.alphas, **options))
        frame = pd.DataFrame(rows)
        metadata = dict(options, setting=setting, source_setting=source, case=case, dgp=dgp,
                        cost_mode=mode, cost_design=COST_DESIGNS[mode],
                        reward="Y_squared", reward_floor=REWARD_FLOOR, boosting="homo")
        metadata["clip"] = (args.clip if args.clip is not None else CLIP_LOOKUP[(dgp, args.sigma)]) if case == 2 else np.nan
        for key, value in metadata.items():
            frame[key] = value
        frame.to_csv(args.output / f"reward_setting_{setting}_seeds_{args.seed_start}_{args.seed_start + args.nrep - 1}.csv",
                     index=False)
    print(f"Saved reward results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
