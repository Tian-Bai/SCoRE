"""Marginal-risk comparison on the same six DGPs as simulation/wsr.py.

MSR is expected total selected loss divided by the full test size m.
Settings 4--6 retain L=c(X)*B and the known bound c(X); costs do not
normalize MSR. All methods use the same independently trained loss score.

With gamma=alpha, every positive marginal SCoRE e-value is at least 1/alpha,
so homogeneous and heterogeneous boosting give the deterministic selection.
We therefore compare one SCoRE curve and binarization at u=alpha/2, alpha/4.
For each binarized baseline, L <= u + (c-u)*1{L>u}. Control the marginal
risk of the residual (c-u)*1{L>u} at alpha-u. In unit-cost settings this is
ordinary conformal testing at (alpha-u)/(1-u), without BH.

Only raw replication CSVs are saved. Plot with simulation/plot_msr.R.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SCoRE import SCoRE_MSR
from simulation.wsr import SETTINGS, COST_DESIGNS, CLIP_LOOKUP, make_data


METHODS = ("score_dtm", "binarized_half", "binarized_quarter")


def compare_methods(Lc, Sc, St, alpha, cc=None, ct=None):
    """Selections depend on calibration outcomes and test scores/bounds only."""
    cc = np.ones(len(Lc)) if cc is None else np.asarray(cc)
    ct = np.ones(len(St)) if ct is None else np.asarray(ct)
    selections = {"score_dtm": SCoRE_MSR((Lc, Sc), (None, St),
                                        alpha, alpha, Lbar_t=ct)}
    for method, divisor in (("binarized_half", 2), ("binarized_quarter", 4)):
        u = alpha / divisor
        # All six designs have c >= 1 and the alpha grid lies in (0,1),
        # so c-u is strictly positive. This is an absolute loss threshold.
        if np.any(cc <= u) or np.any(ct <= u):
            raise ValueError("Binarization requires all loss bounds to exceed u.")
        residual = (cc-u) * (Lc > u)
        level = alpha-u
        selections[method] = SCoRE_MSR((residual, Sc), (None, St),
                                        level, level, finite_t=ct-u)
    return selections


def run_replication(setting, seed, alphas, **data_options):
    source_setting, cost_mode = SETTINGS[setting]
    Lc, Sc, Lt, St, cc, ct = make_data(source_setting, seed,
                                      cost_mode=cost_mode, **data_options)
    rows = []
    for alpha in alphas:
        for method, selected in compare_methods(Lc, Sc, St, alpha, cc, ct).items():
            msr = float(np.sum(Lt[selected]) / len(Lt))
            u = alpha/2 if method == "binarized_half" else alpha/4 if method == "binarized_quarter" else np.nan
            level = alpha if np.isnan(u) else alpha-u
            rows.append(dict(seed=seed, alpha=alpha, method=method, gamma=level,
                             u=u, testing_level=level, msr=msr,
                             n_selected=len(selected), fraction_selected=len(selected)/len(Lt),
                             nonempty=int(len(selected)>0), violation=int(msr>alpha)))
    return rows


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
                        help="Include known c(X) alongside X in the loss predictor")
    parser.add_argument("--alphas", nargs="+", type=float, default=[i / 20 for i in range(1, 11)])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results_msr")
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
                        target="MSR", risk_denominator="ntest",
                        binarization="absolute_loss_residual", cost_mode=cost_mode,
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
