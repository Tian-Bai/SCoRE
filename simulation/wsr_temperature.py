"""Sigmoid-temperature sweep for WSR tightness, with unit or independent costs.

Both panels use the main-text sigmoid DGP (source setting 6), with
tau=1,2,5,10,30, n=100,300,1000, fixed m=100 and alpha=gamma=0.2.
Compare the three SCoRE boosting methods and two binarized baselines.
Each temperature refits the loss forest on the same training X,Y. Calibration
and test draws and boosting uniforms are shared across temperatures. Smaller
calibration samples are prefixes of the largest pool. Finite temperatures
always use interval support (0,c(X)], even if base losses round to 0/1.

    python simulation/wsr_temperature.py --jobs 4
    Rscript simulation/plot_wsr_temperature.R

Only raw replication CSVs are saved; the plotter writes PDF and LaTeX.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.wsr import COST_DESIGNS, METHODS, compare_methods, make_data


# Match the panel IDs in the main WSR comparison.
SETTINGS = {3: (6, "unit"), 4: (6, "unrelated")}
TEMPERATURES = (1., 2., 5., 10., 30.)
CALIB_SIZES = (100, 300, 1000)


def run_replication(setting, seed, alpha=.2, temperatures=TEMPERATURES,
                    calib_sizes=CALIB_SIZES, ntest=100, **data_options):
    """Return a paired sweep; test losses enter evaluation only."""
    nmax = max(calib_sizes)
    source_setting, cost_mode = SETTINGS[setting]
    rows = []
    for tau in temperatures:
        # make_data seeds the underlying X,Y and forest independently of tau.
        # Keeping pool sizes fixed pairs observations across the entire sweep.
        Lc, Sc, Lt, St, cc, ct = make_data(source_setting, seed, tau=tau, ncalib=nmax,
                                        ntest=ntest, cost_mode=cost_mode,
                                        cost_features=False, **data_options)
        for n in calib_sizes:
            costs = {} if cost_mode == "unit" else dict(cc=cc[:n], ct=ct)
            # The source ID also pairs boosting draws between cost cases.
            selections = compare_methods(Lc[:n], Sc[:n], St, alpha, seed, source_setting, **costs)
            for method, selected in selections.items():
                wsr = float(Lt[selected].sum()/ct[selected].sum()) if len(selected) else 0.
                u = alpha/2 if method == "binarized_half" else alpha/4 if method == "binarized_quarter" else np.nan
                level = alpha if np.isnan(u) else (alpha-u)/(1-u)
                rows.append(dict(setting=setting, seed=seed, tau=float(tau),
                                 ncalib=n, ntest=ntest, alpha=alpha,
                                 gamma=alpha if method.startswith("score_") else level if costs else np.nan,
                                 method=method, u=u, testing_level=level, wsr=wsr,
                                 n_selected=len(selected), fraction_selected=len(selected)/ntest,
                                 nonempty=int(len(selected)>0), violation=int(wsr>alpha)))
    return rows


def _run_task(task):
    setting, seed, alpha, temperatures, calib_sizes, ntest, options = task
    return run_replication(setting, seed, alpha, temperatures, calib_sizes, ntest, **options)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", nargs="+", type=int, choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--temperatures", nargs="+", type=float, default=list(TEMPERATURES))
    parser.add_argument("--calib-sizes", nargs="+", type=int, default=list(CALIB_SIZES))
    parser.add_argument("--alpha", type=float, default=.2)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=.1)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--nrep", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1,
                        help="Independent replication workers; results do not depend on worker count")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent/"results_wsr_temperature")
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 1:
        parser.error("alpha must lie strictly between 0 and 1.")
    if any(not np.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error("temperatures must be finite and positive.")
    if any(getattr(args, key) <= 0 for key in ("ntest", "ntrain", "trees", "nrep", "jobs")) or any(n<=0 for n in args.calib_sizes):
        parser.error("Sample sizes, trees, nrep, and jobs must be positive.")
    if args.seed_start < 0 or args.dim < 4:
        parser.error("seed-start must be nonnegative and dim at least 4.")
    if 4 in args.settings and args.dim < 10:
        parser.error("Independent costs require dim at least 10.")
    if not np.isfinite(args.sigma) or args.sigma < 0:
        parser.error("sigma must be finite and nonnegative.")
    for key in ("settings", "temperatures", "calib_sizes"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            parser.error(f"{key.replace('_', '-')} must not contain duplicates.")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    options = {key: getattr(args, key) for key in ("ntrain", "trees", "sigma", "dim")}
    context = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs>1 else nullcontext(None)
    with context as executor:
        for setting in args.settings:
            source_setting, cost_mode = SETTINGS[setting]
            tasks = [(setting, seed, args.alpha, args.temperatures, args.calib_sizes,
                      args.ntest, options) for seed in range(args.seed_start, args.seed_start+args.nrep)]
            results = executor.map(_run_task, tasks) if executor else map(_run_task, tasks)
            rows = []
            for result in tqdm(results, total=args.nrep, desc=f"Temperature setting {setting}"):
                rows.extend(result)
            frame = pd.DataFrame(rows)
            metadata = dict(options, source_setting=source_setting, case=3, dgp=2,
                            pool_ncalib=max(args.calib_sizes), target="WSR", cost_mode=cost_mode,
                            cost_design=COST_DESIGNS[cost_mode], cost_features=False, score="loss",
                            loss_support="interval", training="refit_per_temperature",
                            pairing="same_xy_nested_calibration_v1")
            for key, value in metadata.items():
                frame[key] = value
            for n, group in frame.groupby("ncalib", sort=False):
                stem = f"temperature_setting_{setting}_n{n}_m{args.ntest}_seeds_{args.seed_start}_{args.seed_start+args.nrep-1}"
                group.to_csv(args.output/f"{stem}.csv", index=False)
    print(f"Saved raw temperature-sweep replications to {args.output.resolve()}")


if __name__ == "__main__":
    main()
