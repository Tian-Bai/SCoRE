"""SCoRE versus Hoeffding/Rademacher threshold certification on the six WSR DGPs.

Separate experiment: no binarization or reward, gamma=alpha. By default,
calibrate beta and delta to guarantee expected finite-batch WSR at most alpha.
Use --calibration pac for the previous population-ratio certificate.
Only raw replication CSVs are saved; plot with plot_wsr_concentration.R.
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
from simulation.wsr import SETTINGS, COST_DESIGNS, CLIP_LOOKUP, make_data
from simulation.wsr import compare_methods as compare_wsr_methods


METHODS = ("score_dtm", "score_hete", "score_homo", "hoeffding", "rademacher")


def threshold_means(values, scores, grid):
    """Empirical E[values * 1{score <= t}], including all tied scores."""
    order = np.argsort(scores, kind="stable")
    cumulative = np.r_[0., np.cumsum(values[order])]
    return cumulative[np.searchsorted(scores[order], grid, side="right")]/len(scores)


def rademacher_maxima(values, scores, signs):
    """One supremum per sign vector; thresholds cannot split ties.

    Include the zero function (select nobody). This covers all real cutoffs,
    so the resulting bound permits the calibration-score candidate grid.
    """
    order = np.argsort(scores, kind="stable")
    ends = np.r_[np.diff(scores[order]) != 0, True]
    prefixes = np.cumsum(signs[:, order]*values[order], axis=1)
    return np.maximum(0., np.max(prefixes[:, ends], axis=1))/len(scores)


def rademacher_upper(values, scores, rng, draws, failure):
    """MC upper bound on empirical Rademacher complexity, conditional on data.

    Changing one sign changes the average of draws suprema by at most
    2*values[i]/(draws*n). McDiarmid gives the correction below.
    This accounts for MC error separately from calibration-sample error.
    """
    total = 0.
    for start in range(0, draws, 64):
        signs = 2*rng.integers(0, 2, size=(min(64, draws-start), len(scores)))-1
        total += rademacher_maxima(values, scores, signs).sum()
    estimate = total/draws
    correction = np.sqrt(2*np.sum(values**2)*np.log(1/failure)/(draws*len(scores)**2))
    return estimate+correction


def concentration_bounds(Lc, Sc, cc, cost_bound, delta=.05, grid_size=100,
                         rad_draws=100, seed=0, setting=1, denominator="cost"):
    """Return candidate grids and uniform numerator/denominator bounds.

    cost_bound is a known deterministic bound (1 or 4), never a sample max.
    Both L and c are bounded by it. Dividing both by cost_bound would give
    exactly the same ratio bounds as scaling the penalties here.
    """
    Lc, Sc, cc = (np.asarray(x, dtype=float) for x in (Lc, Sc, cc))
    if Lc.ndim != 1 or not len(Lc) or Lc.shape != Sc.shape or Lc.shape != cc.shape:
        raise ValueError("Losses, scores, and costs must be equally sized nonempty vectors.")
    if not all(np.all(np.isfinite(x)) for x in (Lc, Sc, cc)):
        raise ValueError("Calibration inputs must be finite.")
    if not np.isfinite(cost_bound) or cost_bound <= 0 or np.any(cc <= 0) or np.any(cc > cost_bound) or np.any(Lc < 0) or np.any(Lc > cc):
        raise ValueError("Require 0 <= L <= c <= the known cost bound, with c > 0.")
    if not 0 < delta < 1 or grid_size < 2 or rad_draws < 1:
        raise ValueError("Require delta in (0,1), at least two grid points and positive rad_draws.")
    if np.any(Sc < 0) or np.any(Sc > cost_bound):
        raise ValueError("Loss-prediction scores must lie in [0, cost_bound].")
    if denominator not in ("cost", "count"):
        raise ValueError("denominator must be cost or count.")
    den_values = cc if denominator == "cost" else np.ones(len(cc))
    den_bound = cost_bound if denominator == "cost" else 1.
    n = len(Lc)
    grid_h = np.linspace(0., cost_bound, grid_size)
    eps_h = np.sqrt(np.log(4*grid_size/delta)/(2*n))
    # Each of the two empirical-process bounds and two MC bounds gets delta/4.
    rng = np.random.default_rng(np.random.SeedSequence([seed, setting, 2718]))
    rad_num = rademacher_upper(Lc, Sc, rng, rad_draws, delta/4)
    rad_den = rademacher_upper(den_values, Sc, rng, rad_draws, delta/4)
    eps_r = 3*np.sqrt(np.log(8/delta)/(2*n))
    grid_r = np.unique(Sc)
    return {
        "hoeffding": (grid_h, threshold_means(Lc, Sc, grid_h)+cost_bound*eps_h,
                      threshold_means(den_values, Sc, grid_h)-den_bound*eps_h),
        "rademacher": (grid_r, threshold_means(Lc, Sc, grid_r)+2*rad_num+cost_bound*eps_r,
                       threshold_means(den_values, Sc, grid_r)-2*rad_den-den_bound*eps_r),
    }


def certified_cutoff(grid, upper_num, lower_den, alpha):
    feasible = (lower_den > 0) & (upper_num <= alpha*lower_den)
    if not np.any(feasible):
        return -np.inf, np.nan
    index = np.flatnonzero(feasible)[-1]
    return float(grid[index]), float(upper_num[index]/lower_den[index])


def expectation_levels(alpha, delta_fraction=.1, expectation_margin=0., risk_bound=1.):
    """A data-independent PAC-to-expectation allocation with optional slack.

    Delta consumes delta_fraction of alpha; the remaining budget goes to beta.
    The bound is (1-delta)*beta+delta*risk_bound=(1-margin)*alpha.
    risk_bound bounds the realized normalized risk, not individual raw losses.
    """
    if not np.isfinite(risk_bound) or not 0 < alpha < risk_bound:
        raise ValueError("Require 0 < alpha < finite risk_bound.")
    if not 0 <= expectation_margin < 1 or not 0 < delta_fraction < 1-expectation_margin:
        raise ValueError("Require 0 <= expectation_margin < 1 and 0 < delta_fraction < 1-expectation_margin.")
    delta = delta_fraction*alpha/risk_bound
    budget = (1-expectation_margin)*alpha
    beta = (budget-delta*risk_bound)/(1-delta)
    return beta, delta, budget


def run_replication(setting, seed, alphas, delta=.05, grid_size=100,
                    rad_draws=100, calibration="expected", delta_fraction=.1,
                    expectation_margin=0., **data_options):
    source, cost_mode = SETTINGS[setting]
    Lc, Sc, Lt, St, cc, ct = make_data(source, seed, cost_mode=cost_mode, **data_options)
    cost_bound = 1. if cost_mode == "unit" else 4.
    if calibration not in ("expected", "pac"):
        raise ValueError("calibration must be expected or pac.")
    # For any nonempty test selection, WSR <= average selected L because c>=1.
    # Certifying E[L | selected] therefore bounds expected finite-batch WSR.
    if calibration == "expected" and (np.any(cc < 1) or np.any(ct < 1)):
        raise ValueError("Expected-WSR conversion requires the design's known cost lower bound 1.")
    denominator = "count" if calibration == "expected" else "cost"
    bounds = None
    costs = {} if cost_mode == "unit" else dict(cc=cc, ct=ct)
    rows = []
    for alpha in alphas:
        beta, failure, budget = (expectation_levels(alpha, delta_fraction, expectation_margin)
                                 if calibration == "expected" else (alpha, delta, np.nan))
        if calibration == "expected" or bounds is None:
            bounds = concentration_bounds(Lc, Sc, cc, cost_bound, failure, grid_size,
                                          rad_draws, seed, setting, denominator)
        score_selections = compare_wsr_methods(Lc, Sc, St, alpha, seed, setting, **costs)
        for method in METHODS:
            cutoff, certified_risk = np.nan, np.nan
            if method.startswith("score_"):
                selected = score_selections[method]
            else:
                cutoff, certified_risk = certified_cutoff(*bounds[method], beta)
                selected = np.flatnonzero(St <= cutoff)
            count = len(selected)
            wsr = float(Lt[selected].sum()/ct[selected].sum()) if count else 0.
            rows.append(dict(seed=seed, alpha=alpha, method=method,
                             gamma=alpha if method.startswith("score_") else np.nan,
                             beta=beta if method in bounds else np.nan,
                             delta=failure if method in bounds else np.nan,
                             expected_risk_bound=budget if method in bounds else np.nan,
                             wsr=wsr, n_selected=count, fraction_selected=count/len(Lt),
                             nonempty=int(count > 0), violation=int(wsr > alpha),
                             cutoff=cutoff, certified_risk=certified_risk))
    return rows


def _run_task(task):
    setting, seed, alphas, baseline_options, options = task
    return run_replication(setting, seed, alphas, **baseline_options, **options)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", type=int, choices=SETTINGS, default=list(SETTINGS))
    for name, default in (("nrep", 100), ("ntrain", 1000), ("ncalib", 1000),
                          ("ntest", 100), ("trees", 100), ("jobs", 1),
                          ("grid-size", 100), ("rad-draws", 100)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=.1)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--tau", type=float, default=10.)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--cost-features", action="store_true")
    parser.add_argument("--alphas", nargs="+", type=float, default=[i/20 for i in range(1, 11)])
    parser.add_argument("--calibration", choices=("expected", "pac"), default="expected")
    parser.add_argument("--delta", type=float, default=.05, help="Failure probability in pac mode only")
    parser.add_argument("--delta-fraction", type=float, default=.1,
                        help="Expected mode: delta = delta_fraction * alpha (risk bound is 1)")
    parser.add_argument("--expectation-margin", type=float, default=0.,
                        help="Expected mode: expectation bound = (1-margin)*alpha")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"results_wsr_concentration")
    args = parser.parse_args(argv)
    if any(getattr(args, k) <= 0 for k in ("nrep", "ntrain", "ncalib", "ntest", "trees", "jobs", "rad_draws")) or args.grid_size < 2:
        parser.error("Sample sizes, workers, trees, and draws must be positive; grid-size >= 2.")
    if args.seed_start < 0 or args.dim < 4 or (4 in args.settings and args.dim < 10):
        parser.error("Require seed-start >= 0, dim >= 4 (>= 10 for independent costs).")
    if not np.isfinite(args.sigma) or args.sigma < 0 or not np.isfinite(args.tau) or args.tau <= 0:
        parser.error("Require finite sigma >= 0 and finite tau > 0.")
    if not 0 < args.delta < 1 or any(not 0 < a < 1 for a in args.alphas):
        parser.error("delta and alphas must lie strictly between 0 and 1.")
    try:
        expectation_levels(args.alphas[0], args.delta_fraction, args.expectation_margin)
    except ValueError as error:
        parser.error(str(error))
    if len(set(args.settings)) != len(args.settings) or len(set(args.alphas)) != len(args.alphas):
        parser.error("Duplicate settings or alphas.")
    if args.clip is not None and (not np.isfinite(args.clip) or args.clip <= 0):
        parser.error("clip must be finite and positive.")
    if 2 in args.settings and args.clip is None and (2, args.sigma) not in CLIP_LOOKUP:
        parser.error("Specify --clip for squared-error settings at this sigma.")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    options = {key: getattr(args, key) for key in
               ("ntrain", "ncalib", "ntest", "sigma", "dim", "tau", "trees", "clip", "cost_features")}
    baseline_options = {key: getattr(args, key) for key in
                        ("delta", "grid_size", "rad_draws", "calibration", "delta_fraction", "expectation_margin")}
    context = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else nullcontext(None)
    with context as executor:
        for setting in args.settings:
            source, cost_mode = SETTINGS[setting]
            tasks = [(setting, seed, args.alphas, baseline_options, options)
                     for seed in range(args.seed_start, args.seed_start+args.nrep)]
            results = executor.map(_run_task, tasks) if executor else map(_run_task, tasks)
            rows = []
            for result in tqdm(results, total=args.nrep, desc=f"Concentration setting {setting}"):
                rows.extend(result)
            frame = pd.DataFrame(rows)
            case, dgp = (source-1)//2+1, (source-1)%2+1
            metadata = dict(options, setting=setting, source_setting=source, case=case, dgp=dgp,
                            target="WSR", cost_mode=cost_mode, cost_design=COST_DESIGNS[cost_mode],
                            score="loss", pac_delta=args.delta, grid_size=args.grid_size,
                            calibration=args.calibration, delta_fraction=args.delta_fraction,
                            expectation_margin=args.expectation_margin, risk_bound=1., cost_lower_bound=1.,
                            certificate="conditional_loss" if args.calibration == "expected" else "population_cost_ratio",
                            rad_draws=args.rad_draws, rad_correction="mcdiarmid_split_v1",
                            cost_bound=1. if cost_mode == "unit" else 4.)
            metadata["clip"] = (args.clip if args.clip is not None else CLIP_LOOKUP[(dgp, args.sigma)]) if case == 2 else np.nan
            for key, value in metadata.items():
                frame[key] = value
            stem = f"setting_{setting}_seeds_{args.seed_start}_{args.seed_start+args.nrep-1}"
            frame.to_csv(args.output/f"{stem}.csv", index=False)
    print(f"Saved raw concentration comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
