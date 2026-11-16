"""Fixed-alpha gamma sweep for the excess-loss and squared-error settings.

Defaults: alpha=0.2, 100 replications, and (n,m)=(500,50),(500,200),(2000,200).
Gamma ranges from 0 to 1.5*alpha, with a denser grid near alpha.
Unit costs and reward 1 make average reward equal to fraction selected.
Models, nested data pools, and boosting draws are shared across the sweep.
Only raw replication CSVs are saved. For example:

    python simulation/wsr_gamma.py
    Rscript simulation/plot_wsr_gamma.R
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SCoRE import SCoRE_WSR, eBH

if __package__:
    from .wsr import CLIP_LOOKUP, SETTINGS as BASE_SETTINGS, make_data
else:
    from wsr import CLIP_LOOKUP, SETTINGS as BASE_SETTINGS, make_data


# Use the same two unit-cost DGP/loss combinations as the method comparison.
SETTINGS = {setting: BASE_SETTINGS[setting][0] for setting in (1, 2)}
METHODS = ("score_dtm", "score_hete", "score_homo")
SAMPLE_SIZES = ((500, 50), (500, 200), (2000, 200))
# Integer ticks avoid duplicate floating-point grid points. Resolve the
# boundary at 1, with additional points just below and above it.
GAMMA_RATIOS = np.array(sorted(set(range(0, 151, 10)) |
                               set(range(80, 121, 2)) | {95, 99, 101, 105})) / 100


def select_at_gamma(Lc, Sc, St, alpha, gamma, uniform_hete, uniform_homo):
    """Compute e-values once at gamma; apply every selection rule at alpha.

    Test losses are deliberately excluded. Gamma=0 is passed to the core
    unchanged, and gamma>alpha is evaluated rather than forced to select none.
    """
    selected, evalues = SCoRE_WSR((Lc, Sc), (None, St), alpha, gamma,
                                 Lbar_t=1, return_evals=True)
    selections = {"score_dtm": selected}
    for method, uniform in (("score_hete", uniform_hete), ("score_homo", uniform_homo)):
        with np.errstate(divide="ignore"):
            boosted = np.divide(evalues, uniform, out=np.zeros_like(evalues), where=evalues > 0)
        selections[method] = eBH(boosted, alpha)
    return selections


def run_replication(setting, seed, alpha, gamma_ratios=GAMMA_RATIOS,
                    sample_sizes=SAMPLE_SIZES, **data_options):
    """Fit once, then evaluate a paired gamma/sample-size sweep.

    Smaller configurations use prefixes of the largest calibration and test
    pools. The pool sizes are recorded in output metadata: changing their
    maxima changes the paired experiment, even for an unchanged smaller pair.
    """
    nmax = max(n for n, _ in sample_sizes)
    mmax = max(m for _, m in sample_sizes)
    Lc, Sc, Lt, St, _, _ = make_data(SETTINGS[setting], seed,
                                    ncalib=nmax, ntest=mmax, cost_mode="unit",
                                    cost_features=False, **data_options)
    level_key = list(np.float64(alpha).tobytes())
    # These streams match wsr.py at gamma=alpha. Draw once for the entire
    # sweep so grid order or extra gamma values cannot change existing points.
    hete_rng = np.random.default_rng(np.random.SeedSequence([seed, setting, 1, *level_key]))
    homo_rng = np.random.default_rng(np.random.SeedSequence([seed, setting, 2, *level_key]))
    uniform_hete, uniform_homo = hete_rng.uniform(size=mmax), homo_rng.uniform()
    rows = []
    for n, m in sample_sizes:
        for ratio in gamma_ratios:
            gamma = float(alpha * ratio)
            selections = select_at_gamma(Lc[:n], Sc[:n], St[:m], alpha, gamma,
                                          uniform_hete[:m], uniform_homo)
            for method, selected in selections.items():
                n_selected = len(selected)
                wsr = float(Lt[:m][selected].mean()) if n_selected else 0.0
                fraction = n_selected / m
                rows.append(dict(seed=seed, setting=setting, ncalib=n, ntest=m,
                                 alpha=alpha, gamma=gamma, gamma_ratio=float(ratio),
                                 method=method, wsr=wsr, n_selected=n_selected,
                                 fraction_selected=fraction, avg_reward=fraction,
                                 nonempty=int(n_selected > 0), violation=int(wsr > alpha)))
    return rows


def sample_size_pair(value):
    try:
        n, m = map(int, value.split("/"))
        if n <= 0 or m <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("Use positive calibration/test sizes, e.g. 500/50.")
    return n, m


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", nargs="+", type=int, choices=SETTINGS, default=[1, 2])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma-ratios", nargs="+", type=float, default=GAMMA_RATIOS.tolist(),
                        help="Values of gamma/alpha in [0,1.5]; defaults to a grid dense near 1")
    parser.add_argument("--sample-sizes", nargs="+", type=sample_size_pair, default=list(SAMPLE_SIZES),
                        metavar="N/M", help="Calibration/test sizes (default: 500/50 500/200 2000/200)")
    parser.add_argument("--nrep", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results_wsr_gamma")
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 1:
        parser.error("alpha must lie strictly between 0 and 1.")
    if any(not np.isfinite(r) or not 0 <= r <= 1.5 for r in args.gamma_ratios):
        parser.error("gamma-ratios must be finite and lie in [0,1.5].")
    if any(getattr(args, key) <= 0 for key in ("nrep", "ntrain", "trees")):
        parser.error("nrep, ntrain, and trees must be positive.")
    if args.seed_start < 0 or args.dim < 4:
        parser.error("seed-start must be nonnegative and dim must be at least 4.")
    if not np.isfinite(args.sigma) or args.sigma < 0:
        parser.error("sigma must be finite and nonnegative.")
    for key in ("settings", "sample_sizes", "gamma_ratios"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            parser.error(f"{key.replace('_', '-')} must not contain duplicates.")
    if args.clip is not None and (not np.isfinite(args.clip) or args.clip <= 0):
        parser.error("clip must be finite and positive.")
    for setting in args.settings:
        source = SETTINGS[setting]
        if source in (3, 4) and args.clip is None and (source - 2, args.sigma) not in CLIP_LOOKUP:
            parser.error("Specify --clip for squared-error settings at this sigma.")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    options = {key: getattr(args, key) for key in ("ntrain", "sigma", "dim", "trees", "clip")}
    nmax, mmax = max(n for n, _ in args.sample_sizes), max(m for _, m in args.sample_sizes)
    for setting in args.settings:
        rows = []
        for seed in tqdm(range(args.seed_start, args.seed_start + args.nrep),
                         desc=f"Gamma setting {setting}"):
            rows.extend(run_replication(setting, seed, args.alpha, args.gamma_ratios,
                                         args.sample_sizes, **options))
        frame = pd.DataFrame(rows)
        source = SETTINGS[setting]
        case, dgp = (source - 1) // 2 + 1, (source - 1) % 2 + 1
        metadata = dict(options, source_setting=source, case=case, dgp=dgp,
                        pool_ncalib=nmax, pool_ntest=mmax, tau=10.0,
                        target="WSR", cost_mode="unit", score="loss", reward="constant_one")
        metadata["clip"] = (args.clip if args.clip is not None else CLIP_LOOKUP[(dgp, args.sigma)]) if case == 2 else np.nan
        for key, value in metadata.items():
            frame[key] = value
        for (n, m), group in frame.groupby(["ncalib", "ntest"], sort=False):
            stem = f"gamma_setting_{setting}_n{n}_m{m}_seeds_{args.seed_start}_{args.seed_start + args.nrep - 1}"
            group.to_csv(args.output / f"{stem}.csv", index=False)
    print(f"Saved raw gamma-sweep replications to {args.output.resolve()}")


if __name__ == "__main__":
    main()
