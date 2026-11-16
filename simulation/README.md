# Revision simulations

The MSR experiment reuses the data generators and loss predictors in `wsr.py`.
Both use six panels: excess loss, squared error, and sigmoid loss with unit
costs, followed by sigmoid loss with independent, positively correlated, and
negatively correlated costs. Training, calibration, and test data use separate
seeded streams. Defaults are 1,000 training, 1,000 calibration, 100 test units,
100 random-forest trees, and 100 replications.

Run from the repository root:

```powershell
py -3.9 simulation/msr.py
& 'C:/Program Files/R/R-4.4.1/bin/Rscript.exe' simulation/plot_msr.R
```

The runner saves raw replication CSVs under `simulation/results_msr/`.
The R script produces `msr_comparison.pdf` and its standalone LaTeX source,
with shaded bands of plus/minus two Monte Carlo standard errors. It does not
write summary CSVs. Optional plot arguments are input directory and output file.

Use `--settings`, `--alphas`, `--nrep`, `--seed-start`, `--ntrain`, `--ncalib`,
`--ntest`, `--trees`, or `--output` to customize the run. `--cost-features`
exposes the observed cost alongside X to the loss predictor. The remaining
DGP options match `wsr.py`; see `py -3.9 simulation/msr.py --help`.

## MSR definition and comparisons

The reported realization is `sum(L_test[selected]) / ntest`, including zero
for an empty selection. It estimates marginal selection risk, not loss per
selected unit or per selected cost. Settings 4--6 keep `L=c(X)*B` and the
known loss bound `c(X)`. Cost is part of the loss but never the MSR denominator.

All methods use the same loss-prediction score:

- SCoRE MSR with `gamma=alpha` and bound `Lbar_t=c_test`.
- Binarization at the absolute loss threshold `u=alpha/2`.
- Binarization at the absolute loss threshold `u=alpha/4`.

For the baselines, `L <= u + (c-u)*1{L>u}`. Applying marginal SCoRE to
`(c-u)*1{L>u}` at level `alpha-u`, with exact positive support `c_test-u`,
controls the original MSR: its expectation is bounded by
`u*P(selected) + (alpha-u) <= alpha`. The defaults ensure `c>u`.
With unit costs this equals conformal testing at `(alpha-u)/(1-u)`, without
BH. Unlike the WSR cost panels, binarization thresholds the absolute loss,
since MSR has no cost denominator.

At `gamma=alpha`, every positive marginal SCoRE e-value is at least
`1/alpha`; the marginal rule already selects every positive e-value.
Homogeneous and heterogeneous boosting cannot enlarge that selection, so the
plot has one SCoRE curve rather than three identical curves.

`wsr.py`, `wsr_reward.py`, and `wsr_gamma.py` remain separate entry points;
the new MSR experiment does not change their behavior. Legacy `mdr*.py`
scripts are not used by this experiment.

## WSR temperature and tightness

```powershell
py -3.9 simulation/wsr_temperature.py --jobs 4
& 'C:/Program Files/R/R-4.4.1/bin/Rscript.exe' simulation/plot_wsr_temperature.R
```

This sweep uses only the main-text sigmoid DGP (source setting 6). Panel IDs
`3` and `4` match `wsr.py`: unit costs and independent observed costs
`c(X) = 1 + 3 sigmoid(3 X_9 X_10)`, respectively. Both use
`alpha=gamma=0.2`, temperatures `1 2 5 10 30`, calibration sizes
`100 300 1000`, and `m=100`. It compares all five methods from `wsr.py`.
The training covariates/responses, calibration/test pools, and boosting draws
are paired across temperatures and cost cases. A separate forest is fitted to each
temperature's loss on the same training observations. Smaller calibration
samples use prefixes of a shared maximum-size pool, so changing that maximum
changes the paired data. The forest predicts the base loss `B`; the score is
`c(X) * predicted B`, the loss is `L = c(X) B`, and evaluation divides selected
total loss by selected total cost. Costs are observed, not predicted.
Finite temperatures use interval support `(0,c(X)]`;
the algorithm never substitutes a binary support simply because the loss is
numerically close to zero or one.

Defaults use 100 replications and the same 1,000 training observations and
100-tree forests as the WSR comparison. `--jobs` controls independent
replication workers without changing results. Use `--temperatures`,
`--calib-sizes`, `--ntest`, `--alpha`, `--settings`, and the usual training/run
options to change the configuration; write different configurations to
different `--output` directories.

Raw CSVs go to `results_wsr_temperature/`. The plotter writes
`wsr_temperature_comparison.pdf` and `.tex`, with temperature on a logarithmic
x-axis, realized WSR above fraction selected, and a horizontal target line.
The bands are plus/minus two Monte Carlo standard errors. No summary CSVs
are written. Closeness to the target is an empirical result of this sweep,
not an imposed monotonicity constraint.

## WSR versus concentration bounds

```powershell
py -3.9 simulation/wsr_concentration.py --jobs 4
& 'C:/Program Files/R/R-4.4.1/bin/Rscript.exe' simulation/plot_wsr_concentration.R
```

This separate experiment uses all six main `wsr.py` settings, the same data
and fitted loss scores, and the same SCoRE boosting seeds. The five curves
are SCoRE deterministic/heterogeneous/homogeneous, Hoeffding, and Rademacher.
Defaults are 100 replications, `ntrain=ncalib=1000`, `ntest=100`, 100 trees,
`alpha=0.05,0.10,...,0.50`, `gamma=alpha`, and expectation calibration below.
The remaining DGP options match `wsr.py`; `--jobs`, `--grid-size` (100),
and `--rad-draws` (100) configure the new comparison.

### Converting high-probability control to expectation control (default)

For each real target alpha, choose the levels independently of all data:

`delta = delta_fraction * alpha / Rmax`,
`beta = ((1-expectation_margin)*alpha - delta*Rmax)/(1-delta)`.

Defaults `delta_fraction=0.1`, `expectation_margin=0`, and `Rmax=1` give
`delta=0.1 alpha`, `beta=(alpha-delta)/(1-delta)=0.9 alpha/(1-0.1 alpha)`, and

`(1-delta)*beta + delta*Rmax = alpha`.

Here `Rmax` bounds the **realized normalized risk**: it equals 1 in all six
settings, including cost cases because `L<=c`. The raw loss upper bound 4
still scales the numerator concentration penalty in cost cases. The allocation
is a fixed configurable rule, not an optimized choice; do not tune these
parameters on calibration outcomes without a separate validity argument.
Use `--delta-fraction` to change the allocation; `--expectation-margin` is
optional if a strictly smaller bound is desired. At alpha 0.2, delta is 0.02,
beta is approximately 0.18367347, and the expectation bound is 0.2.
SCoRE continues to use the original alpha and gamma=alpha.

For unit costs, certify `E[L | S<=t] <= beta` with failure probability delta.
Conditional on the training/calibration data and the chosen cutoff, the
selected test losses are i.i.d. from the conditional selected distribution.
Given any positive selection count K, their average has that conditional
mean; an empty selection has risk zero. Thus on the certificate event the
conditional expected batch SDR is at most beta. On its complement it is at
most 1. If the actual failure probability is p<=delta, the expectation is
at most `(1-p)*beta+p <= (1-delta)*beta+delta` because beta<1.

For nonconstant costs, the population ratio below alone does **not** imply
the corresponding finite-batch expectation bound. For example, equally likely
selected types `(c,L)=(4,0)` and `(1,1)` have population ratio 0.2 but expected
WSR 0.5 for a batch of size one. To give a valid conversion, the expected-mode
baselines conservatively certify `E[L | S<=t] <= beta` using the known design
bound `c>=1`. For every nonempty batch,

`sum(selected L)/sum(selected c) <= sum(selected L)/K`.

The same conditional-expectation proof therefore applies, and the bad-event
bound remains 1 because L<=c. This changes the certification denominator to
the selection probability, with denominator values and range bound both 1;
the numerator still uses the actual cost-scaled losses and range bound C.
The plotted metric remains the actual cost-normalized WSR. This conversion
can be more conservative than certifying the population cost ratio.

### Concentration bounds and the optional population-PAC mode

`--calibration pac --delta 0.05` reproduces the previous high-probability
comparison. For a fixed independently trained score, it certifies the
**population ratio**

`R(t) = E[L 1{S<=t}] / E[c 1{S<=t}]`.

Both modes choose the largest candidate cutoff whose numerator upper bound divided
by denominator lower bound is at most beta (alpha in PAC mode), requiring a positive denominator
bound. If no cutoff is certified, they select nothing. The figure evaluates
the same finite-test-batch WSR and fraction selected for every method. In PAC
mode the population certificate does not assert that this realized batch ratio is below
alpha with probability `1-delta`, especially with nonconstant costs. Thus the
methods' theoretical guarantees differ, as discussed in the appendix.

The following formulas describe the PAC-mode denominator. In expected mode,
replace c by 1 and C by 1 in denominator terms only, and use the converted
delta and beta above. Let `C=1` in unit-cost settings and `C=4` in cost settings, using the known
deterministic bound, not a maximum estimated from the data. Both `0<=L<=C`
and `0<c<=C`. Write `Ahat(t)=mean(L 1{S<=t})` and
`Bhat(t)=mean(c 1{S<=t})`.

- Hoeffding uses `K=100` fixed equally spaced cutoffs on `[0,C]`, with
  `h=C sqrt(log(4K/delta)/(2n))`, and bounds `Ahat+h`, `Bhat-h`.
- Rademacher uses all distinct calibration scores as candidates. Its
  complexity is uniform over all thresholds, including the empty selection.
  For `a=L` and `a=c`, estimate
  `Rhat_a = mean_k max(0, max_t sum_i sign_ki a_i 1{S_i<=t})/n`.
  Sorting once and taking cumulative sums at the ends of tied score groups
  evaluates each supremum exactly. There is no threshold discretization.

Unlike an uncorrected Monte Carlo approximation, the Rademacher implementation
accounts for sign-sampling error. With `k=rad_draws`, use

`Rplus_a = Rhat_a + sqrt(2 sum_i a_i^2 log(4/delta)/(k n^2))`.

Changing one of the independent signs changes the Monte Carlo average by
at most `2 a_i/(kn)`, so McDiarmid's inequality gives an upper bound on the
exact empirical complexity with conditional failure probability `delta/4`.
Allocate another `delta/4` to each of the two empirical-process bounds.
Then, with `r=3C sqrt(log(8/delta)/(2n))`, use

`Aupper=Ahat+2 Rplus_L+r`, `Blower=Bhat-2 Rplus_c-r`.

A union bound gives joint failure probability at most delta, including both
calibration and Monte Carlo randomness. The underlying empirical Rademacher
bound is given in [Mohri's lecture, page 6](https://cs.nyu.edu/~mohri/mls/lecture_3.pdf).
These formulas correct the denominator's plus-sign typo in the current
appendix and add the Monte Carlo correction absent from the old experiment;
the manuscript files themselves are unchanged.

Results go to `results_wsr_concentration/`: raw replication CSVs (including
alpha, beta, delta, the expectation budget, certified cutoffs and risk bounds), plus
`wsr_concentration_comparison.pdf` and `.tex`. The plot uses the existing
LaTeX style and shaded plus/minus two Monte Carlo standard errors, with no
summary CSVs. Store runs with different configurations in different output
directories. The original fixed-delta results are archived in
`results_wsr_concentration/population_pac_delta005/`.
Tests: `py -3.9 -m unittest discover -s tests -p test_simulation_wsr_concentration.py -v`.
