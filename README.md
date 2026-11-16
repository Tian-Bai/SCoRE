# SCoRE

[![PyPI package](https://img.shields.io/pypi/v/score-select?label=pypi%20package)](https://pypi.org/project/score-select/)

SCoRE implements conformal selective prediction procedures for marginal
deployment risk (MDR) and selective deployment risk (SDR) control.

This repository also contains the simulation and application code used for the
paper [Conformal Selective Prediction with General Risk Control](https://arxiv.org/abs/2603.24704).

## Installation

```bash
python -m pip install score-select
```

Install the package from a local checkout:

```bash
python -m pip install -e .
```

Install optional dependencies for the research scripts:

```bash
python -m pip install -e ".[experiments]"
```

## Quickstart

```python
import numpy as np
from SCoRE import SCoRE_MDR, SCoRE_SDR

lcalib = np.array([0, 1, 0, 1])
scalib = np.array([0.1, 0.4, 0.2, 0.8])
stest = np.array([0.15, 0.5, 0.9])

dcalib = (lcalib, scalib)
dtest = stest

mdr_selected = SCoRE_MDR(dcalib, dtest, alpha=0.5, gamma=0.5)
sdr_selected = SCoRE_SDR(dcalib, dtest, alpha=0.5, gamma=0.5)
```

Functions return NumPy integer index arrays, so selections can be used directly
to index NumPy arrays.

When using randomized pruning, pass `random_state` for reproducible results:

```python
selected = SCoRE_SDR(
    dcalib,
    dtest,
    alpha=0.5,
    gamma=1.0,
    prune="hete",
    random_state=123,
)
```

## Cost-aware selection and attainable losses

The revised construction controls weighted selection risk:
the expected total selected loss divided by total selected cost, with zero
loss for an empty selection. Supply positive costs with `ccalib` and `ctest`;
omitting them recovers unit-cost SDR. Costs are distinct from the covariate
shift weights `wcalib` and `wtest` in `SCoRE_SDR_w`.

For example, when a false selection wastes its development cost:

```python
ccalib = np.array([1.0, 2.0, 0.5, 3.0])
ctest = np.array([1.5, 2.0, 0.75])

selected, evalues = SCoRE_SDR(
    (ccalib * lcalib, scalib),
    stest,
    alpha=0.2,
    gamma=0.2,
    ccalib=ccalib,
    ctest=ctest,
    finite_t=ctest,  # positive attainable loss is exactly ctest[j]
    return_evals=True,
)
```

Losses are supplied explicitly: passing costs does not multiply losses.
For count normalization of the same cost-scaled losses, omit both cost
arguments while keeping `finite_t=ctest`.

- `Lbar_t`: a scalar or one bound per test unit, for positive attainable losses
  in `(0, Lbar_t[j]]`. A zero bound denotes a surely zero loss.
- `finite_t`: one positive attainable loss per test unit (shape `(m,)`), or
  multiple values per unit (shape `(m, K)`). Pad missing values with `np.nan`;
  an empty or all-NaN row denotes a surely zero loss.
- Supply at most one of these options. The default remains the continuous
  interval `(0, 1]`. For binary losses, use `finite_t=np.ones(len(stest))`.
  Bounds and sets must cover every possible loss given the test features;
  do not infer them from the observed calibration losses alone.
- `block` controls how many test units are processed together (default 512).

The same loss-support options are available for MDR procedures. Positive
`alpha` and `gamma` may exceed one for general loss/cost scales; `gamma=0`
remains accepted for compatibility. Existing positional calls and legacy
`Dtest=(ignored, stest)` inputs remain supported.

The revision takes the infimum over **positive** attainable losses, so exact
zero-loss feasibility boundaries can differ from the earlier implementation.
`SCoRE_MDR_bf` now computes the exact infimum rather than checking only losses
zero and one. Returned e-values include randomized boosting when requested.

For direct access, `score_evalues` computes unboosted e-values using `cc`,
`ct`, `rho_c`, and `rho_t` for costs and shift weights. `cost_ebh(E, c, alpha)`
applies the cost-aware selection rule; equivalently use
`eBH(E, alpha, costs=c)`. `eval_SDR(L, R, selected, costs=c)` returns realized
loss per selected cost as its first output (its binary-power output remains
meaningful only for binary losses).

## Public API

The top-level package exports the main procedures and utilities:

Recommended package entry points:

- `SCoRE_MDR`
- `SCoRE_SDR`

Additional utilities:

- `CS`
- `SCoRE_MDR_bf`, `SCoRE_MDR_w`, `SCoRE_SDR_w`
- `BH`, `eBH`
- `score_evalues`, `cost_ebh`
- `eval_MDR`, `eval_SDR`
- `loss_Jin2023`, `loss_1`, `loss_2`
- `gen_data_Jin2023`, `gen_data_1`, `gen_data_2`
- `Lpredictor`

## Repository Layout

- `SCoRE/`: installable Python package
- `tests/`: package tests
- `applications/`: real-data applications
  - `applications/drug/`: efficient, cost-aware drug discovery
  - `applications/icu/`: clinical prediction error management
  - `applications/llm/`: flexible LLM abstention
- `simulation/`: simulation experiments
- `simulation_w/`: simulation experiments with covariate shifts
